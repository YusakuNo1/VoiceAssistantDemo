
import requests
from fastapi import FastAPI, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
import mlx_lm


# MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_ID = "mlx-community/Qwen3-4B-4bit"
# MODEL_ID = "mlx-community/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct-bf16"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def call_asr_server(data_url, config=None):
	payload = {
		"file": data_url,
		"config": config or {}
	}
	resp = requests.post(
		"http://127.0.0.1:8081/asr-data-url",
		json=payload,
		timeout=60
	)
	resp.raise_for_status()
	return resp.json()

def parse_with_llm(chat_history):
	# Convert chat history to LLM prompt
	prompt = ""
	for turn in chat_history:
		if turn["role"] == "user":
			prompt += f"user: {turn['content']}\n"
		else:
			prompt += f"assistant: {turn['content']}\n"
	model, tokenizer = mlx_lm.load(MODEL_ID)
	response = mlx_lm.generate(
		model=model,
		tokenizer=tokenizer,
		prompt=prompt,
		max_tokens=256
	)
	return response["text"] if isinstance(response, dict) else response


# In-memory chat history (use database/cache in production)
session_histories = {}


@app.post("/v1/content")
async def asr_llm(request: Request, session_id: str = Cookie(default=None)):
	"""
	Receives JSON: {"file": data_url, "config": {...}}. Calls ASR, then LLM, and maintains chat history.
	"""
	data = await request.json()
	data_url = data.get("file")
	config = data.get("config", {})
	if not data_url:
		return {"error": "Missing data_url"}

	try:
		asr_result = call_asr_server(data_url, config)
	except Exception as e:
		return {"error": f"ASR server error: {e}"}
	if "text" not in asr_result:
		return {"error": "ASR failed", "detail": asr_result}

	if not session_id:
		session_id = str(uuid4())
	chat_history = session_histories.get(session_id, [])
	if not chat_history or chat_history[0].get("role") != "system":
		language = asr_result.get("language") if asr_result.get("language") else "en"
		if language == "en":
			system_msg = "You are a helpful assistant."
		elif language == "zh":
			system_msg = "你是一个乐于助人的助手。"
		chat_history = [{"role": "system", "content": system_msg}] + chat_history

	chat_history.append({"role": "user", "content": asr_result["text"]})
	try:
		llm_result = parse_with_llm(chat_history)
	except Exception as e:
		return {"error": f"LLM error: {e}"}
	chat_history.append({"role": "assistant", "content": llm_result})
	session_histories[session_id] = chat_history

	resp = JSONResponse({
		"asr": asr_result,
		"llm": llm_result,
		"history": chat_history
	})
	resp.set_cookie(key="session_id", value=session_id, httponly=True)
	return resp



if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8080)
