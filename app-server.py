
import requests
from fastapi import FastAPI, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
import mlx_lm


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
		"http://localhost:8000",
		"http://127.0.0.1:8000",
	],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
# model, tokenizer = mlx_lm.load("mlx-community/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct-bf16")
# model, tokenizer = mlx_lm.load("mlx-community/Qwen3-4B-4bit")


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
	prompt = tokenizer.apply_chat_template(
		chat_history, 
		tokenize=False, 
		add_generation_prompt=True,
		enable_thinking=False,  # Qwen3 specific, it'll add empty <think>\n\n</think> tags to avoid thinking steps
	)

	response = mlx_lm.generate(
		model=model,
		tokenizer=tokenizer,
		prompt=prompt,
		max_tokens=256
	)
	return response["text"] if isinstance(response, dict) else response

# Shared chat logic for both audio and text endpoints
def run_llm_chat(user_content, session_id=None, language=None):
	print("[LLM REQUEST] Session ID: ", session_id)
	if not session_id:
		session_id = str(uuid4())
		print("[LLM REQUEST] Generated new session ID: ", session_id)
	chat_history = session_histories.get(session_id, [])
	if not chat_history or chat_history[0].get("role") != "system":
		if language == "zh":
			system_msg = "You are a helpful assistant."
		else:
			system_msg = "You are a helpful assistant."
		chat_history = [{"role": "system", "content": system_msg}] + chat_history
	chat_history.append({"role": "user", "content": user_content})
	print("[LLM REQUEST] Last user input:", user_content)
	print(f"[LLM REQUEST] Chat history turns: {len(chat_history)}")
	try:
		llm_result = parse_with_llm(chat_history)
	except Exception as e:
		print("[LLM ERROR]", e)
		return {"error": f"LLM error: {e}"}, session_id, chat_history, None
	print("[LLM RESPONSE]", llm_result)
	chat_history.append({"role": "assistant", "content": llm_result})
	print(f"[LLM] Chat history now has {len(chat_history)} turns.")
	session_histories[session_id] = chat_history
	return {"llm": llm_result, "history": chat_history}, session_id, chat_history, llm_result


# In-memory chat history (use database/cache in production)
session_histories = {}


@app.post("/v1/audio")
async def api_v1_audio(request: Request, session_id: str = Cookie(default=None)):

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

	language = asr_result.get("language") if asr_result.get("language") else "en"
	result, session_id, chat_history, llm_result = run_llm_chat(asr_result["text"], session_id, language)
	resp = JSONResponse({
		"asr": asr_result,
		**result
	})
	resp.set_cookie(key="session_id", value=session_id, httponly=True)
	return resp


# New endpoint for direct text input
@app.post("/v1/text")
async def api_v1_text(request: Request, session_id: str = Cookie(default=None)):
	"""
	Receives JSON: {"text": "..."}. Runs LLM and maintains chat history.
	"""
	data = await request.json()
	user_text = data.get("text")
	if not user_text:
		return {"error": "Missing text"}
	# Optionally allow language selection from client, default to English
	language = data.get("language", "en")
	result, session_id, chat_history, llm_result = run_llm_chat(user_text, session_id, language)
	resp = JSONResponse(result)
	resp.set_cookie(key="session_id", value=session_id, httponly=True)
	return resp

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8080)
