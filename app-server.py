import requests
import base64
import re
from fastapi import FastAPI, Request
from fastapi import Cookie
from fastapi.middleware.cors import CORSMiddleware
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
	# chat_history: list of {role, content}
	# 拼接为 LLM prompt
	prompt = ""
	for turn in chat_history:
		if turn["role"] == "user":
			prompt += f"用户: {turn['content']}\n"
		else:
			prompt += f"助手: {turn['content']}\n"
	model, tokenizer = mlx_lm.load(MODEL_ID)
	response = mlx_lm.generate(
		model=model,
		tokenizer=tokenizer,
		prompt=prompt,
		max_tokens=256
	)
	return response["text"] if isinstance(response, dict) else response

from fastapi.responses import JSONResponse
from uuid import uuid4

# 简单内存 chat history，生产环境应用数据库或缓存
session_histories = {}

@app.post("/app")
async def asr_llm(request: Request, session_id: str = Cookie(default=None)):
	"""
	接收 JSON: {"file": data_url, "config": {...}}，调用 ASR，再用 LLM 解析，维护 chat history
	"""
	data = await request.json()
	data_url = data.get("file")
	config = data.get("config", {})
	if not data_url:
		return {"error": "Missing data_url"}

	# 可加 data_url 格式校验
	try:
		asr_result = call_asr_server(data_url, config)
	except Exception as e:
		return {"error": f"ASR server error: {e}"}
	if "text" not in asr_result:
		return {"error": "ASR failed", "detail": asr_result}

	# 获取/生成 session_id
	if not session_id:
		session_id = str(uuid4())
	# 获取历史
	chat_history = session_histories.get(session_id, [])
	# 若历史为空，插入 system message
	if not chat_history or chat_history[0].get("role") != "system":
		language = asr_result.get("language") if asr_result.get("language") else "en"
		if language == "en":
			system_msg = "You are a helpful assistant."
		elif language == "zh":
			system_msg = "你是一个乐于助人的助手。"
		chat_history = [{"role": "system", "content": system_msg}] + chat_history

	# 新增用户输入
	chat_history.append({"role": "user", "content": asr_result["text"]})
	try:
		print("ASR result:", asr_result)
		llm_result = parse_with_llm(chat_history)
	except Exception as e:
		return {"error": f"LLM error: {e}"}
	# 新增助手回复
	chat_history.append({"role": "assistant", "content": llm_result})
	# 保存历史
	session_histories[session_id] = chat_history
	
	print("* * * history: ", session_histories[session_id])

	resp = JSONResponse({
		"asr": asr_result,
		"llm": llm_result,
		"history": chat_history
	})
	resp.set_cookie(key="session_id", value=session_id, httponly=True)
	return resp


if __name__ == "__main__":
    import uvicorn
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=8080)
