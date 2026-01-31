import base64
import time
import io
import re
import soundfile as sf
import tempfile
import subprocess
import mlx_whisper
from fastapi import FastAPI, UploadFile, File, Request


app = FastAPI()

# 预先定义模型 ID，推荐使用 base 或 small 以兼顾速度与准确度
MODEL_ID = "mlx-community/whisper-large-v3-mlx"


@app.post("/asr-data-url")
async def speech_to_text_data_url(request: Request):
    """
    接收 JSON: {"file": data_url, "config": {...}}，返回识别结果
    """
    data = await request.json()
    data_url = data.get("file")
    config = data.get("config", {})  # 目前不使用 config

    match = re.match(r"data:.*?;base64,(.*)", data_url or "")
    if not match:
        return {"error": "Invalid data url"}
    audio_base64 = match.group(1)
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Base64 decode failed: {e}"}

    start_time = time.perf_counter()
    # 尝试直接读取，如果失败则用 ffmpeg 转换
    try:
        with io.BytesIO(audio_bytes) as audio_file:
            audio_data, samplerate = sf.read(audio_file)
    except Exception:
        # 可能是 webm/ogg 格式，需转为 wav
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            f_in.write(audio_bytes)
            f_in.flush()
            f_out.flush()
            # ffmpeg -y -i input.webm -ar 16000 -ac 1 -sample_fmt s16 output.wav
            subprocess.run([
                "ffmpeg", "-y", "-i", f_in.name,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                f_out.name
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with sf.SoundFile(f_out.name) as snd:
                audio_data = snd.read()
                samplerate = snd.samplerate
        # 清理临时文件
        import os
        os.unlink(f_in.name)
        os.unlink(f_out.name)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=MODEL_ID)
    print(f"识别耗时: {time.perf_counter() - start_time:.2f} 秒")
    return {
        "text": result["text"].strip(),
        "language": result.get("language")
    }


if __name__ == "__main__":
    import uvicorn
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=8081)
