import base64
import time
import io
import re
import soundfile as sf
import tempfile
import subprocess
import mlx_whisper
from fastapi import FastAPI, Request


app = FastAPI()
MODEL_ID = "mlx-community/whisper-large-v3-mlx"


@app.post("/asr-data-url")
async def speech_to_text_data_url(request: Request):
    """
    Receives JSON: {"file": data_url, "config": {...}}. Returns ASR result.
    """
    data = await request.json()
    data_url = data.get("file")
    config = data.get("config", {})

    match = re.match(r"data:.*?;base64,(.*)", data_url or "")
    if not match:
        return {"error": "Invalid data url"}
    audio_base64 = match.group(1)
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Base64 decode failed: {e}"}

    start_time = time.perf_counter()
    # Try to read audio directly, if fails, convert with ffmpeg
    try:
        with io.BytesIO(audio_bytes) as audio_file:
            audio_data, samplerate = sf.read(audio_file)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            f_in.write(audio_bytes)
            f_in.flush()
            f_out.flush()
            subprocess.run([
                "ffmpeg", "-y", "-i", f_in.name,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                f_out.name
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with sf.SoundFile(f_out.name) as snd:
                audio_data = snd.read()
                samplerate = snd.samplerate
        import os
        os.unlink(f_in.name)
        os.unlink(f_out.name)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=MODEL_ID)
    print(f"ASR time: {time.perf_counter() - start_time:.2f} seconds")
    return {
        "text": result["text"].strip(),
        "language": result.get("language")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
