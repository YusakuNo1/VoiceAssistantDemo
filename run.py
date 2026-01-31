import base64
import requests
import mimetypes
import json

def wav_to_data_url(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
        mime_type = 'audio/wav'
    with open(filepath, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"

def call_asr_data_url(file_path):
    data_url = wav_to_data_url(file_path)
    payload = {
        "file": data_url,
        "config": {}
    }
    resp = requests.post(
        "http://127.0.0.1:8081/asr-data-url",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    print("Response:", resp.status_code, resp.text)


def call_app_server_data_url(file_path):
    data_url = wav_to_data_url(file_path)
    payload = {
        "file": data_url,
        "config": {}
    }
    resp = requests.post(
        "http://127.0.0.1:8080/v1/content",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    print("Response:", resp.status_code, resp.text)


if __name__ == "__main__":
    # 请确保你有一个名为 test.wav 的音频文件
    # call_asr_data_url("test.wav")
    call_app_server_data_url("test.wav")