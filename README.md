# Summary

Quick demo for ASR application with Apple Silicon chip as backend

# Steps

## Generate test file for Mac OSX

```console
export VOICE_TEXT="Tell me a very short story for the cats"
say $VOICE_TEXT -o test.aiff
rm test.wav
ffmpeg -i test.aiff test.wav
rm test.aiff
```

## Run Servers & Tests

Run following commands from the current folder but the different terminals.

Terminal 1: Run ASR server
```console
pip install -r requirements.txt
python asr.py
```

Terminal 2 (the same Python environment like Terminal 1): Run App server
```console
python app-server.py
```

Terminal 3 (the same Python environment like Terminal 1): (Optional) Run the test
```console
python run.py
```

Terminal 4: Run the web server
```console
python -m http.server
```

Open browser with http://127.0.0.1:8000/web-app.html

## Architecture

```markdown
+--------------------+         +---------------+         +---------+
| Web Page           |---(1)-->| app-server.py |---(2)-->| asr.py  |
| (Microphone Input) |<--(4)---| (App Logics)  |<--(3)---| (ASR)   |
+--------------------+         +---------------+         +---------+
```
