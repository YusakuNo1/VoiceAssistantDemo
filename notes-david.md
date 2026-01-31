# Generate test file for Mac OSX

```console
say "Tell me a very short story for the cats" -o test.aiff
rm test.wav
ffmpeg -i test.aiff test.wav
rm test.aiff
```

# Run ASR server

Terminal 1: Run ASR server
```console
python asr.py
```

Terminal 2: Run App server
```console
python app-server.py
```

Terminal 3: Run the test
```console
python run.py
```

Terminal 4: Run the web server
```console
python -m http.server
```

Open browser with http://127.0.0.1:8000/web-app.html
