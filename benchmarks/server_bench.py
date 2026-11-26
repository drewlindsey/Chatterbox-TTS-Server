"""Time a running server end to end: fixed lines, fixed seed, wall time against audio length.

Sends each line to /v1/audio/speech `--reps` times after two warm-up requests, and prints the
mean wall time, the audio's length and their ratio (real-time factor) per line and overall.
Uses only the standard library, so it runs from any machine that can reach the server.

  python benchmarks/server_bench.py [--url URL] [--voice Jade.wav] [--reps 3] [--out DIR] [--vram]

TTS_BASE_URL sets the default URL (http://127.0.0.1:8004). TTS_API_KEY, if set, is sent as
`Authorization: Bearer <key>` for servers started with TTS_API_KEYS_FILE. --out saves each line's
first take as WAV, for listening. --vram prints this machine's GPU processes after the run, so it
only describes the server when run on the same host.
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

LINES = [
    "Morning! It's going to rain around noon, so grab a jacket.",
    "The garage door has been open for twenty minutes. Want me to close it?",
    "Okay, I added oat milk and eggs to the grocery list.",
    "Your build finished. Two tests failed in the scheduler package, and the rest passed.",
    "Honestly? I think that's a great idea. [chuckle] Let's try it tonight.",
    "Someone is at the front door. It looks like a delivery, and they left a box on the porch.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8004"))
    ap.add_argument("--voice", default="Jade.wav", help="predefined voice or reference file name")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="directory for each line's first take")
    ap.add_argument("--vram", action="store_true", help="print local nvidia-smi GPU processes")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("TTS_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def model_info():
        with urllib.request.urlopen(base + "/api/model-info", timeout=10) as r:
            return json.loads(r.read())

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=headers,
                                     method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def speak(text):
        start = time.perf_counter()
        data = post("/v1/audio/speech", {"model": "chatterbox", "input": text, "voice": a.voice,
                                         "response_format": "wav", "seed": a.seed})
        wall = time.perf_counter() - start
        with wave.open(io.BytesIO(data)) as w:
            audio = w.getnframes() / w.getframerate()
        return wall, audio, data

    if not model_info().get("loaded"):
        print("model not loaded; asking the server to load it", file=sys.stderr)
        post("/restart_server", {})
        while not model_info().get("loaded"):
            time.sleep(1)

    for text in LINES[:2]:
        speak(text)  # warm-up

    if a.out:
        os.makedirs(a.out, exist_ok=True)
    total_wall = total_audio = 0.0
    print("line  best_s  mean_s  audio_s   rtf")
    for i, text in enumerate(LINES, 1):
        walls = []
        for rep in range(a.reps):
            wall, audio, data = speak(text)
            walls.append(wall)
            if rep == 0 and a.out:
                with open(os.path.join(a.out, f"{i}.wav"), "wb") as f:
                    f.write(data)
        mean = sum(walls) / len(walls)
        total_wall += mean
        total_audio += audio
        print(f"{i:>4}  {min(walls):6.2f}  {mean:6.2f}  {audio:7.2f}  {mean / audio:4.2f}")
    print(f" all          {total_wall:6.2f}  {total_audio:7.2f}  {total_wall / total_audio:4.2f}")

    if a.vram:
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout
            print("local GPU processes (pid, memory):")
            print(out.strip() or "  none")
        else:
            print("--vram: nvidia-smi not found on this machine", file=sys.stderr)


if __name__ == "__main__":
    main()
