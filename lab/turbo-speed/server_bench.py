"""Time the running server: the same lines, fixed seed, wall time vs audio length.

Usage: python3 server_bench.py <label>. Calls the server at TTS_BASE_URL.
Saves each first-rep WAV to $LAB_OUT/server-<label>/ for listening.
"""
import json, os, subprocess, sys, time, urllib.request
import lab_config

LABEL = sys.argv[1]
BASE = lab_config.BASE_URL
H = {"Content-Type": "application/json"}

TEXTS = [
    "Morning! It's going to rain around noon, so grab a jacket.",
    "The garage door has been open for twenty minutes. Want me to close it?",
    "Okay, I added oat milk and eggs to the grocery list.",
    "Your build finished. Two tests failed in the scheduler package, and the rest passed.",
    "Honestly? I think that's a great idea. [chuckle] Let's try it tonight.",
    "Someone is at the front door. It looks like a delivery, and they left a box on the porch.",
]
REPS = 3
OUT = os.path.join(lab_config.OUT, f"server-{LABEL}")
os.makedirs(OUT, exist_ok=True)


def call(path, body=None, method="POST"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=H, method=method)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


info = json.loads(urllib.request.urlopen(BASE + "/api/model-info", timeout=5).read())
if not info.get("loaded"):
    call("/restart_server", {})
    while not json.loads(urllib.request.urlopen(BASE + "/api/model-info", timeout=5).read()).get("loaded"):
        time.sleep(1)


def speak(text):
    t = time.perf_counter()
    wav = call("/v1/audio/speech", {"model": "chatterbox", "input": text, "voice": "Jade.wav",
                                    "response_format": "wav", "seed": 42})
    return time.perf_counter() - t, wav


for t in TEXTS[:2]:  # warm-up
    speak(t)

rows = []
for i, text in enumerate(TEXTS):
    walls = []
    for r in range(REPS):
        wall, wav = speak(text)
        walls.append(wall)
        if r == 0:
            open(f"{OUT}/{i + 1}.wav", "wb").write(wav)
    audio = (len(wav) - 44) / (24000 * 2)
    rows.append((i + 1, min(walls), sum(walls) / REPS, audio))

vram = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print(f"label={LABEL}")
print("line  best_s  mean_s  audio_s  rtf")
for n, best, mean, audio in rows:
    print(f"{n:>4}  {best:6.2f}  {mean:6.2f}  {audio:7.2f}  {mean / audio:4.2f}")
tot_mean = sum(r[2] for r in rows); tot_audio = sum(r[3] for r in rows)
print(f"all   mean_total={tot_mean:.2f}s audio_total={tot_audio:.2f}s rtf={tot_mean / tot_audio:.2f}")
print("gpu processes (pid, MiB):"); print(vram)
