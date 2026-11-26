"""Exercise the server's endpoints: /tts, /v1/audio/speech, unload and reload, with VRAM after each."""
import json, os, subprocess, time, urllib.request
import lab_config
B = lab_config.BASE_URL
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter(); d = urllib.request.urlopen(r, timeout=300).read(); return time.perf_counter() - t, d
smi = lambda: subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
long = ("Okay, here's the plan for tomorrow. (Not literally parenthetical.) First, the plumber comes at nine, so someone "
        "needs to be home. Then the grocery pickup is at eleven thirty. [sigh] After that, I'd suggest a quiet afternoon, "
        "because the evening is busy: dinner with friends at seven, and the movie after. Honestly, it's a lot. [laugh] "
        "But we've done busier days than this, and we'll be fine.")
t, d = post("/tts", {"text": "Hey! Abigail here. [chuckle] Testing the custom endpoint.", "voice_mode": "predefined",
                     "predefined_voice_id": "Abigail.wav", "output_format": "wav"})
print(f"/tts Abigail: {t:.2f}s {len(d)} bytes")
t, d = post("/tts", {"text": long, "voice_mode": "predefined", "predefined_voice_id": "Connor.wav", "output_format": "wav",
                     "split_text": True, "chunk_size": 120})
print(f"/tts Connor, long, chunked: {t:.2f}s audio {(len(d)-44)/48000:.1f}s")
t, d = post("/v1/audio/speech", {"model": "x", "input": long, "voice": "Jade.wav", "response_format": "wav"})
print(f"/v1 Jade, long unchunked: {t:.2f}s audio {(len(d)-44)/48000:.1f}s")
os.makedirs(lab_config.OUT, exist_ok=True)
open(os.path.join(lab_config.OUT, "long-jade.wav"), "wb").write(d)
print("before unload:", smi())
post("/api/unload", {}); time.sleep(3)
print("after unload:", smi(), json.loads(urllib.request.urlopen(B + "/api/model-info").read())["loaded"])
t, _ = post("/restart_server", {})
while not json.loads(urllib.request.urlopen(B + "/api/model-info").read())["loaded"]:
    time.sleep(0.5)
print(f"reload {t:.1f}s:", smi())
t, d = post("/v1/audio/speech", {"model": "x", "input": "Back again after the reload.", "voice": "Jade.wav", "response_format": "wav"})
print(f"after reload: {t:.2f}s audio {(len(d)-44)/48000:.1f}s")
