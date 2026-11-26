"""Lure Turbo on the same lines, cloning LAB_VOICE. Run with the lure venv's python."""
import argparse, os, subprocess, time
import numpy as np, soundfile as sf, torch
from lure_tts import Lure
import lab_config
ap = argparse.ArgumentParser()
ap.add_argument("--turbo", default="on")
ap.add_argument("--save", default="")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
REF = ("We believe that exploring new ideas and sharing knowledge helps make the world a brighter place for everyone. "
       "Continuous learning and open communication are essential for progress and mutual understanding.")
TEXTS = [
    "Morning! It's going to rain around noon, so grab a jacket.",
    "The garage door has been open for twenty minutes. Want me to close it?",
    "Okay, I added oat milk and eggs to the grocery list.",
    "Your build finished. Two tests failed in the scheduler package, and the rest passed.",
    "Honestly? I think that's a great idea. (chuckle) Let's try it tonight.",
    "Someone is at the front door. It looks like a delivery, and they left a box on the porch.",
]
smi = lambda: subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
t0 = time.perf_counter()
kw = dict(backend="cuda", max_batch=1, max_seq_len=1024)
if a.turbo == "off":
    kw["turbo"] = False
tts = Lure.load(os.path.join(lab_config.MODELS, "lure-turbo"), **kw)
print(f"load {time.perf_counter()-t0:.1f}s turbo={tts.turbo}  smi: {smi()}", flush=True)
voice = tts.make_voice(lab_config.VOICE, REF, name="jade")
for t in TEXTS[:2]:
    tts.speak(t, voice, seed=1)
SR = 24000
tw = ta = 0.0
for i, t in enumerate(TEXTS):
    ws = []
    for r in range(a.reps):
        s = time.perf_counter(); wav = tts.speak(t, voice, seed=42 + i); ws.append(time.perf_counter() - s)
        if r == 0 and a.save:
            os.makedirs(a.save, exist_ok=True); sf.write(f"{a.save}/{i+1}.wav", wav, SR)
    au = len(wav) / SR; mw = sum(ws) / len(ws); tw += mw; ta += au
    print(f"{i+1}: {mw:.2f}s audio {au:.2f}s rtf {mw/au:.2f}", flush=True)
# Time to first audio, streaming, on the longest line.
s = time.perf_counter(); first = None
for pcm in tts.stream(TEXTS[3], voice, seed=7):
    if first is None: first = time.perf_counter() - s
print(f"all rtf={tw/ta:.3f} total={tw:.2f}s  first audio (streamed, line 4)={first:.2f}s")
if a.save:
    d = tts.speak("I'm sorry. (sigh) I never meant for any of this to happen.", voice, seed=3)
    sf.write(f"{a.save}/sigh.wav", d, SR)
    d = tts.speak("You did it! You actually did it!", voice, instruct="Thrilled, almost shouting with joy.", seed=3)
    sf.write(f"{a.save}/direction-thrilled.wav", d, SR)
print(f"smi: {smi()}  torch peak allocated (this process)={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
tts.close()
