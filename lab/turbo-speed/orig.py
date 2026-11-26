"""Original (non-Turbo) Chatterbox: time, phases and VRAM on the same lines."""
import argparse, os, subprocess, time, torch, torchaudio
import lab_config
ap = argparse.ArgumentParser()
ap.add_argument("--exaggeration", type=float, default=0.5)
ap.add_argument("--cfg", type=float, default=0.5)
ap.add_argument("--save", default="")
ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()
TEXTS = [
    "Morning! It's going to rain around noon, so grab a jacket.",
    "The garage door has been open for twenty minutes. Want me to close it?",
    "Okay, I added oat milk and eggs to the grocery list.",
    "Your build finished. Two tests failed in the scheduler package, and the rest passed.",
    "Honestly? I think that's a great idea. [chuckle] Let's try it tonight.",
    "Someone is at the front door. It looks like a delivery, and they left a box on the porch.",
]
smi = lambda: subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cuda")
torch.cuda.empty_cache()
print(f"loaded: allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB  smi: {smi()}")
T = {"t3": 0.0, "s3": 0.0}
def timed(k, fn):
    def w(*x, **kw):
        torch.cuda.synchronize(); s = time.perf_counter(); r = fn(*x, **kw); torch.cuda.synchronize(); T[k] += time.perf_counter() - s; return r
    return w
m.t3.inference = timed("t3", m.t3.inference); m.s3gen.inference = timed("s3", m.s3gen.inference)
m.prepare_conditionals(lab_config.VOICE, exaggeration=a.exaggeration)
gen = lambda t: m.generate(t, exaggeration=a.exaggeration, cfg_weight=a.cfg)
torch.manual_seed(1)
for t in TEXTS[:2]: gen(t)
T.update(t3=0.0, s3=0.0); torch.cuda.reset_peak_memory_stats()
tw = ta = 0.0
for i, t in enumerate(TEXTS):
    ws = []
    for r in range(a.reps):
        torch.manual_seed(42 + i); torch.cuda.synchronize(); s = time.perf_counter(); wav = gen(t); torch.cuda.synchronize(); ws.append(time.perf_counter() - s)
        if r == 0 and a.save:
            os.makedirs(a.save, exist_ok=True); torchaudio.save(f"{a.save}/{i+1}.wav", wav.cpu(), m.sr)
    au = wav.shape[-1] / m.sr; mw = sum(ws) / len(ws); tw += mw; ta += au
    print(f"{i+1}: {mw:.2f}s audio {au:.2f}s rtf {mw/au:.2f}", flush=True)
print(f"all rtf={tw/ta:.3f} total={tw:.2f}s  t3={T['t3']/a.reps:.2f}s s3gen={T['s3']/a.reps:.2f}s")
print(f"peak allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB  smi: {smi()}")
