"""Profile Chatterbox Turbo phase by phase: token model (T3), decoder (S3Gen), watermark.

Options:
  --t3-dtype fp32|bf16|fp16   cast T3's weights (no autocast unless --autocast)
  --autocast                  wrap generate in bf16 autocast, as the server's TTS_BF16 does
  --graph                     use the CUDA-graph token loop (t3_graph.py)
  --save DIR                  save each line's first take as WAV
"""
import argparse, os, subprocess, sys, time
import torch, torchaudio
import lab_config

ap = argparse.ArgumentParser()
ap.add_argument("--t3-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
ap.add_argument("--autocast", action="store_true")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--slim", action="store_true")
ap.add_argument("--s3-half", default="", choices=["", "fp16", "bf16"])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--voice", default=lab_config.VOICE)
ap.add_argument("--save", default="")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

TEXTS = [
    "Morning! It's going to rain around noon, so grab a jacket.",
    "The garage door has been open for twenty minutes. Want me to close it?",
    "Okay, I added oat milk and eggs to the grocery list.",
    "Your build finished. Two tests failed in the scheduler package, and the rest passed.",
    "Honestly? I think that's a great idea. [chuckle] Let's try it tonight.",
    "Someone is at the front door. It looks like a delivery, and they left a box on the porch.",
]


def gib(x):
    return f"{x / 2**30:.2f}"


def smi():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out.replace("\n", " | ")


from chatterbox.tts_turbo import ChatterboxTurboTTS

t0 = time.perf_counter()
model = ChatterboxTurboTTS.from_pretrained(device="cuda")
torch.cuda.synchronize()
load_s = time.perf_counter() - t0
if a.t3_dtype != "fp32":
    model.t3 = model.t3.to(torch.bfloat16 if a.t3_dtype == "bf16" else torch.float16)
torch.cuda.empty_cache()
print(f"load {load_s:.1f}s  allocated={gib(torch.cuda.memory_allocated())} GiB "
      f"reserved={gib(torch.cuda.memory_reserved())} GiB  smi: {smi()}")

if a.graph:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t3_graph
    t3_graph.install(model.t3)

if a.slim:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t3_graph as _tg
    freed = _tg.slim(model)
    torch.cuda.empty_cache()
    print(f"slim: dropped {gib(freed)} GiB; allocated={gib(torch.cuda.memory_allocated())} GiB "
          f"reserved={gib(torch.cuda.memory_reserved())} GiB  smi: {smi()}")
model.prepare_conditionals(a.voice)
if a.t3_dtype != "fp32" and not a.autocast:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t3_graph as _tg
    model.conds.t3 = _tg.cast_cond(model.conds.t3, next(model.t3.parameters()).dtype)

# Phase timers around the three stages generate() calls.
T = {"t3": 0.0, "s3": 0.0, "wm": 0.0, "tokens": 0}


def timed(name, fn):
    def wrap(*args, **kw):
        torch.cuda.synchronize(); s = time.perf_counter()
        r = fn(*args, **kw)
        torch.cuda.synchronize(); T[name] += time.perf_counter() - s
        if name == "t3":
            T["tokens"] += r.shape[-1]
        return r
    return wrap


model.t3.inference_turbo = timed("t3", model.t3.inference_turbo)
if a.s3_half:
    _s3_inf = model.s3gen.inference
    _dt = torch.float16 if a.s3_half == "fp16" else torch.bfloat16

    def _s3_half(*args, **kw):
        with torch.autocast("cuda", dtype=_dt):
            wav, x = _s3_inf(*args, **kw)
        return wav.float(), x
    model.s3gen.inference = _s3_half
model.s3gen.inference = timed("s3", model.s3gen.inference)
model.watermarker.apply_watermark = timed("wm", model.watermarker.apply_watermark)

ctx = torch.autocast("cuda", dtype=torch.bfloat16) if a.autocast else torch.autocast("cuda", enabled=False)


def speak(text):
    torch.cuda.synchronize(); s = time.perf_counter()
    with torch.inference_mode(), ctx:
        wav = model.generate(text)
    torch.cuda.synchronize()
    return time.perf_counter() - s, wav


torch.manual_seed(a.seed)
for t in TEXTS[:2]:
    speak(t)
torch.cuda.reset_peak_memory_stats()
for k in T: T[k] = 0 if k == "tokens" else 0.0

tot_wall = tot_audio = 0.0
print("line  mean_s  audio_s  rtf")
for i, text in enumerate(TEXTS):
    walls = []
    for r in range(a.reps):
        torch.manual_seed(a.seed + i)
        wall, wav = speak(text)
        walls.append(wall)
        if r == 0 and a.save:
            os.makedirs(a.save, exist_ok=True)
            torchaudio.save(f"{a.save}/{i + 1}.wav", wav.float().cpu(), model.sr)
    audio = wav.shape[-1] / model.sr
    mean = sum(walls) / len(walls)
    tot_wall += mean; tot_audio += audio
    print(f"{i + 1:>4}  {mean:6.3f}  {audio:7.2f}  {mean / audio:4.2f}")

n = a.reps
print(f"all   total={tot_wall:.2f}s audio={tot_audio:.2f}s rtf={tot_wall / tot_audio:.3f}")
print(f"phases per pass of six lines: t3={T['t3'] / n:.2f}s s3gen={T['s3'] / n:.2f}s "
      f"watermark={T['wm'] / n:.2f}s other={tot_wall - (T['t3'] + T['s3'] + T['wm']) / n:.2f}s")
print(f"t3 tokens/s: {T['tokens'] / T['t3']:.0f}")
print(f"peak allocated={gib(torch.cuda.max_memory_allocated())} GiB "
      f"reserved={gib(torch.cuda.memory_reserved())} GiB  smi: {smi()}")
