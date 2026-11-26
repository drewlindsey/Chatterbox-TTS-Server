import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_config
import t3_graph
from chatterbox.tts_turbo import ChatterboxTurboTTS
m = ChatterboxTurboTTS.from_pretrained(device="cuda")
t3_graph.slim(m)
m.t3 = m.t3.to(torch.bfloat16)
d = t3_graph.GraphDecoder(m.t3)
d.pos.fill_(300)
def bench(name, fn, n=300):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n): g.replay()
    torch.cuda.synchronize(); print(f"{name}: {(time.perf_counter()-t)/n*1000:.2f} ms/token", flush=True)
bench("eager step in a graph", lambda: (d.pos.fill_(300), d._step()))
t = time.perf_counter()
cl = torch.compile(d._logits, mode="max-autotune-no-cudagraphs", fullgraph=True)
with torch.inference_mode():
    ref = d._logits(); out = cl()
torch.cuda.synchronize()
print(f"compile took {time.perf_counter()-t:.0f}s; max logit diff {(out-ref).abs().max().item():.4f}", flush=True)
d._logits = cl
bench("compiled step in a graph", lambda: (d.pos.fill_(300), d._step()))
print("vram smi peak GiB", round(torch.cuda.max_memory_reserved()/2**30, 2))
