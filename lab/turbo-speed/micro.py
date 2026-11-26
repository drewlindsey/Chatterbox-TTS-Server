import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_config
import t3_graph
from chatterbox.tts_turbo import ChatterboxTurboTTS
m = ChatterboxTurboTTS.from_pretrained(device="cuda")
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
    torch.cuda.synchronize(); print(f"{name}: {(time.perf_counter()-t)/n*1000:.2f} ms")
logits = torch.zeros(1, d.V, device="cuda")
bench("logits (24 blocks + head)", lambda: logits.copy_(d._logits()))
bench("sample (temp, top-k, top-p, penalty, draw)", lambda: d._sample(logits))
bench("topk 1000 only", lambda: torch.topk(logits, 1000, dim=-1))
bench("sort only", lambda: torch.sort(logits, dim=-1))
x = torch.randn(1,1,1024, device="cuda", dtype=torch.bfloat16)
blk = m.t3.tfmr.h[0]
bench("one block mlp", lambda: blk.mlp(x))
bench("one block c_attn", lambda: blk.attn.c_attn(x))
q = torch.randn(1,16,1,64, device="cuda", dtype=torch.bfloat16)
K = torch.randn(1,16,1536,64, device="cuda", dtype=torch.bfloat16)
mask = (torch.arange(1536, device="cuda") <= 300).view(1,1,1,1536)
bench("sdpa bool mask, 1536", lambda: torch.nn.functional.scaled_dot_product_attention(q, K, K, attn_mask=mask))
bench("sdpa no mask, 301", lambda: torch.nn.functional.scaled_dot_product_attention(q, K[:,:,:301], K[:,:,:301]))
bench("ln_1", lambda: blk.ln_1(x))
bench("speech_head", lambda: m.t3.speech_head(x))
W = blk.mlp.c_fc.weight
WT = W.t().contiguous()
bench("plain matmul 1024x4096", lambda: x.view(1,1024) @ W)
bench("addmm conv1d-style", lambda: torch.addmm(blk.mlp.c_fc.bias, x.view(1,1024), W))
bench("F.linear with W.T contiguous", lambda: torch.nn.functional.linear(x.view(1,1024), WT))
import torch.nn.functional as F
fmask = torch.zeros(1,1,1,1536, device="cuda", dtype=torch.bfloat16).masked_fill(~mask, float("-inf"))
bench("sdpa float mask, 1536", lambda: F.scaled_dot_product_attention(q, K, K, attn_mask=fmask))
def manual():
    s = (q @ K.transpose(-1, -2)) * 0.125
    s = s + fmask
    return s.softmax(-1) @ K
bench("manual attention, 1536", manual)
with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
    bench("sdpa mem-efficient float mask", lambda: F.scaled_dot_product_attention(q, K, K, attn_mask=fmask))
bench("gelu_new (HF act)", lambda: blk.mlp.act(x))
bench("F.gelu tanh", lambda: F.gelu(x, approximate="tanh"))
print("gelu max diff", (blk.mlp.act(x) - F.gelu(x, approximate="tanh")).abs().max().item())
