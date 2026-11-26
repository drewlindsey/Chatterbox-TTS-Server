import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
m = ChatterboxTurboTTS.from_pretrained(device="cuda")
torch.cuda.empty_cache()
g = lambda x: round(x / 2**30, 3)
print("after load allocated GiB", g(torch.cuda.memory_allocated()), "reserved", g(torch.cuda.memory_reserved()))
def size(mod):
    return sum(x.numel()*x.element_size() for x in mod.parameters()) + sum(x.numel()*x.element_size() for x in mod.buffers())
for name in ["t3", "s3gen", "ve"]:
    mod = getattr(m, name, None)
    if mod is not None:
        print(name, "params M", round(sum(x.numel() for x in mod.parameters())/1e6, 1), "GiB", g(size(mod)))
        for n, c in mod.named_children():
            print("   ", name + "." + n, g(size(c)))
