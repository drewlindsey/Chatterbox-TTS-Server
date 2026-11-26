import torch, transformers
from chatterbox.tts_turbo import ChatterboxTurboTTS
m = ChatterboxTurboTTS.from_pretrained(device="cuda")
g = lambda x: round(x / 2**30, 3)
tf = m.t3.tfmr
print("transformers", transformers.__version__, "torch", torch.__version__)
print("config n_positions", tf.config.n_positions, "attn impl", getattr(tf.config, "_attn_implementation", None))
pb = sum(x.numel()*x.element_size() for x in tf.parameters()); bb = sum(x.numel()*x.element_size() for x in tf.buffers())
print("tfmr params GiB", g(pb), "buffers GiB", g(bb))
for n, b in list(tf.named_buffers())[:3]:
    print("  buffer", n, tuple(b.shape), b.dtype)
print("dtype of params", next(tf.parameters()).dtype)
import inspect, chatterbox.models.t3.t3 as t3mod
src = inspect.getsource(t3mod)
print("text_head used in inference_turbo:", "text_head" in inspect.getsource(m.t3.inference_turbo))
