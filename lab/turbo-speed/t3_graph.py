"""A CUDA-graph token loop for Chatterbox Turbo's T3 (GPT-2 Medium).

Stock `inference_turbo` runs one Hugging Face forward per token from Python, with a growing KV
cache and two GPU-to-CPU waits per token. Here one decode step (embedding, 24 blocks, the speech
head and Turbo's sampling, in stock order: temperature, top-k, top-p, repetition penalty) runs on
a preallocated cache, is captured once as a CUDA graph, and is replayed per token. The prompt
(prefill) and the first token still go through the stock code, so they match it exactly.

Sampling draws with the exponential trick (argmax of p / Exp(1)), which has the same distribution
as torch.multinomial but a different random stream, so a seed doesn't reproduce stock's audio.

install(t3) replaces t3.inference_turbo. check(t3, ...) compares our step's logits with HF's.
"""
import math
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper)

STOP_CHECK_EVERY = 16


class GraphDecoder:
    def __init__(self, t3, max_len=1536, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2):
        self.t3 = t3
        tf = t3.tfmr
        cfg = tf.config
        p = next(tf.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.L = cfg.n_layer
        self.H = cfg.n_head
        self.D = cfg.n_embd // cfg.n_head
        self.E = cfg.n_embd
        self.max_len = max_len
        self.V = t3.speech_head.out_features
        self.temperature, self.top_k, self.top_p, self.rep = temperature, min(top_k, self.V), top_p, repetition_penalty
        self.stop_token = t3.hp.stop_speech_token
        self.scale = 1.0 / math.sqrt(self.D)
        assert cfg.scale_attn_weights and not cfg.scale_attn_by_inverse_layer_idx and not cfg.reorder_and_upcast_attn

        z = dict(device=self.dev)
        self.k = torch.zeros(self.L, 1, self.H, max_len, self.D, dtype=self.dtype, **z)
        self.v = torch.zeros_like(self.k)
        self.tok = torch.zeros(1, 1, dtype=torch.long, **z)        # current token, in and out
        self.pos = torch.zeros(1, dtype=torch.long, **z)            # its position
        self.seen = torch.zeros(1, self.V, dtype=torch.bool, **z)   # tokens generated so far
        self.out = torch.zeros(max_len, dtype=torch.long, **z)      # tokens written by the graph
        self.n = torch.zeros(1, dtype=torch.long, **z)              # how many
        self.stopped = torch.zeros(1, dtype=torch.bool, **z)
        self.ar = torch.arange(max_len, **z)
        self.graph = None

    # One decode step, written so it can be captured.
    def _logits(self):
        tf = self.t3.tfmr
        x = self.t3.speech_emb(self.tok) + tf.wpe(self.pos).unsqueeze(0)       # (1, 1, E)
        # Attention by hand: PyTorch's SDPA with a mask over the whole cache was 4x slower here.
        # Scores and softmax in float32, then back to the weights' dtype.
        mask = torch.zeros(1, 1, 1, self.max_len, device=self.dev).masked_fill(self.ar > self.pos, -float("inf"))
        for i, blk in enumerate(tf.h):
            h = blk.ln_1(x)
            q, k, v = blk.attn.c_attn(h).split(self.E, dim=2)
            q = q.view(1, 1, self.H, self.D).transpose(1, 2)
            k = k.view(1, 1, self.H, self.D).transpose(1, 2)
            v = v.view(1, 1, self.H, self.D).transpose(1, 2)
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            s = (q @ self.k[i].transpose(-1, -2)).float() * self.scale + mask
            a = s.softmax(dim=-1).to(self.dtype) @ self.v[i]
            a = a.transpose(1, 2).reshape(1, 1, self.E)
            x = x + blk.attn.c_proj(a)
            x = x + blk.mlp(blk.ln_2(x))
        x = tf.ln_f(x)
        return self.t3.speech_head(x)[:, -1, :].float()                      # (1, V)

    def _sample(self, s):
        s = s / self.temperature
        kth = torch.topk(s, self.top_k, dim=-1).values[..., -1:]
        s = s.masked_fill(s < kth, -float("inf"))
        srt, idx = torch.sort(s, descending=False, dim=-1)
        cum = srt.softmax(dim=-1).cumsum(dim=-1)
        rm = cum <= (1 - self.top_p)
        rm[..., -1:] = False
        s = s.masked_fill(rm.scatter(1, idx, rm), -float("inf"))
        pen = torch.where(s < 0, s * self.rep, s / self.rep)
        s = torch.where(self.seen, pen, s)
        probs = s.softmax(dim=-1)
        q = torch.empty_like(probs).exponential_(1.0)
        return (probs / q).argmax(dim=-1, keepdim=True)                       # (1, 1)

    def _step(self):
        nxt = self._sample(self._logits())
        self.tok.copy_(nxt)
        self.seen.scatter_(1, nxt, True)
        self.out.index_copy_(0, self.n, nxt.view(1))
        self.stopped |= (nxt.view(1) == self.stop_token)
        self.pos += 1
        self.n += 1

    def capture(self):
        # Warm up on a side stream, then record one step.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            self.pos.fill_(0); self.n.fill_(0)
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            self.pos.fill_(0); self.n.fill_(0)
            with torch.cuda.graph(self.graph):
                self._step()

    @torch.inference_mode()
    def generate(self, t3_cond, text_tokens, max_gen_len=1000):
        t3 = self.t3
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                             speech_tokens=start, cfg_weight=0.0)
        P = embeds.shape[1]
        if P + max_gen_len + STOP_CHECK_EVERY > self.max_len:
            return None  # too long for the cache: the caller falls back to stock
        o = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past = o.past_key_values
        for i in range(self.L):
            kk, vv = past[i] if isinstance(past, (tuple, list)) else (past.key_cache[i], past.value_cache[i])
            self.k[i, :, :, :P].copy_(kk)
            self.v[i, :, :, :P].copy_(vv)
        # The first token exactly as stock does it: HF processors, input_ids = the start token.
        procs = LogitsProcessorList([TemperatureLogitsWarper(self.temperature), TopKLogitsWarper(self.top_k),
                                     TopPLogitsWarper(self.top_p), RepetitionPenaltyLogitsProcessor(self.rep)])
        logits = t3.speech_head(o[0][:, -1:])[:, -1, :]
        first = torch.multinomial(F.softmax(procs(start, logits), dim=-1), num_samples=1)
        self.tok.copy_(first)
        self.seen.zero_(); self.seen.scatter_(1, first, True)
        self.pos.fill_(P)
        self.n.fill_(0)
        self.stopped.zero_()
        steps = 0
        while steps < max_gen_len:
            for _ in range(STOP_CHECK_EVERY):
                self.graph.replay()
            steps += STOP_CHECK_EVERY
            if bool(self.stopped):
                break
        if bool(first == self.stop_token):
            gen = torch.empty(0, dtype=torch.long, device=self.dev)
        else:
            gen = self.out[: int(self.n)]
            hits = (gen == self.stop_token).nonzero()
            if hits.numel():
                gen = gen[: int(hits[0])]
            gen = gen[:max_gen_len]
        return torch.cat([first.view(1), gen]).view(1, -1).clone()


def install(t3, **kw):
    dec = GraphDecoder(t3, **kw)
    dec.capture()
    stock = t3.inference_turbo

    def inference_turbo(t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95,
                        repetition_penalty=1.2, max_gen_len=1000):
        if (temperature, min(top_k, dec.V), top_p, repetition_penalty) != (dec.temperature, dec.top_k, dec.top_p, dec.rep):
            return stock(t3_cond, text_tokens, temperature, top_k, top_p, repetition_penalty, max_gen_len)
        r = dec.generate(t3_cond, text_tokens, max_gen_len)
        return stock(t3_cond, text_tokens, temperature, top_k, top_p, repetition_penalty, max_gen_len) if r is None else r

    t3.inference_turbo = inference_turbo
    t3._graph_decoder = dec
    return dec


@torch.inference_mode()
def check(t3, t3_cond, text_tokens, n=32):
    """Teacher-forced parity: feed the same tokens to HF and to our step, compare logits."""
    dec = GraphDecoder(t3)
    start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
    P = embeds.shape[1]
    o = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past = o.past_key_values
    for i in range(dec.L):
        kk, vv = past[i] if isinstance(past, (tuple, list)) else (past.key_cache[i], past.value_cache[i])
        dec.k[i, :, :, :P].copy_(kk); dec.v[i, :, :, :P].copy_(vv)
    g = torch.Generator(device=dec.dev).manual_seed(0)
    worst = 0.0
    for j in range(n):
        tok = torch.randint(0, 6561, (1, 1), device=dec.dev, generator=g)
        hf = t3.tfmr(inputs_embeds=t3.speech_emb(tok), past_key_values=past, use_cache=True)
        past = hf.past_key_values
        ref = t3.speech_head(hf[0])[:, -1, :].float()
        dec.tok.copy_(tok); dec.pos.fill_(P + j)
        ours = dec._logits()
        worst = max(worst, ((ours - ref).abs().max() / ref.abs().max()).item())
    return worst


def cast_cond(cond, dtype):
    """T3Cond with its float tensors in `dtype`, for a T3 cast without autocast."""
    import dataclasses
    kw = {}
    for f in dataclasses.fields(cond):
        v = getattr(cond, f.name)
        kw[f.name] = v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
    return type(cond)(**kw)


def slim(model):
    """Free VRAM Turbo's generation never uses. Returns bytes freed.

    - GPT-2's per-layer causal-mask buffers (n_positions^2 bools each, 1.5 GiB in all): only its
      eager attention reads them, and the model runs SDPA, which builds its own mask. A tiny
      stand-in makes any eager call fail loudly instead of silently.
    - T3's text_head: Turbo's inference never calls it.
    - S3Gen's speech tokenizer: only needed to prepare a new voice, and the server caches prepared
      voices. It waits in RAM and visits the GPU while prepare_conditionals runs.
    """
    freed = 0
    for blk in model.t3.tfmr.h:
        b = blk.attn.bias
        freed += b.numel() * b.element_size()
        blk.attn.bias = torch.zeros(1, 1, 1, 1, dtype=torch.bool, device=b.device)
    th = model.t3.text_head
    freed += sum(p.numel() * p.element_size() for p in th.parameters())
    model.t3.text_head = None
    tok = model.s3gen.tokenizer
    freed += sum(p.numel() * p.element_size() for p in list(tok.parameters()) + list(tok.buffers()))
    # S3Gen takes its device and dtype from the tokenizer's weights; pin them to the decoder's.
    s3 = model.s3gen
    flow_p = next(s3.flow.parameters())
    s3.__class__ = type("SlimS3Gen", (type(s3),), {
        "device": property(lambda self: flow_p.device),
        "dtype": property(lambda self: flow_p.dtype),
    })
    tok.to("cpu")
    stock_prepare = model.prepare_conditionals

    def prepare_conditionals(*args, **kw):
        tok.to(model.device)
        try:
            return stock_prepare(*args, **kw)
        finally:
            tok.to("cpu")
            torch.cuda.empty_cache()

    model.prepare_conditionals = prepare_conditionals
    return freed
