"""Faster, lighter Chatterbox Turbo on CUDA (this fork). engine.py applies it behind three switches.

TTS_T3_GRAPH=on: a CUDA-graph token loop for Turbo's T3 (GPT-2 Medium). The stock loop runs one
  Hugging Face forward per token from Python, with a growing KV cache and two GPU-to-CPU waits per
  token, so on a fast card the GPU mostly waits on Python. Here one decode step (embedding, the
  blocks, the speech head and Turbo's sampling, in stock order: temperature, top-k, top-p,
  repetition penalty) runs on a preallocated cache, is captured once as a CUDA graph and replayed
  per token. The prompt and the first token still go through the stock code. Attention is written
  out by hand: PyTorch's SDPA with a mask over the whole cache was 4x slower per layer.
  Sampling draws with the exponential trick (argmax of p / Exp(1)), which has multinomial's
  distribution but a different random stream, so a seed doesn't reproduce the stock loop's audio.
  At load the step is checked against Hugging Face's forward; a mismatch, a prompt too long for
  the cache, a top_k other than the captured one, or any error falls back to the stock loop.

TTS_SLIM=on: frees VRAM Turbo's generation never uses: GPT-2's per-layer causal-mask buffers
  (n_positions^2 bools each, 1.5 GiB for Turbo; only its eager attention reads them, and the model
  runs SDPA), T3's text_head, and S3Gen's speech tokenizer, which only prepares new voices: it
  waits in RAM and visits the GPU while prepare_conditionals runs.

TTS_BF16_AUTOCAST=off (with TTS_BF16 on): bf16 weights without autocast. Autocast's per-op casts
  made Turbo slower than float32; the voice conditioning is cast to bf16 instead.

For Turbo on a 16 GB consumer GPU, single sentences at batch 1: stock float32 took 0.23 of the
audio's length and 4.9 GB of VRAM; all three switches on, 0.12 and 2.4 GB.
"""
import dataclasses
import logging
import math

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper)

logger = logging.getLogger(__name__)

STOP_CHECK_EVERY = 16
MAX_LEN = 1536          # cache slots: prompt (about 220 tokens for a sentence) + up to 1000 generated
CHECK_TOLERANCE = 0.05  # worst relative logit difference against Hugging Face; bf16 rounding is ~0.012


def cast_cond(cond, dtype):
    """A T3Cond with its float tensors in `dtype`."""
    kw = {}
    for f in dataclasses.fields(cond):
        v = getattr(cond, f.name)
        kw[f.name] = v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
    return type(cond)(**kw)


def _cond_in(cond, dtype):
    sp = getattr(cond, "speaker_emb", None)
    return cond if not torch.is_tensor(sp) or sp.dtype == dtype else cast_cond(cond, dtype)


class GraphDecoder:
    def __init__(self, t3, top_k=1000, max_len=MAX_LEN):
        self.t3 = t3
        tf = t3.tfmr
        cfg = tf.config
        if not cfg.scale_attn_weights or cfg.scale_attn_by_inverse_layer_idx or cfg.reorder_and_upcast_attn:
            raise ValueError("unsupported GPT-2 attention settings")
        p = next(tf.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.L, self.H, self.E = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.D = self.E // self.H
        self.scale = 1.0 / math.sqrt(self.D)
        self.max_len = max_len
        self.V = t3.speech_head.out_features
        self.top_k = min(top_k, self.V)
        self.stop_token = t3.hp.stop_speech_token

        z = dict(device=self.dev)
        self.k = torch.zeros(self.L, 1, self.H, max_len, self.D, dtype=self.dtype, **z)
        self.v = torch.zeros_like(self.k)
        self.tok = torch.zeros(1, 1, dtype=torch.long, **z)        # current token, in and out
        self.pos = torch.zeros(1, dtype=torch.long, **z)            # its position
        self.seen = torch.zeros(1, self.V, dtype=torch.bool, **z)   # tokens generated so far
        self.out = torch.zeros(max_len, dtype=torch.long, **z)      # tokens the graph wrote
        self.n = torch.zeros(1, dtype=torch.long, **z)
        self.stopped = torch.zeros(1, dtype=torch.bool, **z)
        self.ar = torch.arange(max_len, **z)
        # Sampling settings are inputs, so a request's values don't need a new graph.
        self.temp = torch.ones(1, **z)
        self.top_p = torch.ones(1, **z)
        self.rep = torch.ones(1, **z)
        self.graph = None

    def _logits(self):
        tf = self.t3.tfmr
        x = self.t3.speech_emb(self.tok) + tf.wpe(self.pos).unsqueeze(0)          # (1, 1, E)
        mask = torch.zeros(1, 1, 1, self.max_len, device=self.dev).masked_fill(self.ar > self.pos, -float("inf"))
        for i, blk in enumerate(tf.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(self.E, dim=2)
            q = q.view(1, 1, self.H, self.D).transpose(1, 2)
            self.k[i].index_copy_(2, self.pos, k.view(1, 1, self.H, self.D).transpose(1, 2))
            self.v[i].index_copy_(2, self.pos, v.view(1, 1, self.H, self.D).transpose(1, 2))
            s = (q @ self.k[i].transpose(-1, -2)).float() * self.scale + mask
            a = (s.softmax(dim=-1).to(self.dtype) @ self.v[i]).transpose(1, 2).reshape(1, 1, self.E)
            x = x + blk.attn.c_proj(a)
            x = x + blk.mlp(blk.ln_2(x))
        return self.t3.speech_head(tf.ln_f(x))[:, -1, :].float()                    # (1, V)

    def _sample(self, s):
        s = s / self.temp
        kth = torch.topk(s, self.top_k, dim=-1).values[..., -1:]
        s = s.masked_fill(s < kth, -float("inf"))
        srt, idx = torch.sort(s, descending=False, dim=-1)
        rm = srt.softmax(dim=-1).cumsum(dim=-1) <= (1 - self.top_p)
        rm[..., -1:] = False
        s = s.masked_fill(rm.scatter(1, idx, rm), -float("inf"))
        s = torch.where(self.seen, torch.where(s < 0, s * self.rep, s / self.rep), s)
        probs = s.softmax(dim=-1)
        return (probs / torch.empty_like(probs).exponential_(1.0)).argmax(dim=-1, keepdim=True)

    def _step(self):
        nxt = self._sample(self._logits())
        self.tok.copy_(nxt)
        self.seen.scatter_(1, nxt, True)
        self.out.index_copy_(0, self.n, nxt.view(1))
        self.stopped |= nxt.view(1) == self.stop_token
        self.pos += 1
        self.n += 1

    @torch.inference_mode()
    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.pos.fill_(0); self.n.fill_(0)
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        self.pos.fill_(0); self.n.fill_(0)
        with torch.cuda.graph(self.graph):
            self._step()

    def _prefill(self, embeds):
        o = self.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past = o.past_key_values
        P = embeds.shape[1]
        for i in range(self.L):
            kk, vv = past[i] if isinstance(past, (tuple, list)) else (past.key_cache[i], past.value_cache[i])
            self.k[i, :, :, :P].copy_(kk)
            self.v[i, :, :, :P].copy_(vv)
        return o

    @torch.inference_mode()
    def generate(self, t3_cond, text_tokens, temperature, top_p, repetition_penalty, max_gen_len):
        """Speech tokens like stock inference_turbo, or None when this request doesn't fit the cache."""
        t3 = self.t3
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                             speech_tokens=start, cfg_weight=0.0)
        P = embeds.shape[1]
        if text_tokens.shape[0] != 1 or P + max_gen_len + STOP_CHECK_EVERY > self.max_len:
            return None
        o = self._prefill(embeds)
        # The first token exactly as stock does it.
        procs = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature))
        procs.append(TopKLogitsWarper(self.top_k))
        if top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        logits = t3.speech_head(o[0][:, -1:])[:, -1, :]
        first = torch.multinomial(F.softmax(procs(start, logits), dim=-1), num_samples=1)
        if bool(first == self.stop_token):
            return first[:, :0].clone()
        self.temp.fill_(temperature if temperature > 0 else 1.0)
        self.top_p.fill_(min(top_p, 1.0))
        self.rep.fill_(repetition_penalty)
        self.tok.copy_(first)
        self.seen.zero_(); self.seen.scatter_(1, first, True)
        self.pos.fill_(P); self.n.fill_(0); self.stopped.zero_()
        steps = 0
        while steps < max_gen_len and not bool(self.stopped):
            for _ in range(STOP_CHECK_EVERY):
                self.graph.replay()
            steps += STOP_CHECK_EVERY
        gen = self.out[: int(self.n)]
        hits = (gen == self.stop_token).nonzero()
        if hits.numel():
            gen = gen[: int(hits[0])]
        return torch.cat([first.view(1), gen[:max_gen_len]]).view(1, -1).clone()

    @torch.inference_mode()
    def check(self, t3_cond, text_tokens, n=32):
        """Teacher-forced: the same tokens through Hugging Face and through our step; worst relative
        logit difference."""
        t3 = self.t3
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
        P = embeds.shape[1]
        past = self._prefill(embeds).past_key_values
        g = torch.Generator(device=self.dev).manual_seed(0)
        worst = 0.0
        for j in range(n):
            tok = torch.randint(0, self.stop_token, (1, 1), device=self.dev, generator=g)
            hf = t3.tfmr(inputs_embeds=t3.speech_emb(tok), past_key_values=past, use_cache=True)
            past = hf.past_key_values
            ref = t3.speech_head(hf[0])[:, -1, :].float()
            self.tok.copy_(tok); self.pos.fill_(P + j)
            worst = max(worst, ((self._logits() - ref).abs().max() / ref.abs().max()).item())
        return worst


def cast_inputs(t3):
    """With T3 in bf16 and no autocast, cast the voice conditioning on the way in."""
    stock = t3.inference_turbo
    dtype = next(t3.parameters()).dtype

    def inference_turbo(t3_cond, text_tokens, *args, **kw):
        return stock(_cond_in(t3_cond, dtype), text_tokens, *args, **kw)

    t3.inference_turbo = inference_turbo


def install_graph(model, check_text="Honestly? I think that's a great idea. [chuckle] Let's try it tonight."):
    """Replace model.t3.inference_turbo with the graph loop, after checking it against Hugging Face
    on the model's built-in voice. Returns the worst logit difference, or raises (caller keeps stock)."""
    t3 = model.t3
    stock = t3.inference_turbo
    dec = GraphDecoder(t3)
    if model.conds is None:
        raise RuntimeError("no built-in voice to check the graph against")
    tokens = model.tokenizer(check_text, return_tensors="pt").input_ids.to(dec.dev)
    worst = dec.check(_cond_in(model.conds.t3, dec.dtype), tokens)
    if not worst < CHECK_TOLERANCE:
        raise RuntimeError(f"graph step differs from Hugging Face by {worst:.4f}")
    dec.capture()

    def inference_turbo(t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95,
                        repetition_penalty=1.2, max_gen_len=1000):
        cond = _cond_in(t3_cond, dec.dtype)
        if min(top_k, dec.V) == dec.top_k:
            try:
                r = dec.generate(cond, text_tokens, temperature, top_p, repetition_penalty, max_gen_len)
                if r is not None:
                    return r
            except Exception:
                logger.exception("T3 graph loop failed; this request uses the stock loop")
        return stock(cond, text_tokens, temperature, top_k, top_p, repetition_penalty, max_gen_len)

    t3.inference_turbo = inference_turbo
    t3._graph_decoder = dec
    return worst


def slim(model):
    """Free VRAM Turbo's generation never uses (see the module docstring). Returns bytes freed."""
    freed = 0
    for blk in model.t3.tfmr.h:
        b = getattr(blk.attn, "bias", None)
        if torch.is_tensor(b) and b.numel() > 1:
            freed += b.numel() * b.element_size()
            # A stand-in, so an eager-attention call fails loudly instead of silently.
            blk.attn.bias = torch.zeros(1, 1, 1, 1, dtype=torch.bool, device=b.device)
    if getattr(model.t3, "text_head", None) is not None:
        freed += sum(p.numel() * p.element_size() for p in model.t3.text_head.parameters())
        model.t3.text_head = None
    s3 = model.s3gen
    tok = s3.tokenizer
    freed += sum(p.numel() * p.element_size() for p in list(tok.parameters()) + list(tok.buffers()))
    # S3Gen takes its device from the tokenizer's weights; pin it (and dtype) to the decoder's.
    flow_p = next(s3.flow.parameters())
    s3.__class__ = type("SlimS3Gen", (type(s3),), {
        "device": property(lambda self: flow_p.device),
        "dtype": property(lambda self: flow_p.dtype),
    })
    device = flow_p.device
    tok.to("cpu")
    stock_prepare = model.prepare_conditionals

    def prepare_conditionals(*args, **kw):
        tok.to(device)
        try:
            return stock_prepare(*args, **kw)
        finally:
            tok.to("cpu")
            torch.cuda.empty_cache()

    model.prepare_conditionals = prepare_conditionals
    torch.cuda.empty_cache()
    return freed
