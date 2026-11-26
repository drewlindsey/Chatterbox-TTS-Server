import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_config
import t3_graph
from chatterbox.tts_turbo import ChatterboxTurboTTS
m = ChatterboxTurboTTS.from_pretrained(device="cuda")
m.prepare_conditionals(lab_config.VOICE)
tt = m.tokenizer("Honestly? I think that's a great idea. [chuckle] Let's try it tonight.", return_tensors="pt").input_ids.cuda()
print("fp32 worst relative logit diff:", t3_graph.check(m.t3, m.conds.t3, tt))
m.t3 = m.t3.to(torch.bfloat16); m.conds.t3 = t3_graph.cast_cond(m.conds.t3, torch.bfloat16)
print("bf16 worst relative logit diff:", t3_graph.check(m.t3, m.conds.t3, tt))
start = m.t3.hp.start_speech_token * torch.ones_like(tt[:, :1])
e, _ = m.t3.prepare_input_embeds(t3_cond=m.conds.t3, text_tokens=tt, speech_tokens=start, cfg_weight=0.0)
print("prompt length", e.shape[1])
