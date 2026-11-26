import os, tarfile, urllib.request
import lab_config
from huggingface_hub import hf_hub_download, snapshot_download
for f in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
    print("chatterbox", f, hf_hub_download("ResembleAI/chatterbox", f), flush=True)
print("lure", snapshot_download("Tesslate/Lure-TTS-Turbo", local_dir=os.path.join(lab_config.MODELS, "lure-turbo")), flush=True)
sha = "3f4ae026d0b0f"  # resolved below to the full commit
url = "https://codeload.github.com/smirki/lure-tts/tar.gz/3f4ae026d0b0"
os.makedirs(lab_config.SRC, exist_ok=True)
tarball = os.path.join(lab_config.SRC, "lure-tts.tar.gz")
urllib.request.urlretrieve(url, tarball)
with tarfile.open(tarball) as t:
    t.extractall(lab_config.SRC, filter="data")
print("src", os.listdir(lab_config.SRC))
