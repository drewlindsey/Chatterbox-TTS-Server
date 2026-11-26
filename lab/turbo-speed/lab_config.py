"""Settings shared by the lab scripts, read from environment variables.

  TTS_BASE_URL  running server for paths.py and server_bench.py (default http://127.0.0.1:8004)
  LAB_VOICE     reference WAV for the direct-model scripts (default: the repository's voices/Jade.wav)
  LAB_OUT       where scripts write WAVs (default: out/ beside these scripts)
  LAB_MODELS    downloaded model weights (default: models/ beside these scripts)
  LAB_SRC       downloaded third-party source (default: src/ beside these scripts)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

BASE_URL = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8004").rstrip("/")
VOICE = os.environ.get("LAB_VOICE", os.path.join(REPO, "voices", "Jade.wav"))
OUT = os.environ.get("LAB_OUT", os.path.join(HERE, "out"))
MODELS = os.environ.get("LAB_MODELS", os.path.join(HERE, "models"))
SRC = os.environ.get("LAB_SRC", os.path.join(HERE, "src"))
