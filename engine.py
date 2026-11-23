# File: engine.py
# Core TTS model loading and speech generation logic.

import gc
import logging
import os
import random
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Defensive Turbo import - Turbo may not be available in older package versions
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Defensive Multilingual import
try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# Log BF16 setting at module load so it's visible in startup logs
# (BF16_ENABLED is resolved after logger is set up — logged in initialize_tts_model)
if TURBO_AVAILABLE:
    logger.info("ChatterboxTurboTTS is available in the installed chatterbox package.")
else:
    logger.info("ChatterboxTurboTTS not available in installed chatterbox package.")

# Log Multilingual availability status at module load time
if MULTILINGUAL_AVAILABLE:
    logger.info("ChatterboxMultilingualTTS is available in the installed chatterbox package.")
    logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
else:
    logger.info("ChatterboxMultilingualTTS not available in installed chatterbox package.")

# Model selector whitelist - maps config values to model types
MODEL_SELECTOR_MAP = {
    # Original model selectors
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",
    # Turbo model selectors
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    # Multilingual model selectors
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
}

# Paralinguistic tags supported by Turbo model
TURBO_PARALINGUISTIC_TAGS = [
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear throat",
    "sniff",
    "groan",
    "shush",
]

# --- BF16 optimization flag ---
# TTS_BF16: controls whether T3 is converted to bfloat16 and whether
# autocast is used during inference. Off by default so existing users
# see no behavior change on upgrade — opt in for the speedup.
#   off (default) — keep T3 in float32, no autocast
#   on / 1 / true  — force-enable (assumes hardware supports bf16)
#   auto           — enable only if torch.cuda.is_bf16_supported()
def _resolve_bf16_setting() -> bool:
    val = os.environ.get("TTS_BF16", "off").strip().lower()
    if val in ("on", "1", "true"):
        return True
    if val == "auto":
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
        return False
    # off / 0 / false / anything else
    return False

BF16_ENABLED: bool = _resolve_bf16_setting()


# --- Turbo fast path (this fork; t3_fast.py says what each does and what it measured) ---
# All off by default, so the server behaves as upstream's. Turbo on CUDA only; other models and
# devices ignore them.
#   TTS_T3_GRAPH=on       the CUDA-graph token loop, checked at load, falling back to the stock loop
#   TTS_SLIM=on           drop the VRAM Turbo's generation never uses
#   TTS_BF16_AUTOCAST=off with TTS_BF16 on, bf16 weights without autocast (autocast was slower)
def _env_on(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("on", "1", "true")


T3_GRAPH: bool = _env_on("TTS_T3_GRAPH", "off")
SLIM: bool = _env_on("TTS_SLIM", "off")
BF16_AUTOCAST: bool = _env_on("TTS_BF16_AUTOCAST", "on")
# Whether synthesize wraps generate in bf16 autocast; load_model settles it for the loaded model.
_use_autocast: bool = BF16_ENABLED

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Track which model type is loaded
loaded_model_type: Optional[str] = None  # "original" or "turbo"
loaded_model_class_name: Optional[str] = None  # "ChatterboxTTS" or "ChatterboxTurboTTS"

# Voice conditioning cache: avoids re-encoding the same voice file on every request.
# Key: (resolved_path, file_mtime, exaggeration) — mtime invalidates if file changes.
_conds_cache: dict = {}


def _conds_cache_key(path: str, exaggeration: float) -> tuple:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (path, mtime, exaggeration)


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def _get_model_class(selector: str) -> tuple:
    """
    Determines which model class to use based on the config selector value.

    Args:
        selector: The value from config model.repo_id

    Returns:
        Tuple of (model_class, model_type_string)

    Raises:
        ImportError: If Turbo or Multilingual is selected but not available in the package
    """
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTurboTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Turbo model (ChatterboxTurboTTS)"
        )
        return ChatterboxTurboTTS, "turbo"

    if model_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxMultilingualTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)"
        )
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "original":
        logger.info(
            f"Model selector '{selector}' resolved to Original model (ChatterboxTTS)"
        )
        return ChatterboxTTS, "original"

    # Unknown selector - default to original with warning
    logger.warning(
        f"Unknown model selector '{selector}'. "
        f"Valid values: chatterbox, chatterbox-turbo, chatterbox-multilingual, original, turbo, multilingual, "
        f"ResembleAI/chatterbox, ResembleAI/chatterbox-turbo. "
        f"Defaulting to original ChatterboxTTS model."
    )
    return ChatterboxTTS, "original"


def get_model_info() -> dict:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.

    Returns:
        Dictionary containing model information
    """
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,  # "original", "turbo", or "multilingual"
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": (
            TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else []
        ),
        "turbo_available_in_package": TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
        "supports_multilingual": loaded_model_type == "multilingual",
        "supported_languages": (
            SUPPORTED_LANGUAGES if loaded_model_type == "multilingual" else {"en": "English"}
        ),
    }


def _from_pretrained_on_device(model_class, device: str):
    """The engine's from_pretrained, but for Turbo on CUDA the modules are built on the GPU.
    The engine builds on the CPU and reads each checkpoint into a second CPU copy before
    moving it, which costs several GB of RAM at every load and most of the load time.

    The checkpoints are still read into RAM and copied into the GPU's modules, so the card
    holds one copy of the model at any moment. Reading them straight to the GPU as well made
    the card hold two for a moment: a load peaked at 7.3 GB of VRAM for a model that settles
    at 4.9 GB, and failed beside another tenant that left less (2026-11-19). RAM peaks at about
    the checkpoints' size while it loads, and is trimmed right after. Other models and devices
    load as before."""
    if device != "cuda" or model_class is not ChatterboxTurboTTS:
        return model_class.from_pretrained(device=device)

    with torch.device(device):
        model = model_class.from_pretrained(device=device)
    gc.collect()  # The checkpoints read into RAM.
    torch.cuda.empty_cache()  # Blocks freed while building (random init, the dropped wte).
    _trim_host_memory()
    return model


def _apply_turbo_fast_path(model, model_type: str, device: str) -> None:
    """TTS_SLIM, TTS_BF16_AUTOCAST and TTS_T3_GRAPH for Turbo on CUDA. Each step that fails is
    logged and skipped, so the model still serves through the stock path."""
    global _use_autocast
    _use_autocast = BF16_ENABLED
    if model_type != "turbo" or device != "cuda":
        if T3_GRAPH or SLIM or not BF16_AUTOCAST:
            logger.info("Turbo fast path: not Turbo on CUDA, so TTS_T3_GRAPH, TTS_SLIM and TTS_BF16_AUTOCAST are ignored.")
        return
    import t3_fast

    if SLIM:
        try:
            freed = t3_fast.slim(model)
            logger.info(f"TTS_SLIM: freed {freed / 2**30:.2f} GiB of VRAM the generation never uses.")
        except Exception:
            logger.exception("TTS_SLIM failed; continuing without it.")
    if BF16_ENABLED and not BF16_AUTOCAST:
        t3_fast.cast_inputs(model.t3)
        _use_autocast = False
        logger.info("TTS_BF16_AUTOCAST=off: T3 runs in bf16 without autocast.")
    if T3_GRAPH:
        try:
            worst = t3_fast.install_graph(model)
            logger.info(f"TTS_T3_GRAPH: CUDA-graph token loop installed (worst logit difference {worst:.4f}).")
        except Exception:
            logger.exception("TTS_T3_GRAPH: the graph loop failed its check; using the stock loop.")
    if torch.cuda.is_available():
        # Hand back what the steps above left reserved (the float32 weights bf16 replaced, the
        # graph's warm-up).
        torch.cuda.empty_cache()
        logger.info(
            f"Turbo VRAM after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved."
        )


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global loaded_model_type, loaded_model_class_name

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")
        logger.info(
            f"BF16 optimization: {'enabled' if BF16_ENABLED else 'disabled'} "
            f"(TTS_BF16={os.environ.get('TTS_BF16', 'off')})"
        )

        # Get the model selector from config
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")

        logger.info(f"Model selector from config: '{model_selector}'")

        try:
            # Determine which model class to use
            model_class, model_type = _get_model_class(model_selector)

            logger.info(
                f"Initializing {model_class.__name__} on device '{model_device}'..."
            )
            logger.info(f"Model type: {model_type}")
            if model_type == "turbo":
                logger.info(
                    f"Turbo model supports paralinguistic tags: {TURBO_PARALINGUISTIC_TAGS}"
                )

            # Load the model using from_pretrained - handles HuggingFace downloads automatically
            chatterbox_model = _from_pretrained_on_device(model_class, model_device)

            # Convert T3 to bfloat16 if enabled.
            # Token generation is memory-bandwidth bound; bf16 halves bytes read per
            # forward pass. S3Gen is intentionally kept in float32 — it runs only
            # 2 CFM timesteps and bf16 causes token/mask size mismatches.
            if BF16_ENABLED:
                if hasattr(chatterbox_model, "t3"):
                    chatterbox_model.t3 = chatterbox_model.t3.bfloat16()
                    logger.info("T3 model converted to bfloat16 for faster token generation.")
            else:
                logger.info("BF16 optimization disabled (TTS_BF16=off or hardware unsupported).")

            _apply_turbo_fast_path(chatterbox_model, model_type, model_device)

            # Store model metadata
            loaded_model_type = model_type
            loaded_model_class_name = model_class.__name__

            logger.info(f"Successfully loaded {model_class.__name__} on {model_device}")
            logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
        except ImportError as e_import:
            logger.error(
                f"Failed to load model due to import error: {e_import}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained: {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code for multilingual model (e.g., 'en', 'it', 'de').

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        # Voice conditioning cache: skip re-encoding the same voice file.
        # Turbo ignores exaggeration in conds; others include it in the key.
        effective_prompt = audio_prompt_path
        conds_key = None
        if audio_prompt_path and hasattr(chatterbox_model, "conds"):
            ex_for_key = 0.0 if loaded_model_type == "turbo" else exaggeration
            conds_key = _conds_cache_key(audio_prompt_path, ex_for_key)
            if conds_key in _conds_cache:
                chatterbox_model.conds = _conds_cache[conds_key]
                effective_prompt = None  # conds already set, skip prepare_conditionals
                logger.debug(f"Voice cache hit: {audio_prompt_path}")

        # Call the core model's generate method.
        # autocast promotes float32 inputs to bfloat16 to match T3/S3Gen weights,
        # keeping numerically sensitive ops (softmax, norms) in float32 automatically.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_autocast):
            if loaded_model_type == "multilingual":
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=effective_prompt,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
            else:
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    audio_prompt_path=effective_prompt,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

        # Store conds in cache after first compute for this voice.
        if conds_key is not None and effective_prompt is not None:
            if chatterbox_model.conds is not None:
                _conds_cache[conds_key] = chatterbox_model.conds
                logger.debug(f"Cached voice conditionals for: {audio_prompt_path}")

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def _trim_host_memory() -> None:
    """Hand freed heap pages back to the OS. glibc keeps them after a model is freed, so
    without this an unload returns no RAM and each reload adds to the process's size."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Not glibc (macOS, Windows, musl): nothing to do.


def unload_model() -> bool:
    """
    Unloads the current model and releases all GPU memory.
    Does NOT reload the model - use reload_model() for that.

    Returns:
        bool: True if the model was unloaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    logger.info("Initiating model unload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags and drop the voice cache: its conditionals are GPU tensors.
    MODEL_LOADED = False
    model_device = None
    loaded_model_type = None
    loaded_model_class_name = None
    _conds_cache.clear()

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    _trim_host_memory()
    logger.info("Model unloaded and GPU memory released.")
    return True


def reload_model() -> bool:
    """
    Unloads the current model, clears GPU memory, and reloads the model
    based on the current configuration. Used for hot-swapping models
    without restarting the server process.

    Returns:
        bool: True if the new model loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name, _conds_cache

    logger.info("Initiating model hot-swap/reload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading existing TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags and clear voice cache (conds are model-specific)
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None
    _conds_cache.clear()
    logger.info("Voice conditioning cache cleared.")

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            # Older PyTorch versions may not have mps.empty_cache()
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    # 6. Reload model from the (now updated) configuration
    logger.info("Memory cleared. Reloading model from updated config...")
    return load_model()


# --- End File: engine.py ---
