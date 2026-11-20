# File: engine_process.py
# The engine in a worker process, so that unloading the model gives back everything it held.
#
# The model, its CUDA context and every library it warms up live in a child process; unloading
# ends that process, which returns all of its RAM and VRAM to the system. An in-process unload
# can't: the CUDA driver keeps the kernels it has loaded (over a GB of host RAM) until the
# context is destroyed, and a context lives as long as its process.
#
# It offers the part of engine.py the server uses (load_model, reload_model, unload_model,
# synthesize, get_model_info, MODEL_LOADED), so server.py picks one or the other at import.
# Turned on with TTS_WORKER_PROCESS=1.

import logging
import multiprocessing as mp
import threading
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

MODEL_LOADED: bool = False

_ctx = mp.get_context("spawn")  # A fork would inherit the parent's state, CUDA included.
_process: Optional[mp.process.BaseProcess] = None
_conn = None
_info: Optional[dict] = None
_lock = threading.Lock()  # One request on the pipe at a time; /tts runs in a thread pool.


def _worker_main(conn) -> None:
    """The child: load the model, then answer requests until told to stop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] worker %(name)s: %(message)s")
    import engine

    ok = engine.load_model()
    conn.send(("loaded", ok, engine.get_model_info()))
    if not ok:
        return
    while True:
        try:
            op, kwargs = conn.recv()
        except EOFError:
            return
        if op == "stop":
            return
        if op == "synthesize":
            wav, sr = engine.synthesize(**kwargs)
            conn.send((wav.detach().cpu().numpy() if wav is not None else None, sr))


def load_model() -> bool:
    global _process, _conn, _info, MODEL_LOADED
    with _lock:
        if MODEL_LOADED and _process is not None and _process.is_alive():
            logger.info("TTS model is already loaded (worker process).")
            return True
        parent_conn, child_conn = _ctx.Pipe()
        process = _ctx.Process(target=_worker_main, args=(child_conn,), name="tts-engine", daemon=True)
        process.start()
        child_conn.close()
        try:
            _, ok, info = parent_conn.recv()
        except EOFError:
            ok, info = False, None
        if not ok:
            logger.error("The TTS worker process failed to load the model.")
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
            return False
        _process, _conn, _info, MODEL_LOADED = process, parent_conn, info, True
        logger.info(f"TTS model loaded in worker process {process.pid}.")
        return True


def _stop_locked() -> None:
    global _process, _conn, _info, MODEL_LOADED
    MODEL_LOADED = False
    if _process is not None:
        try:
            _conn.send(("stop", None))
        except (BrokenPipeError, OSError):
            pass
        _process.join(timeout=10)
        if _process.is_alive():
            _process.kill()
            _process.join()
        logger.info(f"TTS worker process {_process.pid} ended; its RAM and VRAM are released.")
    if _conn is not None:
        _conn.close()
    _process, _conn, _info = None, None, None


def unload_model() -> bool:
    with _lock:
        _stop_locked()
    return True


def reload_model() -> bool:
    unload_model()
    return load_model()


def synthesize(**kwargs) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    with _lock:
        if not MODEL_LOADED or _conn is None:
            logger.error("TTS model is not loaded. Cannot synthesize audio.")
            return None, None
        try:
            _conn.send(("synthesize", kwargs))
            wav, sr = _conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            logger.error("The TTS worker process died during synthesis.", exc_info=True)
            _stop_locked()
            return None, None
    return (torch.from_numpy(wav) if wav is not None else None), sr


def get_model_info() -> dict:
    if MODEL_LOADED and _info is not None:
        return _info
    import engine  # Its own state says "not loaded", in the same shape.

    return engine.get_model_info()
