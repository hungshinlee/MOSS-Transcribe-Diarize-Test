"""Shared helper simulating moss-asr-deploy's real production request path
around our own transformers-based inference: KWS hotword detection (proxy/
kws.py, unmodified) -> hotword-hint prompt -> ASR generation -> Simplified-
Traditional conversion + hotword post-correction (proxy/hotword_correct.py,
unmodified), in that exact order (matches server.py's _transcribe()).

The only variable across calls is the ASR model itself (baseline vs any
fine-tuned checkpoint) -- KWS and post-processing always run the real
production code, unmodified, so results reflect "what would a user actually
see if we swapped the ASR model", not an idealized ground-truth-prompted
upper bound.

Deliberately NOT reproduced: audio_split.py's >40min segment splitting is
handled by the CALLER (eval_deploy_audio.py already does this) -- this
module only covers one already-short-enough audio/segment at a time, mirroring
how server.py calls vLLM once per segment.

Usage:
    from production_pipeline import production_transcribe
    text = production_transcribe(model, processor, audio_path, device, max_new_tokens=1200)
"""

import os
import sys

import numpy as np
import soundfile as sf
import soxr
import torch
from opencc import OpenCC

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../moss_repeat_repro

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPRO_ROOT, "finetune_official"))
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages, generate_transcription  # noqa: E402
from mer_utils import _SEGMENT_RE  # noqa: E402  (shared [start][Sxx]...[end] regex)

# KWS + hotword_correct are moss-asr-deploy/proxy's own code (ADVANCED/OPTIONAL in this
# repro package -- not bundled, see README). Imported lazily so that simply importing
# this module (required by eval_dev_mer.py/eval_hotword_testset.py even in plain vanilla
# mode, which never calls production_transcribe()) doesn't hard-fail when the proxy code
# isn't present. Only calling production_transcribe()/_apply_hotword_correction() without
# it actually installed will raise, with a clear message pointing at the README.
try:
    sys.path.insert(0, os.path.join(REPRO_ROOT, "moss_asr_deploy_proxy"))
    from kws import detect_keywords_from_samples  # noqa: E402  (proxy's own module, unmodified)
    from hotword_correct import correct_text  # noqa: E402  (proxy's own module, unmodified)
except ImportError:
    def detect_keywords_from_samples(*_args, **_kwargs):
        raise RuntimeError(
            "KWS proxy code not available in this repro package -- production_pipeline mode "
            "(--production_pipeline flag) is ADVANCED/OPTIONAL, not required for the primary "
            "vanilla-mode repro. See README.md for how to add moss-asr-deploy/proxy if you want it."
        )

    def correct_text(*_args, **_kwargs):
        raise RuntimeError(
            "hotword_correct proxy code not available in this repro package -- see README.md "
            "for how to add moss-asr-deploy/proxy if you want --production_pipeline mode."
        )

_CC_S2TW = OpenCC("s2tw")  # matches server.py's exact converter config, not our own eval "s2t"
TARGET_SR = 16000


def _load_samples_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """Read any audio file, downmix to mono, resample to 16kHz, return
    [-1,1]-normalized float32 samples -- matches the exact array server.py
    builds for both KWS and ASR (np.frombuffer(...).astype(np.float32)/32768.0),
    and what detect_keywords_from_samples expects."""
    audio, sr = sf.read(audio_path, always_2d=True)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR)
    return mono.astype(np.float32), TARGET_SR


def _apply_hotword_correction(text: str) -> str:
    """Identical logic to server.py's _apply_hotword_correction: only the
    text INSIDE each [start][Sxx]...[end] segment is corrected, tags/
    timestamps are reconstructed untouched. Copied (not imported from
    server.py) to avoid importing the whole FastAPI app module's side
    effects (app/client/semaphore construction) for a 6-line wrapper."""
    def repl(m):
        start, speaker, seg_text, end = m.groups()
        corrected = correct_text(seg_text.strip())
        return f"[{start}][S{speaker}]{corrected}[{end}]"
    return _SEGMENT_RE.sub(repl, text)


def production_transcribe(model, processor, audio_path: str, device, max_new_tokens: int = 1200) -> str:
    """Run the full production-realistic pipeline for one already-short-enough
    audio/segment: KWS detect -> build hint prompt -> generate -> s2tw ->
    hotword-correct. Returns the final tagged text, same format as raw model
    output (caller strips tags for scoring as usual)."""
    samples, sr = _load_samples_16k(audio_path)
    detected = detect_keywords_from_samples(samples, sr)
    keywords = sorted({k for k, _offset in detected})
    prompt = DEFAULT_PROMPT + ("热词提示：" + ", ".join(keywords) if keywords else "")

    messages = build_transcription_messages(audio_path, prompt)
    result = generate_transcription(model, processor, messages, device=device, dtype=torch.bfloat16,
                                      max_new_tokens=max_new_tokens)
    text = _CC_S2TW.convert(result["text"])
    text = _apply_hotword_correction(text)
    return text


def real_production_transcribe(audio_path: str, proxy_base_url: str, timeout: float = 600.0) -> str:
    """Call the REAL moss-asr-deploy stack (actual vLLM server + actual FastAPI
    proxy, running in Docker -- see /home/wago/moss-asr-deploy) instead of
    simulating the production path locally via production_transcribe(). Use
    when exact fidelity to the deployed degenerate-repetition-retry behavior
    (server.py's _post_with_retry_on_empty()) matters more than iteration
    speed. Requires the vLLM+proxy containers to already be running and
    holding the GPU for the whole eval session (unlike production_transcribe(),
    which loads/unloads the model once per script invocation).

    The proxy only accepts WAV (server.py's _audio_duration_seconds() uses the
    stdlib wave module, which rejects mp3/other containers outright) -- caller
    is responsible for passing a .wav path.
    """
    import requests
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{proxy_base_url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()["text"]
