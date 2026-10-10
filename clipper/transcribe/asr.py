"""Word-level transcription with faster-whisper."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from ..config import Config
from ..errors import DependencyError, NoSpeechError
from ..models import Transcript, Word

log = logging.getLogger(__name__)


def _resolve_device(cfg: Config) -> tuple[str, str]:
    device = cfg.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = cfg.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def transcribe(audio_path: str, cfg: Config, progress: Optional[Callable[[float, str], None]] = None) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise DependencyError("faster-whisper is not installed (pip install faster-whisper).") from e

    device, compute = _resolve_device(cfg)
    log.info("Whisper model=%s device=%s compute=%s", cfg.whisper_model, device, compute)
    if device == "cpu" and cfg.whisper_model.startswith(("large", "distil")):
        log.warning("Large Whisper models on CPU are very slow for long videos; consider Colab or a smaller model.")

    model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)
    # faster-whisper streams segments lazily, so memory stays flat even for multi-hour audio.
    segments, info = model.transcribe(
        audio_path,
        language=cfg.language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5 if device == "cuda" else 1,
        condition_on_previous_text=False,
    )
    total = max(info.duration, 1.0)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text, float(w.probability)))
        if progress:
            progress(min(seg.end / total, 1.0), "Transcribing")
    if not words:
        raise NoSpeechError("No speech was detected in this video.")
    return Transcript(language=info.language, duration=float(info.duration), words=words)
