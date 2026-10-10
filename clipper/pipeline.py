"""End-to-end orchestration: acquire -> transcribe -> analyze -> score -> select -> render."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from .analysis.audio import compute_audio_features
from .analysis.candidates import generate_candidates
from .config import Config
from .errors import NoSpeechError
from .ingest.downloader import acquire
from .ingest.engagement import build_engagement_curve, fetch_chat_times
from .models import Clip, Transcript
from .render.exporter import render_clip
from .scoring.llm_scorer import resolve_provider, score_candidates
from .scoring.select import auto_num_clips, select_clips
from .transcribe.asr import transcribe
from .utils.ffmpeg import extract_audio, require_ffmpeg

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, float, str], None]

# stage -> (start, end) share of overall progress
STAGES = {"download": (0.00, 0.12), "audio": (0.12, 0.16), "transcribe": (0.16, 0.55),
          "analyze": (0.55, 0.64), "score": (0.64, 0.74), "render": (0.74, 1.00)}


class Pipeline:
    def __init__(self, cfg: Config, progress: Optional[ProgressFn] = None):
        self.cfg = cfg
        self._progress = progress or (lambda stage, frac, msg: None)

    def _p(self, stage: str, frac: float, msg: str = "") -> None:
        lo, hi = STAGES[stage]
        self._progress(stage, lo + (hi - lo) * max(0.0, min(frac, 1.0)), msg)

    def run(self, source: str) -> dict:
        cfg = self.cfg
        require_ffmpeg()

        self._p("download", 0, "Fetching video")
        src = acquire(source, cfg, progress=lambda f, m: self._p("download", f, m))
        self._p("download", 1, f"Got: {src.title}")

        cache = cfg.cache_dir / src.id
        cache.mkdir(parents=True, exist_ok=True)

        wav = cache / "audio.wav"
        if not wav.exists():
            self._p("audio", 0, "Extracting audio")
            extract_audio(src.video_path, str(wav))
        self._p("audio", 1)

        tr_file = cache / f"transcript_{cfg.whisper_model}.json"
        if tr_file.exists():
            tr = Transcript.from_dict(json.loads(tr_file.read_text()))
            self._p("transcribe", 1, "Transcript loaded from cache")
        else:
            tr = transcribe(str(wav), cfg, progress=lambda f, m: self._p("transcribe", f, m))
            tr_file.write_text(json.dumps(tr.to_dict()))
        if not tr.words:
            raise NoSpeechError("No speech was detected in this video.")

        self._p("analyze", 0, "Analyzing audio and text signals")
        audio = compute_audio_features(str(wav))
        chat: list[float] = []
        if cfg.fetch_chat and src.platform == "twitch" and src.url:
            chat_file = cache / "chat.json"
            if chat_file.exists():
                chat = json.loads(chat_file.read_text())
            else:
                chat = fetch_chat_times(src.url)
                chat_file.write_text(json.dumps(chat))
        duration = src.duration or tr.duration
        curve = build_engagement_curve(duration, src.heatmap, chat)
        n = cfg.num_clips or auto_num_clips(duration)
        cands = generate_candidates(tr, audio, curve, cfg, top_k=n * cfg.candidate_multiplier)
        self._p("analyze", 1, f"{len(cands)} candidate moments")

        provider = resolve_provider(cfg)
        self._p("score", 0, f"Ranking with {provider}")
        llm_cache_file = cache / "llm_scores.json"
        llm_cache = json.loads(llm_cache_file.read_text()) if llm_cache_file.exists() else {}
        score_candidates(cands, cfg, src.title, llm_cache)
        llm_cache_file.write_text(json.dumps(llm_cache))
        clips = select_clips(cands, n, cfg, tr, duration)
        self._p("score", 1, f"Selected {len(clips)} clips")

        out_dir = cfg.output_dir / src.id
        if cfg.render_clips:
            for i, clip in enumerate(clips):
                self._p("render", i / max(len(clips), 1), f"Rendering clip {i + 1}/{len(clips)}")
                render_clip(src, clip, tr, cfg, out_dir)
        self._p("render", 1, "Done")

        result = {"video": {"id": src.id, "title": src.title, "url": src.url,
                            "platform": src.platform, "duration": duration, "language": tr.language},
                  "scoring": provider, "clips": [c.to_dict() for c in clips]}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "clips.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result
