from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


def _default_weights() -> dict:
    return {
        "hook": 0.22, "emotion": 0.18, "opinion": 0.08, "quotable": 0.08,
        "question": 0.04, "pace": 0.06, "audio": 0.16, "engagement": 0.12,
        "completeness": 0.06,
    }


@dataclass
class Config:
    work_dir: str = "workspace"

    # ingest
    max_height: int = 1080
    cookies_file: Optional[str] = None
    fetch_chat: bool = True

    # transcription
    whisper_model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None

    # clip selection
    min_clip_s: float = 20.0
    max_clip_s: float = 60.0
    target_clip_s: float = 40.0
    num_clips: Optional[int] = None
    candidate_multiplier: int = 3
    pad_start: float = 0.25
    pad_end: float = 0.40
    heuristic_weights: dict = field(default_factory=_default_weights)

    # LLM scoring
    llm_provider: str = "auto"  # auto | anthropic | heuristic
    llm_model: str = "claude-sonnet-4-6"
    llm_batch_size: int = 8
    llm_blend: float = 0.85  # weight of LLM score vs heuristic score

    # render
    render_clips: bool = True
    out_width: int = 1080
    out_height: int = 1920
    crf: int = 20
    preset: str = "veryfast"
    video_codec: str = "libx264"  # use h264_nvenc on NVIDIA machines
    reframe: bool = True
    reframe_sample_fps: float = 5.0
    reframe_deadzone: float = 0.12
    reframe_smooth: float = 0.30
    captions: bool = True
    caption_style: str = "pop"
    caption_font: str = "Arial"
    caption_words_per_line: int = 3
    hook_overlay: bool = True
    hook_seconds: float = 2.8
    music_path: Optional[str] = None
    music_volume: float = 0.12

    # paths -------------------------------------------------------------
    @property
    def root(self) -> Path:
        p = Path(self.work_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sub(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def download_dir(self) -> Path:
        return self._sub("downloads")

    @property
    def cache_dir(self) -> Path:
        return self._sub("cache")

    @property
    def output_dir(self) -> Path:
        return self._sub("outputs")

    @property
    def upload_dir(self) -> Path:
        return self._sub("uploads")


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Load defaults, then YAML file, then non-None overrides."""
    import yaml

    cfg = Config()
    valid = {f.name for f in fields(Config)}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in data.items():
            if k in valid:
                setattr(cfg, k, v)
    for k, v in (overrides or {}).items():
        if k in valid and v is not None:
            setattr(cfg, k, v)
    return cfg
