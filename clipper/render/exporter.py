"""Cut -> (reframe) -> captions -> H.264 MP4 + thumbnail for one clip."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import Config
from ..models import Clip, Transcript, VideoSource
from ..utils.ffmpeg import probe, run
from .captions import build_ass
from .reframe import plan_crop, render_simple, render_tracked

log = logging.getLogger(__name__)


def cut(video: str, start: float, end: float, out: str) -> None:
    """Frame-accurate cut (re-encode) so captions line up with the audio."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video),
         "-t", f"{end - start:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-c:a", "aac", "-b:a", "192k", str(out)])


def make_thumbnail(video: str, out_jpg: str, at: float = 1.0) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "3", str(out_jpg)])


def render_clip(src: VideoSource, clip: Clip, tr: Transcript, cfg: Config, out_dir: Path) -> Clip:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"clip_{clip.rank:02d}"
    out_mp4, out_jpg = out_dir / f"{name}.mp4", out_dir / f"{name}.jpg"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        raw = str(Path(tmp) / "raw.mp4")
        cut(src.video_path, clip.start, clip.end, raw)
        info = probe(raw)

        ass_path = None
        if cfg.captions:
            ass = build_ass(tr.words_in(clip.start, clip.end), clip.start, clip.end,
                            style=cfg.caption_style, font=cfg.caption_font,
                            width=cfg.out_width, height=cfg.out_height,
                            words_per_line=cfg.caption_words_per_line,
                            hook_text=clip.hook_text if cfg.hook_overlay else None,
                            hook_seconds=cfg.hook_seconds)
            ass_path = str(Path(tmp) / "captions.ass")
            Path(ass_path).write_text(ass, encoding="utf-8")

        landscape = info["width"] / info["height"] > 1.2
        if cfg.reframe and landscape:
            try:
                plan = plan_crop(raw, cfg)
                log.info("%s: faces in %d/%d sampled frames", name, plan.faces_found, len(plan.times))
                render_tracked(raw, plan, ass_path, str(out_mp4), cfg)
            except Exception as err:  # never lose a clip because tracking failed
                log.warning("%s: face tracking failed (%s); using centre crop.", name, err)
                render_simple(raw, ass_path, str(out_mp4), cfg, info["width"], info["height"])
        else:
            render_simple(raw, ass_path, str(out_mp4), cfg, info["width"], info["height"])

    make_thumbnail(str(out_mp4), str(out_jpg), at=min(1.0, max(clip.duration - 0.2, 0)))
    clip.video_path, clip.thumbnail = str(out_mp4), str(out_jpg)
    return clip