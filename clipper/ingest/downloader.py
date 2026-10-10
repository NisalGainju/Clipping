"""Acquire a video from a URL (yt-dlp) or a local file."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from ..config import Config
from ..errors import DependencyError, VideoUnavailableError
from ..models import VideoSource
from ..utils.ffmpeg import probe

log = logging.getLogger(__name__)
Progress = Optional[Callable[[float, str], None]]

_ERROR_HINTS = [
    (("private video", "sign in to confirm your age", "age-restricted", "age restricted"),
     "This video is private or age-restricted. Provide a cookies.txt (--cookies) from an account with access."),
    (("not available in your country", "geo", "blocked it in your country"),
     "This video is geo-blocked in the region the server runs in."),
    (("confirm you", "not a bot"),
     "The platform is asking for bot verification (common on cloud IPs like Colab). Use --cookies, or download locally and pass the file."),
    (("video unavailable", "has been removed", "does not exist", "no longer available"),
     "This video is unavailable or has been removed."),
    (("unsupported url",), "This URL is not supported by yt-dlp."),
]


def _platform(url: str) -> str:
    u = url.lower()
    for name, key in (("youtube", "youtu"), ("twitch", "twitch.tv"), ("kick", "kick.com")):
        if key in u:
            return name
    return "web"


def _explain(exc: Exception) -> str:
    msg = str(exc).lower()
    for needles, text in _ERROR_HINTS:
        if any(n in msg for n in needles):
            return text
    return f"Could not download the video: {str(exc)[:300]}"


def acquire(source: str, cfg: Config, progress: Progress = None) -> VideoSource:
    path = Path(source).expanduser()
    if path.exists():
        return _from_local(path)
    if not source.lower().startswith(("http://", "https://")):
        raise VideoUnavailableError(f"'{source}' is neither an existing file nor a URL.")
    return _download(source, cfg, progress)


def _from_local(path: Path) -> VideoSource:
    info = probe(str(path))
    vid = hashlib.sha1(f"{path.resolve()}:{path.stat().st_size}".encode()).hexdigest()[:12]
    return VideoSource(id=f"local_{vid}", title=path.stem, video_path=str(path.resolve()),
                       duration=info["duration"], platform="local")


def _download(url: str, cfg: Config, progress: Progress) -> VideoSource:
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise DependencyError("yt-dlp is not installed (pip install yt-dlp).") from e

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(min(d.get("downloaded_bytes", 0) / total, 1.0), "Downloading video")

    opts = {
        "format": f"bv*[height<={cfg.max_height}]+ba/b[height<={cfg.max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(cfg.download_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
    }
    if cfg.cookies_file:
        opts["cookiefile"] = cfg.cookies_file

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = info["entries"][0]
            vid = info["id"]
            candidates = sorted(cfg.download_dir.glob(f"{vid}.mp4")) or sorted(cfg.download_dir.glob(f"{vid}.*"))
            if not candidates:
                raise VideoUnavailableError("Download finished but the output file was not found.")
            video_path = str(candidates[0])
    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(_explain(e)) from e

    duration = float(info.get("duration") or probe(video_path)["duration"])
    return VideoSource(id=f"{_platform(url)}_{vid}", title=info.get("title") or vid,
                       video_path=video_path, duration=duration, url=url,
                       platform=_platform(url), heatmap=info.get("heatmap") or [])
