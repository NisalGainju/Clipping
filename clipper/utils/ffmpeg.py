from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction

from ..errors import DependencyError


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def require_ffmpeg() -> None:
    if not have_ffmpeg():
        raise DependencyError("ffmpeg/ffprobe not found. Install it (apt install ffmpeg / brew install ffmpeg).")


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"Command failed ({cmd[0]}): {tail}")
    return proc.stdout


def probe(path: str) -> dict:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate:format=duration",
               "-of", "json", str(path)])
    d = json.loads(out)
    s = d["streams"][0]
    try:
        fps = float(Fraction(s.get("r_frame_rate", "30/1")))
    except (ZeroDivisionError, ValueError):
        fps = 30.0
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": fps or 30.0, "duration": float(d["format"].get("duration", 0) or 0)}


def extract_audio(video: str, out_wav: str) -> None:
    """16 kHz mono PCM, the format Whisper and the audio analyzer expect."""
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out_wav)])


def filter_path(p: str) -> str:
    """Escape a file path for use inside an ffmpeg filter argument."""
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
