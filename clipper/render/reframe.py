"""Face-tracked 16:9 -> 9:16 reframing with a smoothed crop path."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..utils.ffmpeg import filter_path

log = logging.getLogger(__name__)


@dataclass
class CropPlan:
    times: np.ndarray   # seconds
    xs: np.ndarray      # crop left edge in source pixels
    crop_w: int
    src_w: int
    src_h: int
    fps: float
    faces_found: int


CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + CASCADE_NAME


def _find_cascade(models_dir: Path) -> Optional[str]:
    """Locate the Haar cascade: bundled with OpenCV, cached locally, or downloaded once."""
    import cv2

    bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if bundled and Path(bundled, CASCADE_NAME).exists():
        return str(Path(bundled, CASCADE_NAME))
    local = models_dir / CASCADE_NAME
    if local.exists():
        return str(local)
    try:
        import urllib.request
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CASCADE_URL, str(local))
        return str(local)
    except Exception as e:
        log.warning("Could not obtain face-detection model (%s); falling back to centre crop.", e)
        return None


class FaceDetector:
    """MediaPipe if available, otherwise OpenCV Haar cascade, otherwise no detection (centre crop)."""

    def __init__(self, models_dir: Path) -> None:
        import cv2
        self.cv2 = cv2
        self.mp = None
        self.haar = None
        try:
            import mediapipe as mp
            self.mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            self.mp = None
        if self.mp is None:
            path = _find_cascade(models_dir)
            if path:
                clf = cv2.CascadeClassifier(path)
                self.haar = None if clf.empty() else clf

    @property
    def available(self) -> bool:
        return self.mp is not None or self.haar is not None

    def detect(self, frame) -> list[tuple[float, float]]:
        """Returns [(center_x_px, area_px)] in original frame coordinates."""
        cv2 = self.cv2
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        out: list[tuple[float, float]] = []
        if self.mp is not None:
            res = self.mp.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                bb = d.location_data.relative_bounding_box
                out.append(((bb.xmin + bb.width / 2) * w, bb.width * bb.height * w * h))
        elif self.haar is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self.haar.detectMultiScale(gray, 1.1, 5, minSize=(small.shape[1] // 25,) * 2):
                out.append(((x + fw / 2) / scale, fw * fh / (scale * scale)))
        return out


def plan_crop(clip_path: str, cfg: Config) -> CropPlan:
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    crop_w = int(src_h * cfg.out_width / cfg.out_height) // 2 * 2
    step = max(int(round(fps / cfg.reframe_sample_fps)), 1)

    det = FaceDetector(cfg.root / "models")
    times, raw = [], []
    prev = None
    found = 0
    if det.available:
        i = 0
        try:
            while True:
                if not cap.grab():
                    break
                if i % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    faces = det.detect(frame)
                    cx = np.nan
                    if faces:
                        found += 1
                        # Prefer the biggest face, with a bonus for staying on the previous subject (stops ping-pong).
                        best = max(faces, key=lambda f: f[1] * (1.4 if prev is not None and abs(f[0] - prev) < crop_w * 0.4 else 1.0))
                        cx = best[0]
                        prev = cx
                    times.append(i / fps)
                    raw.append(cx)
                i += 1
        finally:
            cap.release()  # Windows cannot delete a file that is still open
    else:
        cap.release()

    raw_a = np.array(raw, dtype=float) if raw else np.array([src_w / 2.0])
    times_a = np.array(times) if times else np.array([0.0])
    # fill gaps: forward-fill, back-fill, else centre
    if np.all(np.isnan(raw_a)):
        raw_a[:] = src_w / 2.0
    else:
        for k in range(1, len(raw_a)):
            if np.isnan(raw_a[k]):
                raw_a[k] = raw_a[k - 1]
        for k in range(len(raw_a) - 2, -1, -1):
            if np.isnan(raw_a[k]):
                raw_a[k] = raw_a[k + 1]

    # dead-zone + exponential smoothing + speed limit => no jitter, gentle pans
    cur = raw_a[0]
    smooth = np.empty_like(raw_a)
    dz = cfg.reframe_deadzone * crop_w
    max_step = crop_w * 0.08
    for k, v in enumerate(raw_a):
        target = v if abs(v - cur) > dz else cur
        delta = np.clip(cfg.reframe_smooth * (target - cur), -max_step, max_step)
        cur = cur + delta
        smooth[k] = cur
    xs = np.clip(smooth - crop_w / 2, 0, max(src_w - crop_w, 0))
    return CropPlan(times_a, xs, crop_w, src_w, src_h, fps, found)


def _audio_filter_graph(music: bool, volume: float) -> tuple[str, str]:
    if not music:
        return "", "1:a:0?"
    g = (f";[1:a]asplit=2[sp1][sp2];[2:a]volume={volume}[bed];"
         "[bed][sp1]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[duck];"
         "[sp2][duck]amix=inputs=2:duration=first:dropout_transition=0[aout]")
    return g, "[aout]"


def render_tracked(raw_clip: str, plan: CropPlan, ass_path: str | None, out_path: str, cfg: Config) -> None:
    """Crop per frame in OpenCV, pipe raw frames into ffmpeg, burn captions, mux original audio."""
    import cv2

    W, H = cfg.out_width, cfg.out_height
    vf = f"[0:v]ass='{filter_path(ass_path)}'[v]" if ass_path else "[0:v]null[v]"
    music = bool(cfg.music_path)
    ag, amap = _audio_filter_graph(music, cfg.music_volume)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{plan.fps:.3f}", "-i", "-", "-i", str(raw_clip)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(cfg.music_path)]
    cmd += ["-filter_complex", vf + ag, "-map", "[v]", "-map", amap,
            "-c:v", cfg.video_codec, "-preset", cfg.preset, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_path)]
    if cfg.video_codec == "h264_nvenc":  # nvenc has no -crf/-preset veryfast
        i = cmd.index("-preset")
        cmd[i:i + 4] = ["-preset", "p4", "-cq", str(cfg.crf)]

    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
    cap = cv2.VideoCapture(str(raw_clip))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x = int(np.interp(idx / plan.fps, plan.times, plan.xs))
            crop = frame[:, x:x + plan.crop_w]
            proc.stdin.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA).tobytes())
            idx += 1
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        err.seek(0)
        raise RuntimeError("ffmpeg render failed: " + err.read().decode(errors="ignore")[-600:])


def render_simple(raw_clip: str, ass_path: str | None, out_path: str, cfg: Config, src_w: int, src_h: int) -> None:
    """No tracking: centre-crop landscape sources, letterbox everything else. Pure ffmpeg."""
    W, H = cfg.out_width, cfg.out_height
    if src_w / src_h > 1.2:
        v = f"crop=ih*{W}/{H}:ih,scale={W}:{H}"
    else:
        v = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"
    if ass_path:
        v += f",ass='{filter_path(ass_path)}'"
    music = bool(cfg.music_path)
    ag, amap = _audio_filter_graph(music, cfg.music_volume)
    # in this graph the clip is input 0, so rewrite the audio labels
    ag = ag.replace("[1:a]", "[0:a]").replace("[2:a]", "[1:a]")
    amap = amap.replace("1:a:0?", "0:a:0?")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_clip)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(cfg.music_path)]
    cmd += ["-filter_complex", f"[0:v]{v}[v]" + ag, "-map", "[v]", "-map", amap,
            "-c:v", cfg.video_codec, "-preset", cfg.preset, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_path)]
    if cfg.video_codec == "h264_nvenc":
        i = cmd.index("-preset")
        cmd[i:i + 4] = ["-preset", "p4", "-cq", str(cfg.crf)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg render failed: " + proc.stderr[-600:])