"""Visual signals (optional): motion intensity and scene changes, sampled sparsely.

Only run on the final selected clips / top candidates, never on the whole video.
"""
from __future__ import annotations


def motion_and_cuts(video_path: str, start: float, end: float, sample_fps: float = 4.0) -> dict:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / sample_fps)), 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    prev, diffs, idx = None, [], int(start * fps)
    last = int(end * fps)
    while idx < last:
        ok = cap.grab()
        if not ok:
            break
        if (idx - int(start * fps)) % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(g, prev))))
            prev = g
        idx += 1
    cap.release()
    if not diffs:
        return {"motion": 0.0, "cuts": 0}
    d = np.array(diffs)
    return {"motion": float(min(d.mean() / 25.0, 1.0)), "cuts": int((d > 40).sum())}
