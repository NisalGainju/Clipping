"""Engagement signals: YouTube most-replayed heatmap and Twitch chat velocity."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def fetch_chat_times(url: str) -> list[float]:
    """Message timestamps (seconds) for a Twitch VOD. Returns [] if unavailable."""
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        log.info("chat-downloader not installed; skipping chat velocity.")
        return []
    try:
        chat = ChatDownloader().get_chat(url, message_groups=["messages"])
        return [float(m["time_in_seconds"]) for m in chat if m.get("time_in_seconds") is not None and m["time_in_seconds"] >= 0]
    except Exception as e:  # network, auth, unsupported VOD...
        log.warning("Could not fetch chat: %s", e)
        return []


def build_engagement_curve(duration: float, heatmap: list[dict], chat_times: list[float]) -> Optional[np.ndarray]:
    """Per-second engagement in [0, 1], or None when no signal is available."""
    n = int(duration) + 1
    curve = np.zeros(n)
    have = False

    for seg in heatmap or []:
        a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
        curve[a:b] = np.maximum(curve[a:b], float(seg["value"]))
        have = True

    if chat_times:
        counts = np.bincount(np.clip(np.array(chat_times).astype(int), 0, n - 1), minlength=n).astype(float)
        kernel = np.ones(10) / 10.0  # 10 s moving average = messages/sec velocity
        vel = np.convolve(counts, kernel, mode="same")
        p95 = np.percentile(vel, 95) or 1.0
        curve = np.maximum(curve, np.clip(vel / p95, 0, 1))
        have = True

    return curve if have else None


def window_value(curve: Optional[np.ndarray], start: float, end: float) -> Optional[float]:
    if curve is None:
        return None
    seg = curve[int(start): int(end) + 1]
    if seg.size == 0:
        return 0.0
    k = max(1, int(seg.size * 0.2))
    return float(np.sort(seg)[-k:].mean())
