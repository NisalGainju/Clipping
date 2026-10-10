"""Candidate window generation: sentence-aligned windows scored with cheap signals."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Config
from ..ingest.engagement import window_value
from ..models import Candidate, Transcript, Word
from .audio import AudioFeatures
from .nlp import text_features

END_PUNCT = (".", "?", "!", "…")


def split_sentences(words: list[Word], gap: float = 0.8) -> list[list[Word]]:
    """Split on terminal punctuation or a pause longer than `gap` seconds."""
    sents: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt_gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 999.0
        if w.text.endswith(END_PUNCT) or nxt_gap > gap:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def heuristic_score(features: dict, weights: dict) -> float:
    avail = {k: features[k] for k in weights if features.get(k) is not None}
    total = sum(weights[k] for k in avail) or 1.0
    score = sum(weights[k] * v for k, v in avail.items()) / total
    return max(0.0, score - 0.15 * features.get("filler", 0.0))


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(inter, 0.0) / max(min(a.duration, b.duration), 1e-6)


def generate_candidates(tr: Transcript, audio: Optional[AudioFeatures], curve: Optional[np.ndarray],
                        cfg: Config, top_k: int) -> list[Candidate]:
    sents = split_sentences(tr.words)
    raw: list[Candidate] = []
    for i, first in enumerate(sents):
        s = first[0].start
        ends = []
        for j in range(i, len(sents)):
            d = sents[j][-1].end - s
            if d > cfg.max_clip_s:
                break
            if d >= cfg.min_clip_s:
                ends.append(j)
        if not ends:
            continue
        picks = {ends[0], ends[-1],
                 min(ends, key=lambda j: abs(sents[j][-1].end - s - cfg.target_clip_s))}
        for j in picks:
            words = [w for sent in sents[i:j + 1] for w in sent]
            e = words[-1].end
            text = " ".join(w.text for w in words)
            f = text_features(text, e - s)
            if audio is not None:
                a = audio.window(s, e)
                f["audio"] = a["score"]
            f["engagement"] = window_value(curve, s, e)
            raw.append(Candidate(start=s, end=e, text=text, features=f,
                                 heuristic=heuristic_score(f, cfg.heuristic_weights)))

    raw.sort(key=lambda c: c.heuristic, reverse=True)
    kept: list[Candidate] = []
    for c in raw:
        if all(_overlap(c, k) < 0.5 for k in kept):
            kept.append(c)
        if len(kept) >= top_k:
            break
    return kept
