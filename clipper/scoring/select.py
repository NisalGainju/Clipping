"""Final selection: merge overlapping windows and keep the top N, ranked."""
from __future__ import annotations

from ..config import Config
from ..models import Candidate, Clip, Transcript


def auto_num_clips(duration_s: float) -> int:
    return max(3, min(25, round(duration_s / 60.0 / 7.0)))


def select_clips(cands: list[Candidate], n: int, cfg: Config, tr: Transcript, duration: float) -> list[Clip]:
    pool = sorted(cands, key=lambda c: c.final, reverse=True)
    chosen: list[dict] = []  # {start,end,cand}
    for c in pool:
        merged = False
        for sel in chosen:
            inter = min(c.end, sel["end"]) - max(c.start, sel["start"])
            if inter <= 0:
                continue
            union_s, union_e = min(c.start, sel["start"]), max(c.end, sel["end"])
            # Merge near-equal overlapping windows into one clean clip if it still fits the length limit.
            if union_e - union_s <= cfg.max_clip_s and c.final >= 0.9 * sel["cand"].final:
                sel["start"], sel["end"] = union_s, union_e
            merged = True  # overlapping either way: never emit both
            break
        if not merged:
            chosen.append({"start": c.start, "end": c.end, "cand": c})
    chosen.sort(key=lambda s: s["cand"].final, reverse=True)
    chosen = chosen[:n]

    clips: list[Clip] = []
    for rank, sel in enumerate(chosen, start=1):
        c: Candidate = sel["cand"]
        llm = c.llm or {}
        start = max(0.0, sel["start"] - cfg.pad_start)
        end = min(duration, sel["end"] + cfg.pad_end)
        words = tr.words_in(sel["start"], sel["end"])
        clips.append(Clip(
            rank=rank, start=round(start, 2), end=round(end, 2), score=round(c.final, 1),
            dims={k: round(float(llm.get(k, 0)), 1) for k in ("hook", "emotion", "shareability", "independence")},
            title=llm.get("title") or "Highlight", hook_text=llm.get("hook_text") or "",
            hashtags=list(llm.get("hashtags") or []), description=llm.get("description") or "",
            reason=llm.get("reason") or "", text=" ".join(w.text for w in words)))
    return clips
