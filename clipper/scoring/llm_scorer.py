"""Rank candidate windows with an LLM (Anthropic), falling back to heuristics."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..config import Config
from ..models import Candidate

log = logging.getLogger(__name__)

DIM_WEIGHTS = {"hook": 0.30, "emotion": 0.25, "shareability": 0.25, "independence": 0.20}

SYSTEM_PROMPT = """You are a veteran short-form video editor who picks clips that go viral on TikTok, \
YouTube Shorts and Instagram Reels. You will receive candidate segments from a longer video, each with a \
transcript and cheap signal summaries (0-1 scale: audio energy, chat/replay engagement, emotion, etc.).

Score each candidate from 0 to 10 on:
- hook: does the first 3 seconds grab attention (surprising claim, question, tension, strong opinion)?
- emotion: emotional intensity / entertainment value (funny, shocking, inspiring, heated, a clear peak)?
- shareability: would someone send this to a friend? quotable, relatable, controversial-but-safe, novel?
- independence: does it make sense with zero context from the rest of the video, with a clear payoff and a clean ending?

Be harsh: most segments are mediocre. Reserve 8+ for genuinely strong clips. Reject setup-only segments, \
small talk, sponsor reads, and segments that depend on missing context.

For each candidate also write: a catchy title (max 70 chars, no clickbait lies), a short on-screen hook_text \
(max 8 words, shown for the first 3 seconds), 3-5 hashtags, a one-sentence description, and a one-sentence reason.

Return ONLY a JSON array, no markdown fences, one object per candidate:
[{"id": 0, "hook": 0-10, "emotion": 0-10, "shareability": 0-10, "independence": 0-10,
  "title": "...", "hook_text": "...", "hashtags": ["#..."], "description": "...", "reason": "..."}]"""


def resolve_provider(cfg: Config) -> str:
    if cfg.llm_provider == "auto":
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"
    return cfg.llm_provider


def _parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[a:b + 1])


def _payload(batch: list[tuple[int, Candidate]], title: str) -> str:
    items = []
    for idx, c in batch:
        sig = {k: round(v, 2) for k, v in c.features.items() if isinstance(v, (int, float)) and v is not None}
        items.append({"id": idx, "start_s": round(c.start, 1), "end_s": round(c.end, 1),
                      "duration_s": round(c.duration, 1), "transcript": c.text[:3500], "signals": sig})
    return f"Source video title: {title}\n\nCandidates:\n{json.dumps(items, ensure_ascii=False)}"


def _call_anthropic(batch: list[tuple[int, Candidate]], title: str, cfg: Config) -> dict[int, dict]:
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("CLIPPER_LLM_MODEL", cfg.llm_model)
    last: Exception | None = None
    for _ in range(2):  # one retry on malformed output
        try:
            resp = client.messages.create(model=model, max_tokens=4096, system=SYSTEM_PROMPT,
                                          messages=[{"role": "user", "content": _payload(batch, title)}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {int(r["id"]): r for r in _parse_json_array(text)}
        except Exception as e:
            last = e
            log.warning("LLM scoring batch failed: %s", e)
    log.error("Giving up on LLM batch: %s", last)
    return {}


def _heuristic_fallback(c: Candidate) -> dict:
    f = c.features
    first = c.text.split(". ")[0].strip()
    title = (first[:67] + "...") if len(first) > 70 else first
    return {
        "hook": 10 * f.get("hook", 0), "emotion": 10 * f.get("emotion", 0),
        "shareability": 10 * (0.5 * f.get("quotable", 0) + 0.3 * f.get("emotion", 0) + 0.2 * f.get("opinion", 0)),
        "independence": 10 * (0.6 * f.get("completeness", 0) + 0.4 * (1 - f.get("filler", 0))),
        "title": title or "Highlight", "hook_text": " ".join(c.text.split()[:7]),
        "hashtags": ["#shorts", "#fyp"], "description": title, "reason": "Heuristic score (no LLM available).",
    }


def virality(dims: dict) -> float:
    base = sum(DIM_WEIGHTS[k] * float(dims.get(k, 0)) for k in DIM_WEIGHTS) * 10.0  # 0-100
    if float(dims.get("independence", 10)) < 4:  # needs-context clips are penalised hard
        base *= 0.6
    return max(0.0, min(100.0, base))


def score_candidates(cands: list[Candidate], cfg: Config, title: str, cache: Optional[dict] = None) -> None:
    """Fills `c.llm` and `c.final` (0-100) in place. `cache` maps 'start-end' -> llm dict."""
    cache = cache if cache is not None else {}
    provider = resolve_provider(cfg)
    key = lambda c: f"{c.start:.2f}-{c.end:.2f}"

    todo = [(i, c) for i, c in enumerate(cands) if key(c) not in cache]
    if provider == "anthropic" and todo:
        bs = cfg.llm_batch_size
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            for result, batch in zip(ex.map(lambda b: _call_anthropic(b, title, cfg), batches), batches):
                for idx, c in batch:
                    if idx in result:
                        cache[key(c)] = result[idx]
    elif provider != "anthropic":
        log.info("Using heuristic scoring (set ANTHROPIC_API_KEY for LLM ranking).")

    for c in cands:
        llm = cache.get(key(c))
        h100 = c.heuristic * 100.0
        if llm:
            c.llm = llm
            c.final = cfg.llm_blend * virality(llm) + (1 - cfg.llm_blend) * h100
        else:
            c.llm = _heuristic_fallback(c)
            c.final = 0.6 * virality(c.llm) + 0.4 * h100
