"""Cheap, dependency-free linguistic features used to pre-rank candidate windows."""
from __future__ import annotations

import re

EMOTION = set("""amazing insane crazy unbelievable incredible terrible horrible awful wow omg shocked shocking
hate love angry furious scared terrified afraid hilarious funny ridiculous disgusting beautiful worst best
mad nuts wild unreal stupid idiot dead dying cry crying laugh laughing screaming scream damn hell""".split())
LAUGH = {"haha", "hahaha", "lol", "lmao", "hehe", "ha"}
SUPER = set("never always worst best biggest craziest only first last everyone nobody everything nothing".split())
FILLER = {"um", "uh", "uhm", "er", "erm", "like", "basically", "actually", "literally"}
OPINION_PHRASES = ["i think", "i believe", "honestly", "the problem is", "the truth is", "i hate", "i love",
                   "in my opinion", "to be honest", "the reality is", "people don't", "everyone is wrong",
                   "unpopular opinion", "i'm convinced", "here's the thing"]
HOOK_RE = re.compile(
    r"^(so|okay|ok|listen|look|imagine|guess what|you won't believe|here's|the (craziest|biggest|worst|best|"
    r"first|only|real)|nobody|everyone|what if|why (do|does|is|are)|how (do|does|to)|did you know|stop|never|"
    r"this is|let me tell you)\b")
CONTRAST_RE = re.compile(r"\b(not|isn't|aren't|wasn't|don't)\b[^.?!]{3,60}\b(but|it's|they're|you're)\b")
BAD_ENDINGS = {"and", "but", "so", "because", "or", "the", "a", "to", "of", "that", "which"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def text_features(text: str, duration: float) -> dict:
    low = text.lower()
    tokens = re.findall(r"[a-z']+", low)
    n = max(len(tokens), 1)
    first = " ".join(low.split()[:18])

    hook = 0.0
    if "?" in first:
        hook += 0.4
    if HOOK_RE.search(first):
        hook += 0.4
    if re.search(r"\d", first) or any(t in SUPER for t in first.split()):
        hook += 0.2

    laughs = sum(t in LAUGH for t in tokens)
    emo_hits = sum(t in EMOTION for t in tokens) + 2 * text.count("!") + 3 * laughs
    emotion = _clamp(emo_hits / (n / 20.0) / 3.0)

    opinion = _clamp(sum(low.count(p) for p in OPINION_PHRASES) / max(n / 50.0, 1.0) / 2.0)

    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    short = sum(4 <= len(s.split()) <= 16 for s in sentences)
    quotable = _clamp(0.5 * (short / max(len(sentences), 1)) + 0.5 * (1.0 if CONTRAST_RE.search(low) else 0.0))

    wps = n / max(duration, 1.0)
    pace = _clamp(1.0 - abs(wps - 3.0) / 3.0)
    filler = _clamp(sum(t in FILLER for t in tokens) / n * 10.0)

    stripped = text.strip()
    completeness = 0.0
    if stripped:
        completeness += 0.4 if stripped[0].isupper() else 0.0
        completeness += 0.4 if stripped[-1] in ".?!…" else 0.0
        completeness += 0.2 if tokens and tokens[-1] not in BAD_ENDINGS else 0.0

    return {"hook": _clamp(hook), "emotion": emotion, "opinion": opinion,
            "question": _clamp(text.count("?") / 3.0), "quotable": quotable,
            "pace": pace, "filler": filler, "completeness": completeness,
            "laughs": float(laughs)}
