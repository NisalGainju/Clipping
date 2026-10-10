from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Transcript:
    language: str
    duration: float
    words: list[Word]

    def to_dict(self) -> dict:
        return {"language": self.language, "duration": self.duration,
                "words": [asdict(w) for w in self.words]}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(d["language"], d["duration"], [Word(**w) for w in d["words"]])

    def words_in(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]


@dataclass
class VideoSource:
    id: str
    title: str
    video_path: str
    duration: float
    url: Optional[str] = None
    platform: str = "local"
    heatmap: list[dict] = field(default_factory=list)  # YouTube "most replayed"


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    features: dict
    heuristic: float
    llm: Optional[dict] = None
    final: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Clip:
    rank: int
    start: float
    end: float
    score: float  # 0-100 virality score
    dims: dict    # hook / emotion / shareability / independence (0-10)
    title: str
    hook_text: str
    hashtags: list[str]
    description: str
    reason: str
    text: str
    video_path: Optional[str] = None
    thumbnail: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d
