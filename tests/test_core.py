import numpy as np

from clipper.analysis.candidates import generate_candidates, split_sentences
from clipper.analysis.nlp import text_features
from clipper.config import Config
from clipper.ingest.engagement import build_engagement_curve, window_value
from clipper.models import Transcript, Word
from clipper.render.captions import ass_time, build_ass, group_words
from clipper.scoring.llm_scorer import _parse_json_array, score_candidates, virality
from clipper.scoring.select import auto_num_clips, select_clips


def make_transcript(minutes=6):
    words, t = [], 0.0
    sentence = "Honestly this is the craziest thing I have ever seen in my life!".split()
    plain = "We went to the store and then we came back home again today.".split()
    n = 0
    while t < minutes * 60:
        src = sentence if n % 7 == 0 else plain
        for i, w in enumerate(src):
            txt = w if i < len(src) - 1 else w
            words.append(Word(t, t + 0.3, txt))
            t += 0.35
        t += 0.3
        n += 1
    return Transcript("en", t, words)


def test_split_sentences():
    tr = make_transcript(1)
    sents = split_sentences(tr.words)
    assert len(sents) > 3 and all(s[-1].text.endswith((".", "!")) for s in sents)


def test_candidates_respect_length_and_overlap():
    cfg = Config()
    tr = make_transcript()
    cands = generate_candidates(tr, None, None, cfg, top_k=9)
    assert cands
    for c in cands:
        assert cfg.min_clip_s <= c.duration <= cfg.max_clip_s
    assert len(cands) <= 9


def test_scoring_and_selection_without_llm():
    cfg = Config(llm_provider="heuristic")
    tr = make_transcript()
    cands = generate_candidates(tr, None, None, cfg, top_k=12)
    score_candidates(cands, cfg, "title")
    clips = select_clips(cands, 3, cfg, tr, tr.duration)
    assert 1 <= len(clips) <= 3
    assert [c.rank for c in clips] == list(range(1, len(clips) + 1))
    assert clips == sorted(clips, key=lambda c: -c.score)
    spans = sorted((c.start, c.end) for c in clips)
    assert all(a[1] <= b[0] + cfg.pad_start + cfg.pad_end for a, b in zip(spans, spans[1:]))


def test_virality_penalises_dependent_clips():
    good = {"hook": 8, "emotion": 8, "shareability": 8, "independence": 8}
    bad = dict(good, independence=2)
    assert virality(good) > virality(bad)


def test_parse_json_array_handles_fences():
    assert _parse_json_array('```json\n[{"id": 1}]\n```') == [{"id": 1}]


def test_text_features_hook_and_emotion():
    f = text_features("Why do people hate this? It's insane! Honestly the worst thing ever!", 10)
    assert f["hook"] > 0 and f["emotion"] > 0 and f["opinion"] > 0


def test_engagement_curve():
    heat = [{"start_time": 0, "end_time": 10, "value": 0.1}, {"start_time": 10, "end_time": 20, "value": 1.0}]
    curve = build_engagement_curve(30, heat, [])
    assert window_value(curve, 10, 20) > window_value(curve, 0, 10)
    assert build_engagement_curve(30, [], []) is None


def test_captions_ass():
    assert ass_time(3661.5) == "1:01:01.50"
    words = [Word(10 + i * 0.4, 10.3 + i * 0.4, f"word{i}") for i in range(8)]
    lines = group_words(words, max_words=3)
    assert all(len(l) <= 3 for l in lines)
    ass = build_ass(words, 9.5, 14.0, hook_text="You won't believe this")
    assert "[Events]" in ass and "Dialogue: 0" in ass and "Hook" in ass
    assert ass.count("Dialogue: 0") == 8  # one event per word


def test_auto_num_clips_bounds():
    assert auto_num_clips(600) == 3
    assert 8 <= auto_num_clips(3600) <= 9
    assert auto_num_clips(4 * 3600) == 25
