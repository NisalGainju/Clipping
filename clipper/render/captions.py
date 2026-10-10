"""Word-level animated captions as ASS subtitles (burned in with ffmpeg/libass)."""
from __future__ import annotations

import textwrap
from typing import Optional

from ..models import Word

HIGHLIGHT = "&H0000E5FF&"  # yellow (ASS colours are &HBBGGRR&)
WHITE = "&H00FFFFFF&"

# name -> (fontsize, outline colour, outline width, border style, back colour, uppercase, highlight)
STYLES = {
    "pop":   dict(size=88, outline="&H00000000", ow=7, border=1, back="&H00000000", upper=True, hl=HIGHLIGHT),
    "clean": dict(size=72, outline="&H00000000", ow=4, border=1, back="&H00000000", upper=False, hl="&H00FFFFFF&"),
    "boxed": dict(size=72, outline="&H00000000", ow=2, border=3, back="&HA0000000", upper=False, hl="&H0000E5FF&"),
}


def ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[Word], max_words: int = 3, max_chars: int = 22, gap: float = 0.6) -> list[list[Word]]:
    lines: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        too_long = cur and (len(cur) >= max_words or sum(len(x.text) + 1 for x in cur) + len(w.text) > max_chars)
        paused = cur and (w.start - cur[-1].end) > gap
        if too_long or paused:
            lines.append(cur)
            cur = []
        cur.append(w)
        if w.text.endswith((".", "?", "!")) and len(cur) >= 2:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def build_ass(words: list[Word], clip_start: float, clip_end: float, *, style: str = "pop",
              font: str = "Arial", width: int = 1080, height: int = 1920, words_per_line: int = 3,
              hook_text: Optional[str] = None, hook_seconds: float = 2.8) -> str:
    st = STYLES.get(style, STYLES["pop"])
    dur = clip_end - clip_start
    rel = [Word(max(w.start - clip_start, 0.0), min(w.end - clip_start, dur), w.text, w.prob)
           for w in words if w.end > clip_start and w.start < clip_end]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{st['size']},{WHITE},{WHITE},{st['outline']},{st['back']},-1,0,0,0,100,100,0,0,{st['border']},{st['ow']},0,2,60,60,{int(height * 0.24)},1
Style: Hook,{font},{int(st['size'] * 0.8)},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&HB4000000&,-1,0,0,0,100,100,0,0,3,10,0,8,70,70,{int(height * 0.10)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    if hook_text:
        wrapped = "\\N".join(textwrap.wrap(_escape(hook_text).upper(), 22))
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(hook_seconds, dur))},Hook,,0,0,0,,{{\\fad(150,250)}}{wrapped}")

    lines = group_words(rel, max_words=words_per_line)
    for li, line in enumerate(lines):
        line_end_limit = lines[li + 1][0].start if li + 1 < len(lines) else dur
        for wi, w in enumerate(line):
            start = w.start
            end = line[wi + 1].start if wi + 1 < len(line) else min(w.end + 0.25, line_end_limit, dur)
            if end <= start:
                end = start + 0.05
            parts = []
            for k, x in enumerate(line):
                t = _escape(x.text.upper() if st["upper"] else x.text)
                if k == wi:
                    parts.append(f"{{\\1c{st['hl']}\\fscx112\\fscy112}}{t}{{\\1c{WHITE}\\fscx100\\fscy100}}")
                else:
                    parts.append(t)
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(events) + "\n"
