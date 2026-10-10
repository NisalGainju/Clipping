# Viral Clipper

Turn a long-form video (YouTube, Twitch VOD, Kick, or a local file) into ranked, captioned, vertical (9:16) short clips for TikTok / Shorts / Reels.

```
URL -> download -> transcribe (word-level) -> candidate windows -> signal scoring
    -> LLM ranking -> merge/select -> cut -> face-tracked reframe -> animated captions -> MP4 + metadata
```

## Quick start (local, no GPU)

Requires Python 3.10+ and `ffmpeg` on PATH (`sudo apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[all]"
clipper doctor                                          # checks ffmpeg, packages, CUDA, API key

export ANTHROPIC_API_KEY=...                            # optional; enables LLM ranking
clipper run "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-model small --clips 8
```

**No GPU?** Transcription is the slow step. On CPU use `--whisper-model base` or `small` (roughly real-time or slower),
or run the same CLI on Colab with a free T4: see `notebooks/colab_runner.ipynb`. Typical split: develop in VS Code,
push to Git/Drive, run heavy jobs on Colab. `workspace/` caches downloads and transcripts, so reruns are cheap.

Fast iteration on selection only (no rendering):

```bash
clipper run VIDEO --no-render        # writes workspace/outputs/<id>/clips.json
```

## Web UI

```bash
clipper serve --port 8000            # open http://127.0.0.1:8000
```
Paste a URL (or upload a file) -> progress bar -> ranked clip cards with preview, score, title, hashtags, timestamps and download.
API: `POST /api/jobs`, `GET /api/jobs/{id}`, `POST /api/upload`; clips are served under `/files/`.

## Output

`workspace/outputs/<video_id>/clip_01.mp4, clip_01.jpg, ..., clips.json`. Each clip entry has rank, 0-100 virality score,
sub-scores (hook / emotion / shareability / independence), title, on-screen hook text, hashtags, description, start/end in the source, and the file paths.

## Configuration

`configs/default.yaml` (every field of `clipper/config.py`); CLI flags override it. Useful ones:

| Setting | Meaning |
|---|---|
| `whisper_model` | `large-v3` / `distil-large-v3` (GPU), `small` / `base` (CPU) |
| `min_clip_s`, `max_clip_s`, `target_clip_s` | clip length window (default 20-60s, target 40s) |
| `num_clips` | `null` = auto, about one per 7 minutes of source (3-25) |
| `llm_provider` | `auto` (Anthropic if `ANTHROPIC_API_KEY` is set, else heuristics) |
| `caption_style` | `pop` (word highlight), `clean`, `boxed` |
| `music_path`, `music_volume` | optional background bed, ducked under speech |
| `video_codec` | `h264_nvenc` on NVIDIA machines for faster encoding |
| `cookies_file` | cookies.txt for private / age-restricted / bot-checked videos |

## Project layout

```
clipper/
  config.py, models.py, errors.py, pipeline.py, cli.py
  ingest/      downloader.py (yt-dlp, local files, error mapping), engagement.py (YouTube heatmap, Twitch chat)
  transcribe/  asr.py (faster-whisper, word timestamps, VAD)
  analysis/    audio.py (energy/jumps), nlp.py (hook, emotion, opinion...), candidates.py, visual.py (motion/cuts)
  scoring/     llm_scorer.py (rubric prompt + heuristic fallback), select.py (merge + rank)
  render/      captions.py (ASS), reframe.py (face tracking + smoothing), exporter.py (cut->render->thumbnail)
  api/         app.py (FastAPI), static/index.html
configs/ notebooks/ tests/ docs/ examples/
```
See `docs/PIPELINE.md` for how each stage works and where to extend it.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
Covers sentence splitting, candidate generation, scoring/selection, caption generation, engagement curves.

## Known limitations / next steps

- Speaker tracking follows the largest face with stickiness; it does not do true active-speaker detection (lip motion / audio-visual). Good for talking heads and podcasts, weaker for multi-person gameplay with webcams.
- Laughter is detected via audio energy bursts and transcript tokens, not a trained sound classifier (YAMNet is the natural upgrade).
- Gameplay streams without a face fall back to a centre crop. Add a "blurred background + full frame" layout for those.
- Jobs run in-process, one at a time. Swap `ThreadPoolExecutor` in `api/app.py` for RQ/Celery for multi-user use.
- Not yet implemented: manual boundary editing / re-scoring in the UI, live-stream clipping, auto-publishing, feedback loop.
- Only process content you have the right to use, and respect each platform's Terms of Service.
