# Pipeline details

## 1. Ingest (`ingest/`)
`acquire()` accepts a URL or local path. yt-dlp downloads <= `max_height` MP4. Failures (private, age-restricted, geo-blocked,
bot check, removed) are mapped to readable `VideoUnavailableError` messages. YouTube's "most replayed" heatmap comes from the
yt-dlp info dict; Twitch chat timestamps come from `chat-downloader` (optional).
`build_engagement_curve()` merges both into a per-second 0-1 curve.

## 2. Transcribe (`transcribe/asr.py`)
16 kHz mono WAV -> faster-whisper with `word_timestamps=True` and VAD. Segments stream lazily, so memory is flat for multi-hour audio.
The transcript is cached as `workspace/cache/<id>/transcript_<model>.json`.

## 3. Candidates (`analysis/`)
1. Words -> sentences (punctuation or pause > 0.8s).
2. For each sentence start, build windows ending at sentence boundaries that fit `[min_clip_s, max_clip_s]` (shortest, closest to target, longest).
   Clips therefore always start and end on complete sentences.
3. Each window gets: NLP features (hook, emotion, opinion, quotable, pace, filler, completeness), audio score (loudness spikes, sudden changes), engagement score.
4. A weighted heuristic score (weights in config; missing signals are dropped and weights renormalised) ranks windows.
5. Non-maximum suppression removes windows overlapping > 50%. The top `num_clips * candidate_multiplier` go to scoring.

## 4. LLM scoring (`scoring/llm_scorer.py`)
Batches of candidates are sent with a harsh editor rubric; the model returns JSON with hook / emotion / shareability / independence (0-10),
title, on-screen hook text, hashtags, description and reason. `virality = 100 * weighted mean`, with a 0.6x penalty if independence < 4.
Final = `llm_blend * llm + (1 - llm_blend) * heuristic`. Results are cached in `llm_scores.json`. No API key -> heuristic-only fallback.

## 5. Selection (`scoring/select.py`)
Sort by final score; overlapping windows are merged when the union still fits `max_clip_s` and scores are close, otherwise the lower one is dropped.
Small head/tail padding is applied. Output is ranked 1..N.

## 6. Render (`render/`)
`cut` (frame-accurate re-encode) -> `plan_crop` (face detection at ~5 fps, dead-zone + EMA + speed limit smoothing) ->
`render_tracked` (OpenCV crops each frame, piped to ffmpeg which burns ASS captions, mixes optional ducked music, encodes H.264/AAC).
Non-landscape sources or `--no-reframe` use `render_simple` (pure ffmpeg centre-crop / letterbox).

## Extension points
- New signal: add a feature in `analysis/candidates.py` and a weight in `config.heuristic_weights`.
- New LLM provider: implement `_call_*` in `llm_scorer.py` and branch in `resolve_provider`.
- New caption style: add an entry to `STYLES` in `render/captions.py`.
- Better framing: replace `FaceDetector`/`plan_crop` with an active-speaker model.
