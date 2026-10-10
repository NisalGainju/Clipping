# Example runs

Generate the three example outputs the spec asks for (use videos you have rights to):

```bash
clipper run "<podcast URL>"          --work-dir workspace --clips 10    # podcast
clipper run "<gameplay stream VOD>"  --work-dir workspace --clips 12    # gameplay stream (Twitch gives chat signal)
clipper run "<talking-head URL>"     --work-dir workspace --clips 8     # talking-head YouTube video
```

Results land in `workspace/outputs/<video_id>/` (MP4s, thumbnails, `clips.json`). Copy the folders here to keep them with the repo.
