from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .errors import ClipperError


def _progress(stage: str, frac: float, msg: str) -> None:
    bar = "#" * int(frac * 30)
    sys.stderr.write(f"\r[{bar:<30}] {frac * 100:5.1f}%  {stage:<10} {msg[:50]:<50}")
    sys.stderr.flush()


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    cfg = load_config(args.config, {
        "work_dir": args.work_dir, "num_clips": args.clips, "min_clip_s": args.min_len,
        "max_clip_s": args.max_len, "whisper_model": args.whisper_model, "language": args.language,
        "caption_style": args.caption_style, "llm_provider": args.llm, "cookies_file": args.cookies,
        "music_path": args.music, "device": args.device,
        "reframe": False if args.no_reframe else None,
        "captions": False if args.no_captions else None,
        "render_clips": False if args.no_render else None,
    })
    try:
        result = Pipeline(cfg, _progress).run(args.source)
    except ClipperError as e:
        sys.stderr.write(f"\nError: {e}\n")
        return 2
    sys.stderr.write("\n")
    print(f"\n{result['video']['title']}  ({result['scoring']} scoring)")
    for c in result["clips"]:
        print(f"#{c['rank']:>2}  score {c['score']:5.1f}  {c['start']:8.1f}s-{c['end']:8.1f}s  {c['title']}")
        if c.get("video_path"):
            print(f"      {c['video_path']}")
    print(f"\nMetadata: {cfg.output_dir / result['video']['id'] / 'clips.json'}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os
    import uvicorn

    if args.config:
        os.environ["CLIPPER_CONFIG"] = args.config
    if args.work_dir:
        os.environ["CLIPPER_WORK_DIR"] = args.work_dir
    uvicorn.run("clipper.api.app:app", host=args.host, port=args.port)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib
    import os
    from .utils.ffmpeg import have_ffmpeg

    print("ffmpeg/ffprobe:", "OK" if have_ffmpeg() else "MISSING")
    for mod in ("yt_dlp", "faster_whisper", "cv2", "numpy", "yaml", "anthropic", "mediapipe", "fastapi", "chat_downloader"):
        try:
            importlib.import_module(mod)
            print(f"{mod}: OK")
        except ImportError:
            print(f"{mod}: missing")
    try:
        import ctranslate2
        print("CUDA devices:", ctranslate2.get_cuda_device_count())
    except Exception:
        print("CUDA devices: unknown")
    print("ANTHROPIC_API_KEY:", "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set (heuristic scoring only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipper", description="Long video -> ranked vertical short clips")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Process a URL or local file")
    r.add_argument("source", help="Video URL (YouTube/Twitch/Kick/...) or local file path")
    r.add_argument("--config", default=None)
    r.add_argument("--work-dir", default=None)
    r.add_argument("--clips", type=int, default=None, help="Number of clips (default: auto)")
    r.add_argument("--min-len", type=float, default=None)
    r.add_argument("--max-len", type=float, default=None)
    r.add_argument("--whisper-model", default=None, help="tiny/base/small/medium/large-v3/distil-large-v3")
    r.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    r.add_argument("--language", default=None)
    r.add_argument("--caption-style", default=None, choices=["pop", "clean", "boxed"])
    r.add_argument("--llm", default=None, choices=["auto", "anthropic", "heuristic"])
    r.add_argument("--cookies", default=None, help="cookies.txt for private/age-restricted videos")
    r.add_argument("--music", default=None, help="Background music file (ducked under speech)")
    r.add_argument("--no-reframe", action="store_true")
    r.add_argument("--no-captions", action="store_true")
    r.add_argument("--no-render", action="store_true", help="Only select + score moments (fast; writes clips.json)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="Start the web UI / API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--config", default=None)
    s.add_argument("--work-dir", default=None)
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("doctor", help="Check the environment")
    d.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
