"""CLI interface for OpenNoMark watermark removal."""

import argparse
import glob
import json
import os
import sys

from . import __version__
from .enhancer import MODEL_TYPES, NOISE_LEVELS, SCALES, Waifu2xSettings


def resolve_paths(inputs):
    """Resolve input paths: support files, directories, and globs."""
    paths = []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}

    for inp in inputs:
        if os.path.isdir(inp):
            for ext in valid_exts:
                paths.extend(glob.glob(os.path.join(inp, f"*{ext}")))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            expanded = glob.glob(inp)
            paths.extend(f for f in expanded if os.path.isfile(f))

    # Filter valid image extensions and deduplicate
    paths = list(dict.fromkeys(
        p for p in paths if os.path.splitext(p)[1].lower() in valid_exts
    ))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        prog="opennomark",
        description="AI watermark detection and removal",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Image files, directories, or glob patterns",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save detection debug images and masks",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Write one machine-readable JSON result to stdout",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default=None,
        help="Device: cpu, cuda, mps (default: auto)",
    )
    parser.add_argument(
        "--mode", choices=("watermark", "stains", "waifu2x"), default="watermark",
        help="watermark: detect and inpaint marks (default); "
             "stains: clean ChatGPT blotches in flat fills; "
             "waifu2x: denoise and/or upscale only",
    )
    parser.add_argument(
        "--upscale", action="store_true",
        help="Stains mode: run waifu2x after cleanup (settings below)",
    )
    parser.add_argument(
        "--waifu2x-model", choices=MODEL_TYPES, default=Waifu2xSettings.model,
        help="waifu2x model: art (flat graphics, default), art_scan, photo",
    )
    parser.add_argument(
        "--waifu2x-scale", type=int, choices=SCALES, default=Waifu2xSettings.scale,
        help="waifu2x upscale factor (default: %(default)s)",
    )
    parser.add_argument(
        "--waifu2x-noise", type=int, choices=NOISE_LEVELS, default=Waifu2xSettings.noise,
        help="waifu2x noise reduction: -1 off, 0-3 (default: %(default)s)",
    )

    args = parser.parse_args()
    paths = resolve_paths(args.inputs)

    if not paths:
        if args.json:
            print(json.dumps({
                "version": __version__,
                "status": "error",
                "error": "No valid images found.",
                "results": [],
            }))
        else:
            print("No valid images found.", file=sys.stderr)
        return 1

    if not args.json:
        print(f"Found {len(paths)} image(s) to process.")

    from .pipeline import WatermarkRemovalPipeline

    if args.upscale and args.mode != "stains":
        parser.error("--upscale requires --mode stains")
    try:
        waifu2x = Waifu2xSettings(args.waifu2x_model, args.waifu2x_scale, args.waifu2x_noise)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "watermark" or (args.mode == "stains" and not args.upscale):
        waifu2x = None

    device = args.device
    try:
        pipeline = WatermarkRemovalPipeline(
            device=device,
            verbose=not args.json,
            load_models=args.mode == "watermark",
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({
                "version": __version__,
                "status": "error",
                "error": str(exc),
                "results": [],
            }))
        else:
            print(f"Failed to load models: {exc}", file=sys.stderr)
        return 2

    def on_progress(i, total, meta):
        if args.json:
            return
        status = meta["status"]
        found = meta["watermarks_found"]
        name = os.path.basename(meta["input"])
        if status == "cleaned" and args.mode != "watermark":
            print(f"  [{i}/{total}] {name} -> {', '.join(meta['methods'])}")
        elif status == "cleaned":
            print(f"  [{i}/{total}] {name} -> {found} watermark(s) removed")
        else:
            print(f"  [{i}/{total}] {name} -> no watermark found")

    try:
        results = pipeline.process_batch(
            paths,
            args.output,
            save_debug=args.debug,
            callback=on_progress,
            mode=args.mode,
            waifu2x=waifu2x,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({
                "version": __version__,
                "status": "error",
                "error": str(exc),
                "results": [],
            }))
        else:
            print(f"Processing failed: {exc}", file=sys.stderr)
        return 2

    cleaned = sum(1 for r in results if r["status"] == "cleaned")
    skipped = sum(1 for r in results if r["status"] == "no_watermark")
    output_dir = os.path.abspath(args.output)
    for result in results:
        if result.get("output"):
            result["output"] = os.path.abspath(result["output"])

    if args.json:
        print(json.dumps({
            "version": __version__,
            "status": "ok",
            "output_dir": output_dir,
            "summary": {
                "total": len(results),
                "cleaned": cleaned,
                "skipped": skipped,
            },
            "results": results,
        }, ensure_ascii=False))
    else:
        print(f"\nDone! {cleaned} cleaned, {skipped} skipped. Output: {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
