#!/usr/bin/env python3
"""Render a single frame of the synthetic demo sequence to a PNG, headlessly.

Used to regenerate the README hero screenshot, and handy on its own for
grabbing a quick reference image of any frame without opening the
interactive viewer -- e.g. to drop into a bug report or a PR description.

    python scripts/render_preview.py --frame 8 --out assets/pointscope_hero.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointscope.data import generate_synthetic_sequence  # noqa: E402
from pointscope.geometry import OrbitCamera  # noqa: E402
from pointscope.offscreen import save_frame_png  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=8, help="Frame index in the synthetic sequence")
    parser.add_argument("--num-frames", type=int, default=15, help="Total frames to generate before picking one")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="assets/pointscope_hero.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=0.9)
    parser.add_argument("--elevation", type=float, default=0.55)
    parser.add_argument("--distance", type=float, default=38.0)
    parser.add_argument("--point-size", type=float, default=3.5)
    args = parser.parse_args()

    frames = generate_synthetic_sequence(num_frames=args.num_frames, seed=args.seed)
    frame = frames[min(args.frame, len(frames) - 1)]
    camera = OrbitCamera(distance=args.distance, elevation=args.elevation, azimuth=args.azimuth)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_frame_png(frame, str(out_path), width=args.width, height=args.height, camera=camera, point_size=args.point_size)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
