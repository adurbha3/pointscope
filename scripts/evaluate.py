#!/usr/bin/env python3
"""Evaluate detector predictions against ground truth over a sequence.

Prints a per-frame table plus aggregate precision / recall / F1 / mean IoU,
and with --csv writes the per-frame table out for further analysis. This
is the kind of quick QA pass a tools/graphics engineer wires up so a
researcher can sanity-check a detector's output without digging through
raw tensors -- here it defaults to the bundled synthetic sequence, or
point it at a real capture with --data.

    python scripts/evaluate.py --num-frames 60 --iou-threshold 0.5
    python scripts/evaluate.py --data path/to/frames --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointscope.data import generate_synthetic_sequence, load_frames_from_dir  # noqa: E402
from pointscope.metrics import evaluate_sequence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Directory of frame_*.npy/.json to evaluate")
    parser.add_argument("--num-frames", type=int, default=60, help="Frame count if using the synthetic sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write a per-frame CSV report")
    args = parser.parse_args()

    frames = load_frames_from_dir(args.data) if args.data else generate_synthetic_sequence(
        num_frames=args.num_frames, seed=args.seed
    )

    metrics = evaluate_sequence(frames, iou_threshold=args.iou_threshold)

    print(f"{'frame':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>8} {'mean_iou':>9}")
    for i, fm in enumerate(metrics.per_frame):
        print(f"{i:>6} {fm.true_positives:>4} {fm.false_positives:>4} {fm.false_negatives:>4} "
              f"{fm.precision:>10.3f} {fm.recall:>8.3f} {fm.mean_iou:>9.3f}")

    print("-" * 58)
    print(f"Aggregate over {len(frames)} frames (IoU >= {args.iou_threshold}):")
    print(f"  precision = {metrics.precision:.3f}")
    print(f"  recall    = {metrics.recall:.3f}")
    print(f"  f1        = {metrics.f1:.3f}")
    print(f"  mean IoU  = {metrics.mean_iou:.3f} (over matched pairs)")

    if args.csv:
        out_path = Path(args.csv)
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame", "true_positives", "false_positives", "false_negatives", "precision", "recall", "mean_iou"])
            for i, fm in enumerate(metrics.per_frame):
                writer.writerow([i, fm.true_positives, fm.false_positives, fm.false_negatives, fm.precision, fm.recall, fm.mean_iou])
        print(f"\nWrote per-frame CSV report to {out_path}")


if __name__ == "__main__":
    main()
