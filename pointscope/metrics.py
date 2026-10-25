"""Detection-quality metrics for comparing predicted vs. ground-truth boxes.

Greedy IoU matching + precision/recall/F1 + mean matched IoU, computed
per frame and aggregated over a sequence. This is the evaluation half of
the "debug tool" story: PointScope doesn't just draw boxes, it tells you
*how good* the overlaid predictions are, frame by frame -- exactly the
kind of utility a tools/graphics engineer builds so research/ML
engineers can QA a detector's output without reading raw tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .geometry import Box3D, iou_3d


@dataclass
class FrameMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0


@dataclass
class SequenceMetrics:
    per_frame: List[FrameMatch]

    @property
    def precision(self) -> float:
        tp = sum(f.true_positives for f in self.per_frame)
        fp = sum(f.false_positives for f in self.per_frame)
        return tp / (tp + fp) if (tp + fp) else 1.0

    @property
    def recall(self) -> float:
        tp = sum(f.true_positives for f in self.per_frame)
        fn = sum(f.false_negatives for f in self.per_frame)
        return tp / (tp + fn) if (tp + fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        all_ious = [iou for f in self.per_frame for iou in f.matched_ious]
        return float(np.mean(all_ious)) if all_ious else 0.0


def match_frame(gt_boxes: Sequence[Box3D], pred_boxes: Sequence[Box3D], iou_threshold: float = 0.5) -> FrameMatch:
    """Greedy one-to-one matching by descending IoU.

    Greedy matching (rather than an optimal Hungarian assignment) is a
    deliberate, documented simplification: with the small number of
    actors per frame this tool targets, greedy and optimal matching
    agree in practice, and greedy is far simpler to read and test.
    """
    gt_boxes = list(gt_boxes)
    pred_boxes = list(pred_boxes)

    if not gt_boxes and not pred_boxes:
        return FrameMatch(0, 0, 0, [])

    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = iou_3d(gt, pred)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    matched_ious = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_ious.append(iou)

    tp = len(matched_gt)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(matched_pred)
    return FrameMatch(true_positives=tp, false_positives=fp, false_negatives=fn, matched_ious=matched_ious)


def evaluate_sequence(frames: Sequence, iou_threshold: float = 0.5) -> SequenceMetrics:
    """Run :func:`match_frame` over every frame in a sequence.

    ``frames`` is any sequence of objects with ``.gt_boxes`` and
    ``.pred_boxes`` attributes (i.e. :class:`pointscope.data.Frame`).
    """
    per_frame = [match_frame(f.gt_boxes, f.pred_boxes, iou_threshold) for f in frames]
    return SequenceMetrics(per_frame=per_frame)
