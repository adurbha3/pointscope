import pytest

from pointscope.geometry import Box3D
from pointscope.metrics import evaluate_sequence, match_frame


def _box(x=0.0, y=0.0, z=0.0, l=2.0, w=2.0, h=2.0, yaw=0.0, label="vehicle"):
    return Box3D(center=[x, y, z], dims=[l, w, h], yaw=yaw, label=label)


def test_match_frame_perfect_match():
    gt = [_box(0, 0, 0)]
    pred = [_box(0, 0, 0)]
    result = match_frame(gt, pred)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mean_iou == pytest.approx(1.0)


def test_match_frame_missed_detection():
    gt = [_box(0, 0, 0), _box(20, 20, 0)]
    pred = [_box(0, 0, 0)]  # second gt box has no matching prediction
    result = match_frame(gt, pred)
    assert result.true_positives == 1
    assert result.false_negatives == 1
    assert result.false_positives == 0
    assert result.recall == pytest.approx(0.5)


def test_match_frame_false_positive():
    gt = [_box(0, 0, 0)]
    pred = [_box(0, 0, 0), _box(50, 50, 0)]  # spurious extra box far away
    result = match_frame(gt, pred)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == pytest.approx(0.5)


def test_match_frame_below_iou_threshold_counts_as_both_fp_and_fn():
    gt = [_box(0, 0, 0, l=2, w=2, h=2)]
    pred = [_box(5, 5, 0, l=2, w=2, h=2)]  # far enough that IoU is 0
    result = match_frame(gt, pred, iou_threshold=0.5)
    assert result.true_positives == 0
    assert result.false_negatives == 1
    assert result.false_positives == 1


def test_match_frame_empty_inputs():
    result = match_frame([], [])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 0)
    assert result.precision == 1.0 and result.recall == 1.0


def test_match_frame_greedy_prefers_higher_iou_pairing():
    # one GT box has two candidate predictions; the better-overlapping one
    # should be matched, leaving the other as a false positive.
    gt = [_box(0, 0, 0, l=2, w=2, h=2)]
    pred_good = _box(0.1, 0.0, 0, l=2, w=2, h=2)
    pred_bad = _box(0.9, 0.0, 0, l=2, w=2, h=2)
    result = match_frame(gt, [pred_bad, pred_good], iou_threshold=0.3)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.matched_ious[0] > 0.3


def test_evaluate_sequence_aggregates_across_frames():
    class _F:
        def __init__(self, gt, pred):
            self.gt_boxes = gt
            self.pred_boxes = pred

    frames = [
        _F([_box(0, 0, 0)], [_box(0, 0, 0)]),  # perfect
        _F([_box(0, 0, 0)], []),  # miss
    ]
    metrics = evaluate_sequence(frames)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert len(metrics.per_frame) == 2
