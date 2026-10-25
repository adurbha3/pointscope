import numpy as np
import pytest

from pointscope.data import generate_synthetic_sequence, load_frames_from_dir, save_frames_to_dir


def test_generate_synthetic_sequence_shapes():
    frames = generate_synthetic_sequence(num_frames=5, seed=1)
    assert len(frames) == 5
    for frame in frames:
        assert frame.points.ndim == 2 and frame.points.shape[1] == 4
        assert len(frame.points) > 0
        assert len(frame.gt_boxes) > 0  # the demo scene always has actors


def test_generate_synthetic_sequence_is_deterministic():
    a = generate_synthetic_sequence(num_frames=8, seed=42)
    b = generate_synthetic_sequence(num_frames=8, seed=42)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.points, fb.points)
        assert len(fa.gt_boxes) == len(fb.gt_boxes)
        for ba, bb in zip(fa.gt_boxes, fb.gt_boxes):
            assert np.allclose(ba.center, bb.center)


def test_generate_synthetic_sequence_different_seeds_differ():
    a = generate_synthetic_sequence(num_frames=3, seed=1)
    b = generate_synthetic_sequence(num_frames=3, seed=2)
    assert not np.array_equal(a[0].points, b[0].points)


def test_detector_noise_knobs_change_prediction_count():
    rng_frames = generate_synthetic_sequence(num_frames=20, seed=3, detector_miss_rate=0.9, detector_false_positive_rate=0.0)
    total_preds = sum(len(f.pred_boxes) for f in rng_frames)
    total_gt = sum(len(f.gt_boxes) for f in rng_frames)
    # with a 90% miss rate, predictions should be far sparser than ground truth
    assert total_preds < total_gt * 0.4


def test_frame_rejects_malformed_points():
    from pointscope.data import Frame

    with pytest.raises(ValueError):
        Frame(points=np.zeros((5, 3)))  # missing the intensity column


def test_save_and_load_round_trip(tmp_path):
    frames = generate_synthetic_sequence(num_frames=4, seed=0)
    save_frames_to_dir(frames, tmp_path)

    loaded = load_frames_from_dir(tmp_path)
    assert len(loaded) == len(frames)
    for original, restored in zip(frames, loaded):
        assert np.allclose(original.points, restored.points)
        assert len(original.gt_boxes) == len(restored.gt_boxes)
        assert len(original.pred_boxes) == len(restored.pred_boxes)
        for ob, rb in zip(original.gt_boxes, restored.gt_boxes):
            assert np.allclose(ob.center, rb.center)
            assert np.allclose(ob.dims, rb.dims)
            assert ob.yaw == pytest.approx(rb.yaw)
            assert ob.label == rb.label


def test_load_frames_from_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frames_from_dir(tmp_path)
