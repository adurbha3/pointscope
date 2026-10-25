import numpy as np
import pytest

from pointscope.data import generate_synthetic_sequence
from tests.conftest import requires_gl


@requires_gl
def test_render_frame_produces_image_with_expected_shape():
    from pointscope.offscreen import render_frame_to_array

    frames = generate_synthetic_sequence(num_frames=1, seed=0)
    image, renderer = render_frame_to_array(frames[0], width=320, height=180)

    assert image.shape == (180, 320, 3)
    assert image.dtype == np.uint8
    assert renderer.stats.point_count == len(frames[0].points)
    assert renderer.stats.gt_box_count == len(frames[0].gt_boxes)
    assert renderer.stats.draw_calls >= 1


@requires_gl
def test_render_frame_is_not_blank():
    from pointscope.offscreen import render_frame_to_array

    frames = generate_synthetic_sequence(num_frames=1, seed=0)
    image, _ = render_frame_to_array(frames[0], width=320, height=180)

    background = np.array([int(0.04 * 255), int(0.04 * 255), int(0.06 * 255)])
    non_background = np.abs(image.astype(int) - background).sum(axis=-1) > 10
    # a real scene should light up a meaningful fraction of pixels (points + box lines)
    assert non_background.mean() > 0.01


@requires_gl
def test_render_handles_empty_points_and_boxes_gracefully():
    from pointscope.data import Frame
    from pointscope.offscreen import render_frame_to_array

    empty_frame = Frame(points=np.zeros((0, 4), dtype=np.float32), gt_boxes=[], pred_boxes=[])
    image, renderer = render_frame_to_array(empty_frame, width=160, height=90)

    assert image.shape == (90, 160, 3)
    assert renderer.stats.point_count == 0
    assert renderer.stats.draw_calls == 0


@requires_gl
def test_render_toggling_overlays_changes_output():
    from pointscope.offscreen import render_frame_to_array

    frames = generate_synthetic_sequence(num_frames=1, seed=0)
    frame = frames[0]

    with_boxes, _ = render_frame_to_array(frame, width=320, height=180, show_gt=True, show_pred=True)
    without_boxes, _ = render_frame_to_array(frame, width=320, height=180, show_gt=False, show_pred=False)

    assert not np.array_equal(with_boxes, without_boxes)


@requires_gl
def test_color_mode_changes_point_colors():
    from pointscope.offscreen import render_frame_to_array

    frames = generate_synthetic_sequence(num_frames=1, seed=0)
    frame = frames[0]

    height_mode, _ = render_frame_to_array(frame, width=320, height=180, color_mode=0, show_gt=False, show_pred=False)
    intensity_mode, _ = render_frame_to_array(frame, width=320, height=180, color_mode=1, show_gt=False, show_pred=False)

    assert not np.array_equal(height_mode, intensity_mode)
