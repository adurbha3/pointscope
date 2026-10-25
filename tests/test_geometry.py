import numpy as np
import pytest

from pointscope.geometry import (
    Box3D,
    OrbitCamera,
    bev_iou,
    iou_3d,
    look_at,
    obb_bev_polygon,
    obb_corners,
    perspective_matrix,
    polygon_intersection_area,
)


def _rect_poly(cx, cy, w, h):
    return np.array(
        [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]],
        dtype=np.float64,
    )


def test_polygon_intersection_identical_square():
    poly = _rect_poly(0, 0, 2, 2)
    assert polygon_intersection_area(poly, poly) == pytest.approx(4.0)


def test_polygon_intersection_disjoint():
    a = _rect_poly(0, 0, 2, 2)
    b = _rect_poly(100, 100, 2, 2)
    assert polygon_intersection_area(a, b) == 0.0


def test_polygon_intersection_half_overlap():
    a = _rect_poly(0, 0, 2, 2)
    b = _rect_poly(1, 0, 2, 2)  # shifted by half its width
    # overlap region is a 1x2 rectangle
    assert polygon_intersection_area(a, b) == pytest.approx(2.0)


def test_bev_iou_identical_boxes_is_one():
    box = Box3D(center=[1, 2, 0], dims=[3, 2, 1], yaw=0.5)
    assert bev_iou(box, box) == pytest.approx(1.0)


def test_bev_iou_disjoint_boxes_is_zero():
    a = Box3D(center=[0, 0, 0], dims=[2, 2, 2], yaw=0.0)
    b = Box3D(center=[50, 50, 0], dims=[2, 2, 2], yaw=0.0)
    assert bev_iou(a, b) == 0.0


def test_bev_iou_known_half_overlap():
    a = Box3D(center=[0, 0, 0], dims=[2, 2, 2], yaw=0.0)
    b = Box3D(center=[1, 0, 0], dims=[2, 2, 2], yaw=0.0)
    # intersection area 2, union = 4 + 4 - 2 = 6 -> iou = 1/3
    assert bev_iou(a, b) == pytest.approx(1.0 / 3.0)


def test_iou_3d_zero_when_no_height_overlap():
    a = Box3D(center=[0, 0, 0], dims=[2, 2, 2], yaw=0.0)
    b = Box3D(center=[0, 0, 10], dims=[2, 2, 2], yaw=0.0)  # same footprint, far above
    assert iou_3d(a, b) == 0.0


def test_iou_3d_matches_bev_times_height_for_axis_aligned_full_height_overlap():
    a = Box3D(center=[0, 0, 0], dims=[2, 2, 2], yaw=0.0)
    b = Box3D(center=[1, 0, 0], dims=[2, 2, 2], yaw=0.0)  # identical height extent
    assert iou_3d(a, b) == pytest.approx(bev_iou(a, b))


def test_obb_corners_rotation_swaps_footprint_extent():
    # A box twice as long (x) as it is wide (y); rotating 90 degrees should
    # swap which world axis is the long one.
    box = Box3D(center=[0, 0, 0], dims=[4, 2, 2], yaw=np.pi / 2)
    poly = obb_bev_polygon(box)
    x_extent = poly[:, 0].max() - poly[:, 0].min()
    y_extent = poly[:, 1].max() - poly[:, 1].min()
    assert x_extent == pytest.approx(2.0, abs=1e-5)
    assert y_extent == pytest.approx(4.0, abs=1e-5)


def test_obb_corners_count_and_centered():
    box = Box3D(center=[5, -3, 1], dims=[2, 4, 6], yaw=0.0)
    corners = obb_corners(box)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), box.center, atol=1e-4)


def test_bev_iou_random_matches_rasterized_brute_force():
    """Cross-check the analytic polygon-clipping IoU against a dumb pixel
    rasterization, which makes no assumptions about clipping correctness."""
    rng = np.random.default_rng(7)

    def brute_force(a: Box3D, b: Box3D, res=500, extent=6.0) -> float:
        xs = np.linspace(-extent, extent, res)
        ys = np.linspace(-extent, extent, res)
        X, Y = np.meshgrid(xs, ys)

        def inside(poly):
            mask = np.ones(X.shape, dtype=bool)
            for i in range(len(poly)):
                p_a, p_b = poly[i - 1], poly[i]
                edge = p_b - p_a
                cross = edge[0] * (Y - p_a[1]) - edge[1] * (X - p_a[0])
                mask &= cross >= 0
            return mask

        ia = inside(obb_bev_polygon(a))
        ib = inside(obb_bev_polygon(b))
        union = (ia | ib).sum()
        return (ia & ib).sum() / union if union else 0.0

    for _ in range(6):
        a = Box3D(center=[*rng.uniform(-2, 2, 2), 0], dims=[*rng.uniform(1, 3, 2), 2], yaw=rng.uniform(0, np.pi))
        b = Box3D(center=[*rng.uniform(-2, 2, 2), 0], dims=[*rng.uniform(1, 3, 2), 2], yaw=rng.uniform(0, np.pi))
        analytic = bev_iou(a, b)
        brute = brute_force(a, b)
        assert analytic == pytest.approx(brute, abs=0.01)


def test_look_at_eye_maps_to_origin_in_view_space():
    eye = np.array([5.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    view = look_at(eye, target)
    eye_view_space = view @ np.array([*eye, 1.0])
    assert np.allclose(eye_view_space[:3], [0, 0, 0], atol=1e-5)


def test_look_at_rejects_degenerate_eye_target():
    with pytest.raises(ValueError):
        look_at([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def test_perspective_matrix_validates_inputs():
    with pytest.raises(ValueError):
        perspective_matrix(60, -1, 0.1, 100)
    with pytest.raises(ValueError):
        perspective_matrix(200, 1.0, 0.1, 100)
    with pytest.raises(ValueError):
        perspective_matrix(60, 1.0, 10, 1)


def test_orbit_camera_clamps_elevation_and_distance():
    cam = OrbitCamera(distance=10, elevation=0.0, min_distance=5, max_distance=20)
    cam.orbit(0.0, 10.0)  # way past the max elevation
    assert cam.elevation <= cam.max_elevation

    cam.dolly(0.01)  # zoom in far past min_distance
    assert cam.distance >= cam.min_distance

    cam.dolly(1000.0)  # zoom out far past max_distance
    assert cam.distance <= cam.max_distance


def test_orbit_camera_distance_from_target_is_preserved():
    cam = OrbitCamera(target=[1.0, 2.0, 0.0], distance=15.0)
    cam.orbit(1.2, 0.3)
    assert np.linalg.norm(cam.position - cam.target) == pytest.approx(15.0, abs=1e-4)
