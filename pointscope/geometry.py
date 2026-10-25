"""Camera math and 3D oriented-bounding-box geometry for PointScope.

Everything here is pure NumPy so it can be unit tested without a GPU or
display -- the same functions are reused by both the interactive
``app.py`` (real-time orbit camera) and the headless ``offscreen.py``
render path used in CI.

Bounding boxes follow the common autonomous-driving convention (as in
KITTI / nuScenes): a box is a center ``(x, y, z)``, dimensions
``(length, width, height)`` aligned with the box's own heading, and a
single yaw angle (rotation about the vertical/z axis). IoU between two
oriented boxes is computed the same way KITTI's benchmark does it: as a
2D polygon intersection in bird's-eye-view (BEV), scaled by the overlap
of the two boxes' vertical extents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Box representation
# --------------------------------------------------------------------------


@dataclass
class Box3D:
    """An oriented 3D bounding box in ego-centric coordinates."""

    center: np.ndarray  # (3,) x, y, z
    dims: np.ndarray  # (3,) length (x-extent before rotation), width (y), height (z)
    yaw: float  # radians, rotation about +z
    label: str = "object"
    score: float = 1.0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.dims = np.asarray(self.dims, dtype=np.float32).reshape(3)
        self.yaw = float(self.yaw)


# --------------------------------------------------------------------------
# Camera matrices
# --------------------------------------------------------------------------


def perspective_matrix(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective projection matrix (column-vector convention).

    Returned as a ``(4, 4)`` float32 array suitable for uploading to a
    GLSL ``mat4`` uniform (moderngl accepts it via ``.tobytes()`` /
    ``write()`` in column-major order, which is how NumPy already lays
    out a matrix built this way once transposed -- see ``renderer.py``).
    """
    if aspect <= 0:
        raise ValueError("aspect must be > 0")
    if not (0 < fovy_deg < 180):
        raise ValueError("fovy_deg must be in (0, 180)")
    if not (0 < near < far):
        raise ValueError("require 0 < near < far")

    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Right-handed view matrix looking from ``eye`` toward ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target must differ")
    forward = forward / norm

    side = np.cross(forward, up)
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-9:
        # forward is parallel to up; pick an arbitrary perpendicular up
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        side = np.cross(forward, up)
        side_norm = np.linalg.norm(side)
    side = side / side_norm

    true_up = np.cross(side, forward)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -np.dot(side, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(forward, eye)
    return m


@dataclass
class OrbitCamera:
    """A simple orbit (arcball-ish) camera controlled by azimuth/elevation/distance."""

    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    azimuth: float = 0.0  # radians, around +z
    elevation: float = 0.35  # radians above the ground plane
    distance: float = 25.0
    min_distance: float = 2.0
    max_distance: float = 200.0
    min_elevation: float = -1.4
    max_elevation: float = 1.4

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float32).reshape(3)

    @property
    def position(self) -> np.ndarray:
        el = self.elevation
        az = self.azimuth
        x = self.distance * np.cos(el) * np.cos(az)
        y = self.distance * np.cos(el) * np.sin(az)
        z = self.distance * np.sin(el)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, up=(0.0, 0.0, 1.0))

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth = (self.azimuth + d_azimuth) % (2 * np.pi)
        self.elevation = float(np.clip(self.elevation + d_elevation, self.min_elevation, self.max_elevation))

    def dolly(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, self.min_distance, self.max_distance))

    def pan(self, dx: float, dy: float) -> None:
        """Pan the target in the camera's local screen-right / screen-up plane."""
        az = self.azimuth
        right = np.array([-np.sin(az), np.cos(az), 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.target = self.target + right * dx + up * dy


# --------------------------------------------------------------------------
# Oriented bounding boxes
# --------------------------------------------------------------------------


def rotation_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# Fixed local-space corner offsets (unit box, +-0.5 on each axis), used for
# both the 3D wireframe corners and the BEV polygon. Order matters: corners
# 0-3 are the bottom face in a consistent winding order, 4-7 mirror them on top.
_UNIT_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)

# The 12 edges connecting the 8 corners above, for wireframe rendering.
BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def obb_corners(box: Box3D) -> np.ndarray:
    """Return the ``(8, 3)`` world-space corners of an oriented box."""
    scaled = _UNIT_CORNERS * box.dims.astype(np.float64)
    rotated = scaled @ rotation_z(box.yaw).T
    return (rotated + box.center.astype(np.float64)).astype(np.float32)


def obb_bev_polygon(box: Box3D) -> np.ndarray:
    """Return the ``(4, 2)`` bird's-eye-view (x, y) footprint, CCW ordered."""
    corners = obb_corners(box)
    return corners[:4, :2]  # bottom face corners are already CCW by construction


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (the scalar z-component of the 3D cross product).

    NumPy 2.x dropped support for ``np.cross`` on 2-vectors, so this is
    spelled out by hand -- it's also one line either way.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def _clip_polygon(subject: np.ndarray, clip_edge_a: np.ndarray, clip_edge_b: np.ndarray) -> np.ndarray:
    """One Sutherland-Hodgman clip step against the infinite line (a -> b)."""
    if len(subject) == 0:
        return subject

    def is_inside(p: np.ndarray) -> bool:
        edge = clip_edge_b - clip_edge_a
        return _cross2(edge, p - clip_edge_a) >= 0.0

    def intersect(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        edge = clip_edge_b - clip_edge_a
        d = p2 - p1
        denom = _cross2(edge, d)
        if abs(denom) < 1e-12:
            return p2
        t = _cross2(edge, clip_edge_a - p1) / denom
        return p1 + t * d

    output = []
    n = len(subject)
    for i in range(n):
        current = subject[i]
        prev = subject[i - 1]
        current_in = is_inside(current)
        prev_in = is_inside(prev)
        if current_in:
            if not prev_in:
                output.append(intersect(prev, current))
            output.append(current)
        elif prev_in:
            output.append(intersect(prev, current))
    return np.array(output, dtype=np.float64) if output else np.empty((0, 2))


def polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Sutherland-Hodgman intersection area of two convex CCW polygons."""
    clipped = np.asarray(poly_a, dtype=np.float64)
    poly_b = np.asarray(poly_b, dtype=np.float64)
    n = len(poly_b)
    for i in range(n):
        clipped = _clip_polygon(clipped, poly_b[i - 1], poly_b[i])
        if len(clipped) == 0:
            return 0.0
    return _polygon_area(clipped)


def bev_iou(box_a: Box3D, box_b: Box3D) -> float:
    """IoU of the two boxes' footprints in bird's-eye-view."""
    poly_a = obb_bev_polygon(box_a)
    poly_b = obb_bev_polygon(box_b)
    inter = polygon_intersection_area(poly_a, poly_b)
    if inter <= 0.0:
        return 0.0
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _height_overlap(box_a: Box3D, box_b: Box3D) -> float:
    a_min, a_max = box_a.center[2] - box_a.dims[2] / 2, box_a.center[2] + box_a.dims[2] / 2
    b_min, b_max = box_b.center[2] - box_b.dims[2] / 2, box_b.center[2] + box_b.dims[2] / 2
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def iou_3d(box_a: Box3D, box_b: Box3D) -> float:
    """KITTI-style 3D IoU: BEV polygon IoU scaled by vertical overlap.

    This is an approximation (true 3D IoU of two arbitrarily oriented
    boxes requires 3D polyhedron intersection) but it's the same
    approximation used by the KITTI 3D detection benchmark, since in
    practice most LiDAR boxes share a near-vertical up axis.
    """
    poly_a = obb_bev_polygon(box_a)
    poly_b = obb_bev_polygon(box_b)
    inter_area = polygon_intersection_area(poly_a, poly_b)
    if inter_area <= 0.0:
        return 0.0

    h_overlap = _height_overlap(box_a, box_b)
    if h_overlap <= 0.0:
        return 0.0

    inter_vol = inter_area * h_overlap
    vol_a = _polygon_area(poly_a) * box_a.dims[2]
    vol_b = _polygon_area(poly_b) * box_b.dims[2]
    union_vol = vol_a + vol_b - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0
