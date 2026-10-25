"""Synthetic driving-scene sequences + a loader for real LiDAR-style frames.

PointScope ships without any proprietary sensor data, so by default it
visualizes a seeded, reproducible synthetic sequence: an ego vehicle
driving down a corridor past several actors (other vehicles /
pedestrians) that a detector has produced (noisy, sometimes-missing,
sometimes-spurious) predictions for. This exercises the exact same data
path -- ``Frame`` objects full of points + ground-truth boxes +
predicted boxes -- that ``load_frames_from_dir`` builds from real files,
so swapping in an actual LiDAR capture later is a one-line change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .geometry import Box3D


@dataclass
class Frame:
    """One time step of a sensor sequence."""

    points: np.ndarray  # (N, 4): x, y, z, intensity in [0, 1]
    gt_boxes: List[Box3D] = field(default_factory=list)
    pred_boxes: List[Box3D] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError(f"points must be (N, 4), got {self.points.shape}")


@dataclass
class _Actor:
    kind: str
    start_pos: np.ndarray
    velocity: np.ndarray
    dims: np.ndarray
    yaw: float
    point_count: int


def _ground_plane_points(rng: np.random.Generator, radius: float = 40.0, spacing: float = 0.6) -> np.ndarray:
    xs = np.arange(-radius, radius, spacing)
    ys = np.arange(-radius, radius, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    # keep roughly a disc, like a real LiDAR's radial coverage, and only
    # keep a random subset so the ground doesn't look like a perfect grid
    dist = np.sqrt(grid_x ** 2 + grid_y ** 2)
    keep = (dist <= radius) & (rng.random(len(grid_x)) < 0.35)
    grid_x, grid_y = grid_x[keep], grid_y[keep]

    z = rng.normal(0.0, 0.02, size=grid_x.shape[0]).astype(np.float32)
    intensity = np.clip(0.15 + rng.normal(0.0, 0.03, size=grid_x.shape[0]), 0.0, 1.0)
    pts = np.stack([grid_x, grid_y, z, intensity], axis=1).astype(np.float32)
    return pts


def _box_point_cloud(rng: np.random.Generator, box: Box3D, count: int) -> np.ndarray:
    """Sample points roughly on the visible surface of a box (not solid-filled),
    which is what a real LiDAR return from an object actually looks like."""
    local = rng.uniform(-0.5, 0.5, size=(count, 3))
    # push most samples toward the surface (|coord| close to 0.5 on one axis)
    axis = rng.integers(0, 3, size=count)
    sign = rng.choice([-1.0, 1.0], size=count)
    local[np.arange(count), axis] = sign * 0.5 * rng.uniform(0.85, 1.0, size=count)
    scaled = local * box.dims
    from .geometry import rotation_z

    rotated = scaled @ rotation_z(box.yaw).T
    world = rotated + box.center
    intensity = np.clip(rng.normal(0.4, 0.1, size=count), 0.0, 1.0)
    return np.concatenate([world, intensity[:, None]], axis=1).astype(np.float32)


def _make_actors(rng: np.random.Generator) -> List[_Actor]:
    actors = []
    # A handful of oncoming / crossing vehicles and a couple of pedestrians,
    # loosely modeled on a suburban-intersection scene.
    specs = [
        dict(kind="vehicle", start=(18.0, -4.0, 0.75), vel=(-3.2, 0.0, 0.0), dims=(4.6, 1.9, 1.5), yaw=np.pi, pts=220),
        dict(kind="vehicle", start=(-10.0, 5.5, 0.75), vel=(2.4, -0.1, 0.0), dims=(4.2, 1.8, 1.5), yaw=0.0, pts=200),
        dict(kind="pedestrian", start=(6.0, 2.0, 0.9), vel=(0.0, -0.9, 0.0), dims=(0.6, 0.6, 1.7), yaw=0.0, pts=60),
        dict(kind="pedestrian", start=(-4.0, -6.0, 0.9), vel=(0.3, 0.8, 0.0), dims=(0.6, 0.6, 1.6), yaw=0.3, pts=55),
        dict(kind="vehicle", start=(25.0, 3.0, 0.75), vel=(-2.0, -0.4, 0.0), dims=(5.0, 2.0, 1.8), yaw=2.9, pts=240),
    ]
    for s in specs:
        jitter = rng.normal(0, 0.15, size=3)
        actors.append(
            _Actor(
                kind=s["kind"],
                start_pos=np.array(s["start"], dtype=np.float32) + jitter.astype(np.float32),
                velocity=np.array(s["vel"], dtype=np.float32),
                dims=np.array(s["dims"], dtype=np.float32),
                yaw=float(s["yaw"]),
                point_count=int(s["pts"]),
            )
        )
    return actors


def generate_synthetic_sequence(
    num_frames: int = 60,
    dt: float = 0.1,
    seed: int = 0,
    detector_miss_rate: float = 0.08,
    detector_false_positive_rate: float = 0.05,
    detector_noise_std: float = 0.18,
) -> List[Frame]:
    """Build a reproducible synthetic LiDAR-like sequence with GT + "detector" boxes.

    The "predictions" are the ground-truth boxes run through a simple
    noise model (position/size/yaw jitter, occasional misses, occasional
    spurious boxes) so the viewer and metrics.py have something realistic
    to compare against -- the same role real model output would play.
    """
    rng = np.random.default_rng(seed)
    actors = _make_actors(rng)
    frames: List[Frame] = []

    for i in range(num_frames):
        t = i * dt
        ground = _ground_plane_points(rng)
        point_chunks = [ground]
        gt_boxes: List[Box3D] = []

        for actor in actors:
            pos = actor.start_pos + actor.velocity * t
            box = Box3D(center=pos, dims=actor.dims, yaw=actor.yaw, label=actor.kind)
            gt_boxes.append(box)
            point_chunks.append(_box_point_cloud(rng, box, actor.point_count))

        # light sensor noise scattered through the scene
        noise_n = 80
        noise_pts = rng.uniform([-40, -40, -0.3], [40, 40, 2.5], size=(noise_n, 3))
        noise_intensity = rng.uniform(0.0, 0.2, size=noise_n)
        point_chunks.append(np.concatenate([noise_pts, noise_intensity[:, None]], axis=1).astype(np.float32))

        points = np.concatenate(point_chunks, axis=0).astype(np.float32)

        pred_boxes: List[Box3D] = []
        for box in gt_boxes:
            if rng.random() < detector_miss_rate:
                continue  # simulated false negative
            noisy_center = box.center + rng.normal(0, detector_noise_std, size=3).astype(np.float32)
            noisy_dims = box.dims * (1.0 + rng.normal(0, 0.06, size=3)).astype(np.float32)
            noisy_yaw = box.yaw + rng.normal(0, 0.05)
            score = float(np.clip(rng.normal(0.82, 0.1), 0.3, 0.99))
            pred_boxes.append(
                Box3D(center=noisy_center, dims=noisy_dims, yaw=noisy_yaw, label=box.label, score=score)
            )
        if rng.random() < detector_false_positive_rate:
            fp_center = rng.uniform([-20, -20, 0.5], [20, 20, 1.2])
            pred_boxes.append(
                Box3D(
                    center=fp_center.astype(np.float32),
                    dims=np.array([3.8, 1.7, 1.5], dtype=np.float32),
                    yaw=float(rng.uniform(0, 2 * np.pi)),
                    label="vehicle",
                    score=float(rng.uniform(0.3, 0.6)),
                )
            )

        frames.append(Frame(points=points, gt_boxes=gt_boxes, pred_boxes=pred_boxes, timestamp=t))

    return frames


# --------------------------------------------------------------------------
# Loading real sequences
# --------------------------------------------------------------------------


def _box_from_dict(d: dict) -> Box3D:
    return Box3D(
        center=np.array(d["center"], dtype=np.float32),
        dims=np.array(d["dims"], dtype=np.float32),
        yaw=float(d["yaw"]),
        label=d.get("label", "object"),
        score=float(d.get("score", 1.0)),
    )


def _box_to_dict(box: Box3D) -> dict:
    return {
        "center": box.center.tolist(),
        "dims": box.dims.tolist(),
        "yaw": box.yaw,
        "label": box.label,
        "score": box.score,
    }


def load_frames_from_dir(path: Path | str) -> List[Frame]:
    """Load a sequence from ``<path>/frame_*.npy`` + matching ``.json`` sidecars.

    Each ``frame_XXXXXX.npy`` holds an ``(N, 4)`` float array of
    ``x, y, z, intensity``. The optional sidecar
    ``frame_XXXXXX.json`` has the shape::

        {"timestamp": 0.1, "gt_boxes": [...], "pred_boxes": [...]}

    where each box is ``{"center": [x,y,z], "dims": [l,w,h], "yaw": r,
    "label": str, "score": float}``. This mirrors the convention used by
    common AV datasets (KITTI/nuScenes-style per-frame exports), so
    pointing PointScope at a real capture is just a conversion script
    away.
    """
    path = Path(path)
    npy_files = sorted(path.glob("frame_*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No frame_*.npy files found in {path}")

    frames: List[Frame] = []
    for npy_path in npy_files:
        points = np.load(npy_path)
        json_path = npy_path.with_suffix(".json")
        gt_boxes, pred_boxes, timestamp = [], [], 0.0
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            gt_boxes = [_box_from_dict(b) for b in meta.get("gt_boxes", [])]
            pred_boxes = [_box_from_dict(b) for b in meta.get("pred_boxes", [])]
            timestamp = float(meta.get("timestamp", 0.0))
        frames.append(Frame(points=points, gt_boxes=gt_boxes, pred_boxes=pred_boxes, timestamp=timestamp))
    return frames


def save_frames_to_dir(frames: List[Frame], path: Path | str) -> None:
    """Inverse of :func:`load_frames_from_dir` -- used by tests and to cache
    a synthetic sequence to disk so the interactive app can load instantly."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        stem = f"frame_{i:06d}"
        np.save(path / f"{stem}.npy", frame.points)
        meta = {
            "timestamp": frame.timestamp,
            "gt_boxes": [_box_to_dict(b) for b in frame.gt_boxes],
            "pred_boxes": [_box_to_dict(b) for b in frame.pred_boxes],
        }
        (path / f"{stem}.json").write_text(json.dumps(meta, indent=2))
