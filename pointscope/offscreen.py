"""Headless rendering path: draws a :class:`~pointscope.data.Frame` to an
off-screen framebuffer with no window, display, or GPU driver beyond a
software rasterizer (Mesa llvmpipe via EGL works fine). This is what the
automated tests and CI use to actually exercise the real shader/render
code -- not a mock -- and it's also how ``scripts/render_preview.py``
produces the screenshot used in the README.
"""
from __future__ import annotations

from typing import Optional, Tuple

import moderngl
import numpy as np
from PIL import Image

from .data import Frame
from .geometry import OrbitCamera, perspective_matrix
from .renderer import Renderer


def create_standalone_context() -> moderngl.Context:
    """Create a headless GL context, preferring EGL (works without an X
    server / real display, which is what CI runners and this dev sandbox
    both need)."""
    last_error: Optional[Exception] = None
    for backend in ("egl", None):  # None = let moderngl pick the platform default
        try:
            return moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
    raise RuntimeError(f"Could not create a headless GL context: {last_error}")


def render_frame_to_array(
    frame: Frame,
    *,
    width: int = 960,
    height: int = 540,
    camera: Optional[OrbitCamera] = None,
    point_size: float = 3.0,
    color_mode: int = 0,
    show_gt: bool = True,
    show_pred: bool = True,
    ctx: Optional[moderngl.Context] = None,
) -> Tuple[np.ndarray, "Renderer"]:
    """Render one frame off-screen and return it as an ``(H, W, 3)`` uint8 array.

    Returns the renderer too so callers (tests, the preview script) can
    inspect ``renderer.stats`` without re-deriving it.
    """
    owns_ctx = ctx is None
    ctx = ctx or create_standalone_context()
    camera = camera or OrbitCamera(distance=30.0, elevation=0.5)

    color_rbo = ctx.renderbuffer((width, height))
    depth_rbo = ctx.depth_renderbuffer((width, height))
    fbo = ctx.framebuffer(color_attachments=[color_rbo], depth_attachment=depth_rbo)
    fbo.use()
    ctx.viewport = (0, 0, width, height)
    fbo.clear(0.04, 0.04, 0.06, 1.0)

    renderer = Renderer(ctx)
    renderer.set_point_cloud(frame.points)
    renderer.set_boxes(frame.gt_boxes, frame.pred_boxes)

    view = camera.view_matrix()
    proj = perspective_matrix(60.0, width / height, 0.1, 300.0)
    z = frame.points[:, 2] if len(frame.points) else np.array([0.0, 1.0])
    renderer.render(
        view,
        proj,
        point_size=point_size,
        color_mode=color_mode,
        height_range=(float(z.min()), float(z.max())),
        show_gt=show_gt,
        show_pred=show_pred,
    )

    raw = fbo.read(components=3, alignment=1)
    image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    image = np.flipud(image)  # OpenGL's row 0 is the bottom of the image

    if owns_ctx:
        renderer.release()
        fbo.release()
        color_rbo.release()
        depth_rbo.release()

    return image, renderer


def save_frame_png(frame: Frame, path: str, **kwargs) -> None:
    image, _ = render_frame_to_array(frame, **kwargs)
    Image.fromarray(image).save(path)
