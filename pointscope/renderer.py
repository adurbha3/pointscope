"""GPU resource management and draw calls, shared by the interactive app
and the headless offscreen render path.

Keeping this separate from both ``app.py`` (which owns the window/input
loop) and ``offscreen.py`` (which owns the headless context) means the
exact same rendering code is what gets exercised by the automated tests
-- there is only one code path that actually talks to the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import moderngl
import numpy as np

from .geometry import BOX_EDGES, Box3D, obb_corners

SHADER_DIR = Path(__file__).parent / "shaders"


def _read_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


@dataclass
class RenderStats:
    """Lightweight per-frame counters surfaced in the app's HUD / tests."""

    draw_calls: int = 0
    point_count: int = 0
    gt_box_count: int = 0
    pred_box_count: int = 0


def _mat4_to_gl_bytes(m: np.ndarray) -> bytes:
    """Convert a row-major (v' = M @ v) 4x4 matrix to the column-major byte
    layout GLSL's ``mat4`` expects."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


class Renderer:
    """Owns GPU programs + buffers and draws one frame's points + boxes."""

    GT_COLOR: Tuple[float, float, float] = (0.25, 0.95, 0.35)
    PRED_COLOR: Tuple[float, float, float] = (0.95, 0.40, 0.20)

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.point_program = ctx.program(
            vertex_shader=_read_shader("point.vert"),
            fragment_shader=_read_shader("point.frag"),
        )
        self.line_program = ctx.program(
            vertex_shader=_read_shader("line.vert"),
            fragment_shader=_read_shader("line.frag"),
        )

        self._point_vbo: Optional[moderngl.Buffer] = None
        self._point_vao: Optional[moderngl.VertexArray] = None
        self._gt_vbo: Optional[moderngl.Buffer] = None
        self._gt_vao: Optional[moderngl.VertexArray] = None
        self._pred_vbo: Optional[moderngl.Buffer] = None
        self._pred_vao: Optional[moderngl.VertexArray] = None
        self._gt_vertex_count = 0
        self._pred_vertex_count = 0

        self.stats = RenderStats()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        for buf in (self._point_vbo, self._gt_vbo, self._pred_vbo):
            if buf is not None:
                buf.release()
        for vao in (self._point_vao, self._gt_vao, self._pred_vao):
            if vao is not None:
                vao.release()

    # -- uploading per-frame geometry ------------------------------------

    def set_point_cloud(self, points: np.ndarray) -> None:
        """``points``: ``(N, 4)`` float32 array of ``x, y, z, intensity``."""
        points = np.ascontiguousarray(points, dtype=np.float32)
        if self._point_vbo is not None:
            self._point_vbo.release()
        if self._point_vao is not None:
            self._point_vao.release()
        self._point_vbo = self.ctx.buffer(reserve=16) if len(points) == 0 else self.ctx.buffer(points.tobytes())
        self._point_vao = self.ctx.vertex_array(
            self.point_program,
            [(self._point_vbo, "3f 1f", "in_position", "in_intensity")],
        )
        self.stats.point_count = len(points)

    @staticmethod
    def _box_line_vertices(boxes: Sequence[Box3D]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 3), dtype=np.float32)
        segments = []
        for box in boxes:
            corners = obb_corners(box)
            for a, b in BOX_EDGES:
                segments.append(corners[a])
                segments.append(corners[b])
        return np.asarray(segments, dtype=np.float32)

    def set_boxes(self, gt_boxes: Sequence[Box3D], pred_boxes: Sequence[Box3D]) -> None:
        gt_verts = self._box_line_vertices(gt_boxes)
        pred_verts = self._box_line_vertices(pred_boxes)

        if self._gt_vbo is not None:
            self._gt_vbo.release()
            self._gt_vao.release()
        if self._pred_vbo is not None:
            self._pred_vbo.release()
            self._pred_vao.release()

        self._gt_vbo = self.ctx.buffer(reserve=4) if len(gt_verts) == 0 else self.ctx.buffer(gt_verts.tobytes())
        self._gt_vao = self.ctx.vertex_array(self.line_program, [(self._gt_vbo, "3f", "in_position")])
        self._pred_vbo = self.ctx.buffer(reserve=4) if len(pred_verts) == 0 else self.ctx.buffer(pred_verts.tobytes())
        self._pred_vao = self.ctx.vertex_array(self.line_program, [(self._pred_vbo, "3f", "in_position")])

        self._gt_vertex_count = len(gt_verts)
        self._pred_vertex_count = len(pred_verts)
        self.stats.gt_box_count = len(gt_boxes)
        self.stats.pred_box_count = len(pred_boxes)

    # -- drawing -----------------------------------------------------------

    def render(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        *,
        point_size: float = 3.0,
        color_mode: int = 0,
        height_range: Tuple[float, float] = (-1.0, 3.0),
        show_gt: bool = True,
        show_pred: bool = True,
    ) -> None:
        mvp = proj @ view
        mvp_bytes = _mat4_to_gl_bytes(mvp)
        draw_calls = 0

        if self._point_vao is not None and self.stats.point_count > 0:
            self.point_program["mvp"].write(mvp_bytes)
            self.point_program["view"].write(_mat4_to_gl_bytes(view))
            self.point_program["color_mode"].value = int(color_mode)
            self.point_program["height_min"].value = float(height_range[0])
            self.point_program["height_max"].value = float(height_range[1])
            self.point_program["point_size_base"].value = float(point_size)
            self._point_vao.render(moderngl.POINTS)
            draw_calls += 1

        if show_gt and self._gt_vertex_count > 0:
            self.line_program["mvp"].write(mvp_bytes)
            self.line_program["line_color"].value = self.GT_COLOR
            self._gt_vao.render(moderngl.LINES, vertices=self._gt_vertex_count)
            draw_calls += 1

        if show_pred and self._pred_vertex_count > 0:
            self.line_program["mvp"].write(mvp_bytes)
            self.line_program["line_color"].value = self.PRED_COLOR
            self._pred_vao.render(moderngl.LINES, vertices=self._pred_vertex_count)
            draw_calls += 1

        self.stats.draw_calls = draw_calls
