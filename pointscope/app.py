"""Interactive PointScope viewer.

A moderngl_window app around the shared :class:`~pointscope.renderer.Renderer`:
orbit camera, frame-by-frame playback/scrubbing, ground-truth vs. prediction
overlay toggles, and a live title-bar HUD (frame index, point count,
match stats, FPS) -- the same kind of debug tooling a graphics/tools
engineer builds so other engineers can inspect pipeline output without
reading raw arrays.

Controls
--------
left-drag           orbit camera
right-drag          pan camera target
scroll              zoom
space               play / pause
left / right arrow  step one frame back / forward (pauses playback)
g                   toggle ground-truth boxes (green)
p                   toggle predicted boxes (orange)
c                   cycle point color mode (height <-> intensity)
= / -               point size up / down
s                   save a screenshot to ./exports/
esc                 quit
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import moderngl_window as mglw
import numpy as np
from moderngl_window.context.base.keys import KeyModifiers

from .data import Frame, generate_synthetic_sequence, load_frames_from_dir
from .geometry import OrbitCamera, perspective_matrix
from .metrics import match_frame
from .renderer import Renderer

EXPORT_DIR = Path("exports")


class PointScopeApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "PointScope"
    window_size = (1280, 720)
    resizable = True
    aspect_ratio = None
    vsync = True

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data", type=str, default=None,
            help="Directory of frame_*.npy / frame_*.json to load instead of the synthetic demo sequence",
        )
        parser.add_argument("--frames", type=int, default=90, help="Frame count for the synthetic demo sequence")
        parser.add_argument("--seed", type=int, default=0, help="Random seed for the synthetic demo sequence")
        parser.add_argument("--fps", type=float, default=10.0, help="Playback speed in frames/second")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if getattr(self.argv, "data", None):
            self.frames: List[Frame] = load_frames_from_dir(self.argv.data)
        else:
            self.frames = generate_synthetic_sequence(
                num_frames=getattr(self.argv, "frames", 90),
                seed=getattr(self.argv, "seed", 0),
            )

        self.renderer = Renderer(self.ctx)
        self.camera = OrbitCamera(distance=35.0, elevation=0.5)

        self.frame_index = 0
        self.playing = True
        self.playback_fps = getattr(self.argv, "fps", 10.0)
        self._play_accum = 0.0
        self._render_fps_ema = 0.0

        self.show_gt = True
        self.show_pred = True
        self.color_mode = 0  # 0 = height, 1 = intensity
        self.point_size = 3.0

        self._drag_mode: str | None = None

        EXPORT_DIR.mkdir(exist_ok=True)
        self._load_frame(0)

    # -- frame management ---------------------------------------------------

    def _load_frame(self, index: int) -> None:
        index = int(np.clip(index, 0, len(self.frames) - 1))
        self.frame_index = index
        frame = self.frames[index]
        self.renderer.set_point_cloud(frame.points)
        self.renderer.set_boxes(frame.gt_boxes, frame.pred_boxes)

    def _update_title(self) -> None:
        frame = self.frames[self.frame_index]
        match = match_frame(frame.gt_boxes, frame.pred_boxes)
        state = "playing" if self.playing else "paused"
        color = "height" if self.color_mode == 0 else "intensity"
        self.wnd.title = (
            f"PointScope | frame {self.frame_index + 1}/{len(self.frames)} | {state} "
            f"| {self.renderer.stats.point_count} pts, size {self.point_size:.1f} ({color}) "
            f"| GT{'on' if self.show_gt else 'off'}/PRED{'on' if self.show_pred else 'off'} "
            f"| TP {match.true_positives} FP {match.false_positives} FN {match.false_negatives} mIoU {match.mean_iou:.2f} "
            f"| {self._render_fps_ema:.0f} FPS"
        )

    # -- render loop ----------------------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self.ctx.clear(0.04, 0.04, 0.06)

        if frame_time > 1e-6:
            instant_fps = 1.0 / frame_time
            self._render_fps_ema = (
                instant_fps if self._render_fps_ema == 0.0 else 0.9 * self._render_fps_ema + 0.1 * instant_fps
            )

        if self.playing and len(self.frames) > 1:
            self._play_accum += frame_time
            step_dt = 1.0 / max(self.playback_fps, 0.01)
            if self._play_accum >= step_dt:
                self._play_accum = 0.0
                self._load_frame((self.frame_index + 1) % len(self.frames))

        aspect = self.wnd.aspect_ratio or (self.wnd.width / max(self.wnd.height, 1))
        proj = perspective_matrix(60.0, aspect, 0.1, 300.0)
        view = self.camera.view_matrix()

        frame = self.frames[self.frame_index]
        z = frame.points[:, 2] if len(frame.points) else np.array([0.0, 1.0])
        self.renderer.render(
            view,
            proj,
            point_size=self.point_size,
            color_mode=self.color_mode,
            height_range=(float(z.min()), float(z.max())),
            show_gt=self.show_gt,
            show_pred=self.show_pred,
        )
        self._update_title()

    # -- input ------------------------------------------------------------------

    def on_key_event(self, key, action, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.playing = not self.playing
        elif key == keys.LEFT:
            self.playing = False
            self._load_frame(self.frame_index - 1)
        elif key == keys.RIGHT:
            self.playing = False
            self._load_frame(self.frame_index + 1)
        elif key == keys.G:
            self.show_gt = not self.show_gt
        elif key == keys.P:
            self.show_pred = not self.show_pred
        elif key == keys.C:
            self.color_mode = 1 - self.color_mode
        elif key == keys.EQUAL:
            self.point_size = min(self.point_size + 0.5, 12.0)
        elif key == keys.MINUS:
            self.point_size = max(self.point_size - 0.5, 0.5)
        elif key == keys.S:
            self._save_screenshot()
        elif key == keys.ESCAPE:
            self.wnd.close()

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        if button == self.wnd.mouse.left:
            self._drag_mode = "orbit"
        elif button == self.wnd.mouse.right:
            self._drag_mode = "pan"

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        self._drag_mode = None

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        if self._drag_mode == "orbit":
            self.camera.orbit(-dx * 0.01, dy * 0.01)
        elif self._drag_mode == "pan":
            self.camera.pan(-dx * 0.03, dy * 0.03)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.camera.dolly(1.0 - y_offset * 0.1)

    # -- misc -----------------------------------------------------------------

    def _save_screenshot(self) -> None:
        from PIL import Image

        fbo = self.wnd.fbo
        data = fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape((fbo.size[1], fbo.size[0], 3))
        img = np.flipud(img)
        out_path = EXPORT_DIR / f"pointscope_frame{self.frame_index:04d}.png"
        Image.fromarray(img).save(out_path)
        print(f"[PointScope] saved screenshot -> {out_path}")


def run() -> None:
    mglw.run_window_config(PointScopeApp)


if __name__ == "__main__":
    run()
