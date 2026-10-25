"""Shared fixtures. The GL-backed tests need a headless-capable context
(EGL + a software rasterizer like Mesa llvmpipe works fine, no real GPU
or display required) -- if that's not available in a given environment
we skip rather than fail, since that's an environment limitation and not
a bug in PointScope itself.
"""
import pytest


def _gl_context_available() -> bool:
    try:
        from pointscope.offscreen import create_standalone_context

        ctx = create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


requires_gl = pytest.mark.skipif(not _gl_context_available(), reason="No headless OpenGL context available in this environment")
