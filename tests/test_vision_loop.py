import base64
import types

from app.providers import _capture_screenshot_b64, get_scale_factor


def test_capture_screenshot_b64(monkeypatch):
    class Shot:
        size = (100, 100)
        rgb = bytes([255, 0, 0] * 100 * 100)

    class MSSCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def grab(self, monitor):
            assert monitor["width"] == 100
            return Shot()

    monkeypatch.setitem(__import__("sys").modules, "mss", types.SimpleNamespace(mss=lambda: MSSCtx()))

    b64 = _capture_screenshot_b64(100, 100)
    data = base64.b64decode(b64)
    assert data.startswith(b"\x89PNG")
    assert get_scale_factor(1920, 1080) < 1.0
    assert get_scale_factor(800, 600) == 1.0
