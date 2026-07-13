"""Unit tests for the resize helpers in ficelle_client._obsval — shared by
both PolicyClient (websocket) and IrohPolicyClient so the resize logic isn't
duplicated (see client/ficelle_client/_obsval.py's module docstring)."""

import cv2
import numpy as np
from PIL import Image

from ficelle_client import _obsval


def _gradient_frame(h: int, w: int) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(0, 255, w, dtype=np.uint8),
                         np.linspace(0, 255, h, dtype=np.uint8))
    return np.stack([xx, yy, np.full_like(xx, 128)], axis=-1)


def test_cv2_resizer_never_uses_inter_linear(monkeypatch):
    """Never cv2.INTER_LINEAR (measured to alias and deviate most from the
    server's own resize) — the cv2 fallback must request INTER_AREA."""
    seen = {}
    real_resize = cv2.resize

    def spy_resize(src, dsize, interpolation=None):
        seen["interpolation"] = interpolation
        return real_resize(src, dsize, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", spy_resize)
    resizer = _obsval._cv2_resizer()
    out = resizer(_gradient_frame(96, 96), [48, 48])
    assert seen["interpolation"] == cv2.INTER_AREA
    assert seen["interpolation"] != cv2.INTER_LINEAR
    assert out.shape == (48, 48, 3)


def test_pick_resizer_prefers_pil():
    resizer = _obsval.pick_resizer()
    frame = _gradient_frame(96, 96)
    out = resizer(frame, [48, 48])
    assert out.shape == (48, 48, 3)
    ref = np.array(Image.fromarray(frame).resize((48, 48), Image.BILINEAR))
    assert np.array_equal(out, ref)


def test_pick_resizer_falls_back_to_cv2_when_pil_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "PIL", None)
    resizer = _obsval.pick_resizer()
    out = resizer(_gradient_frame(96, 96), [48, 48])
    assert out.shape == (48, 48, 3)


def test_resize_frames_single_frame_hwc():
    spec = {
        "observation.images.cam0": {"kind": "image", "dtype": "uint8",
                                    "frame_shape": [96, 96, 3], "resize_to": [48, 48]},
        "observation.state": {"kind": "state", "dtype": "float32", "frame_shape": [2]},
    }
    obs = {
        "observation.images.cam0": _gradient_frame(96, 96),
        "observation.state": np.zeros(2, dtype=np.float32),
    }
    resizer = _obsval.pick_resizer()
    out = _obsval.resize_frames(obs, spec, resizer)
    assert out["observation.images.cam0"].shape == (48, 48, 3)
    # untouched keys (no resize_to) pass through unchanged
    assert np.array_equal(out["observation.state"], obs["observation.state"])


def test_resize_frames_stacked_frames():
    spec = {
        "observation.images.cam0": {"kind": "image", "dtype": "uint8",
                                    "frame_shape": [96, 96, 3], "resize_to": [48, 48]},
    }
    frame0 = _gradient_frame(96, 96)
    frame1 = np.roll(frame0, 10, axis=0)
    obs = {"observation.images.cam0": np.stack([frame0, frame1])}
    resizer = _obsval.pick_resizer()
    out = _obsval.resize_frames(obs, spec, resizer)
    assert out["observation.images.cam0"].shape == (2, 48, 48, 3)


def test_resize_frames_skips_keys_without_resize_to():
    spec = {
        "observation.images.cam0": {"kind": "image", "dtype": "uint8",
                                    "frame_shape": [96, 96, 3]},  # no resize_to
    }
    obs = {"observation.images.cam0": _gradient_frame(96, 96)}
    resizer = _obsval.pick_resizer()
    out = _obsval.resize_frames(obs, spec, resizer)
    assert out["observation.images.cam0"].shape == (96, 96, 3)
    assert np.array_equal(out["observation.images.cam0"], obs["observation.images.cam0"])


def test_resolve_resize_true_when_any_image_key_advertises_resize_to():
    metadata = {"observations": {
        "observation.images.cam0": {"kind": "image", "resize_to": [48, 48]},
        "observation.state": {"kind": "state"},
    }}
    assert _obsval.resolve_resize(metadata) is True


def test_resolve_resize_false_when_no_key_advertises_resize_to():
    metadata = {"observations": {
        "observation.images.cam0": {"kind": "image"},
        "observation.state": {"kind": "state"},
    }}
    assert _obsval.resolve_resize(metadata) is False
