"""Observation validation + JPEG encode helpers shared by every ficelle client
transport (websocket PolicyClient, iroh IrohPolicyClient).

Factored out of PolicyClient (the original single-transport home for this
logic) so both clients validate/encode identically instead of duplicating the
logic — messages here are byte-identical to what PolicyClient produced
before this module existed.
"""

import io

import numpy as np


def check_jpeg_quality(quality) -> None:
    if not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError(f"jpeg_quality must be an integer between 1 and 100, got {quality!r}")


def pick_jpeg_encoder(quality: int):
    """cv2 if available, else PIL — chosen once, fail fast with install hint."""
    try:
        import cv2

        def encode(frame):  # frame: uint8 HWC RGB
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()

        return encode
    except ImportError:
        pass
    try:
        from PIL import Image

        def encode(frame):
            out = io.BytesIO()
            Image.fromarray(frame).save(out, format="JPEG", quality=quality)
            return out.getvalue()

        return encode
    except ImportError:
        raise ValueError(
            "jpeg_quality requires an image codec: pip install opencv-python-headless (or pillow)"
        ) from None


def validate_obs(metadata: dict, obs: dict) -> None:
    """Client-side mirror of the server's own validation, run against the
    metadata the client fetched — fails fast with a message naming the
    offending key, before anything is sent over the wire."""
    spec = metadata["observations"]
    n_obs = metadata.get("n_obs_steps", 1)
    if "task" in obs and not isinstance(obs["task"], str):
        raise ValueError(f"'task' must be a string, got {type(obs['task']).__name__}")
    for key in obs:
        if key != "task" and key not in spec:
            raise ValueError(
                f"unknown observation key '{key}' — expected {sorted(spec)} (+ optional 'task')"
            )
    for key, s in spec.items():
        if key not in obs:
            raise ValueError(
                f"missing observation key '{key}' — expected {sorted(spec)} (+ optional 'task')"
            )
        arr = obs[key]
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"'{key}' must be a numpy array, got {type(arr).__name__}")
        if list(arr.shape) != list(s["wire_shape"]):
            hint = f" ({n_obs} frames stacked oldest-first)" if n_obs > 1 else ""
            raise ValueError(
                f"'{key}' has shape {list(arr.shape)}, expected {list(s['wire_shape'])}{hint}"
            )
        if str(arr.dtype) != s["dtype"]:
            raise ValueError(f"'{key}' has dtype {arr.dtype}, expected {s['dtype']}")


def encode_jpeg(metadata: dict, obs: dict, encode_frame) -> dict:
    """Encode validated raw image arrays to per-frame JPEG bytes (oldest
    first). Callers keep passing raw arrays — this is transport-only."""
    encoded = dict(obs)
    for key, spec in metadata["observations"].items():
        if spec["kind"] != "image":
            continue
        arr = obs[key]
        frames = arr if arr.ndim == 4 else arr[None]
        encoded[key] = [encode_frame(np.ascontiguousarray(f)) for f in frames]
    return encoded
