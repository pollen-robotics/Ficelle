"""Client for a ficelle policy server.

Deliberately dependency-light (websockets + msgpack + numpy) so it can run in
any robot-side eval loop without torch or lerobot. Validates observations
against the server's metadata BEFORE sending, so mistakes fail fast with a
message naming the offending key.
"""

import logging
import time

import numpy as np
import websockets.exceptions
import websockets.sync.client

from ficelle_client import msgpack_numpy

logger = logging.getLogger("ficelle.client")


class PolicyServerError(RuntimeError):
    """The server replied with an error (text frame)."""


class PolicyClient:
    """Connects to a ficelle server, reads its metadata, then serves
    .infer(obs) -> {"actions": (n_action_steps, action_dim) float32, "server_timing": {...}}.

    Blocks retrying until the server is reachable (like openpi's client) and
    transparently reconnects once if the server restarted between requests.
    """

    def __init__(self, host: str, port: int | None = 8000, api_key: str | None = None,
                 connect_retry_s: float = 5.0, jpeg_quality: int | None = None):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None and ":" not in self._uri.split("//", 1)[-1]:
            self._uri += f":{port}"
        self._api_key = api_key
        self._connect_retry_s = connect_retry_s
        self._packer = msgpack_numpy.Packer()
        self._jpeg_quality = jpeg_quality
        self._encode_frame = None
        self.last_request_bytes: int | None = None
        if jpeg_quality is not None:
            self._check_jpeg_quality(jpeg_quality)
            self._encode_frame = self._pick_jpeg_encoder(jpeg_quality)
        self._ws, self.metadata = self._connect()
        if jpeg_quality is not None and "jpeg" not in self.metadata.get("image_encodings", ["raw"]):
            raise ValueError(
                f"server at {self._uri} does not support jpeg encoding "
                "(metadata lacks 'jpeg' in image_encodings) — upgrade the server or drop jpeg_quality"
            )

    @staticmethod
    def _check_jpeg_quality(quality) -> None:
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError(f"jpeg_quality must be an integer between 1 and 100, got {quality!r}")

    @staticmethod
    def _pick_jpeg_encoder(quality: int):
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
            import io

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

    def _connect(self):
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        while True:
            try:
                ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(ws.recv())
                logger.info(
                    f"connected to {self._uri}: {metadata.get('policy_type')} "
                    f"from {metadata.get('checkpoint')}"
                )
                return ws, metadata
            except websockets.exceptions.InvalidStatus as e:
                if e.response.status_code == 403:
                    raise PolicyServerError(
                        f"server at {self._uri} rejected the api key — check --api_key on both sides"
                    ) from None
                raise
            except OSError:
                logger.info(f"waiting for server at {self._uri} "
                            f"(retrying every {self._connect_retry_s:.0f}s)...")
                time.sleep(self._connect_retry_s)

    def infer(self, obs: dict) -> dict:
        """One observation dict in, one action chunk out. See server metadata
        (self.metadata['observations']) for the expected keys/shapes/dtypes."""
        self._validate(obs)
        if self._encode_frame is not None:
            obs = self._encode_jpeg(obs)
        payload = self._packer.pack(obs)
        self.last_request_bytes = len(payload)
        try:
            self._ws.send(payload)
            response = self._ws.recv()
        except (websockets.exceptions.ConnectionClosed, OSError):
            logger.warning("connection to policy server lost; reconnecting...")
            self._ws, self.metadata = self._connect()
            self._ws.send(payload)
            response = self._ws.recv()
        if isinstance(response, str):
            raise PolicyServerError(response)
        reply = msgpack_numpy.unpackb(response)
        if isinstance(reply.get("actions"), np.ndarray):
            # msgpack-decoded arrays are read-only views of the receive buffer;
            # robot code mutates chunks (clipping, scaling) — hand back a copy
            reply["actions"] = np.array(reply["actions"])
        return reply

    def _validate(self, obs: dict) -> None:
        spec = self.metadata["observations"]
        n_obs = self.metadata.get("n_obs_steps", 1)
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

    def _encode_jpeg(self, obs: dict) -> dict:
        """Encode validated raw image arrays to per-frame JPEG bytes (oldest
        first). Callers keep passing raw arrays — this is transport-only."""
        encoded = dict(obs)
        for key, spec in self.metadata["observations"].items():
            if spec["kind"] != "image":
                continue
            arr = obs[key]
            frames = arr if arr.ndim == 4 else arr[None]
            encoded[key] = [self._encode_frame(np.ascontiguousarray(f)) for f in frames]
        return encoded

    def close(self) -> None:
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
