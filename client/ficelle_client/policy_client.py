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
                 connect_retry_s: float = 5.0):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None and ":" not in self._uri.split("//", 1)[-1]:
            self._uri += f":{port}"
        self._api_key = api_key
        self._connect_retry_s = connect_retry_s
        self._packer = msgpack_numpy.Packer()
        self._ws, self.metadata = self._connect()

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
            except ConnectionRefusedError:
                logger.info(f"waiting for server at {self._uri} "
                            f"(retrying every {self._connect_retry_s:.0f}s)...")
                time.sleep(self._connect_retry_s)

    def infer(self, obs: dict) -> dict:
        """One observation dict in, one action chunk out. See server metadata
        (self.metadata['observations']) for the expected keys/shapes/dtypes."""
        self._validate(obs)
        payload = self._packer.pack(obs)
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
        return msgpack_numpy.unpackb(response)

    def _validate(self, obs: dict) -> None:
        spec = self.metadata["observations"]
        n_obs = self.metadata.get("n_obs_steps", 1)
        if "task" in obs and not isinstance(obs["task"], str):
            raise ValueError(f"'task' must be a string, got {type(obs['task']).__name__}")
        for key in obs:
            if key != "task" and key not in spec:
                raise ValueError(
                    f"unknown observation key '{key}' — the server expects {sorted(spec)}"
                )
        for key, s in spec.items():
            if key not in obs:
                raise ValueError(
                    f"missing observation key '{key}' — the server expects {sorted(spec)}"
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

    def close(self) -> None:
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
