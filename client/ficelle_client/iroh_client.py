"""Sync-over-async iroh client for a ficelle policy server.

Speaks the exact wire protocol IrohPolicyServer implements — see
iroh_transport.py (repo root, server side) for the protocol description; that
file is the source of truth for the 4 opcode/status constants + ALPN
re-declared below. This package must NOT import that module: the server pulls
in torch-adjacent deps (lerobot, cv2, ...) this dependency-light client avoids.

Runs its own asyncio event loop on a daemon thread so PolicyClient-style
callers get a plain synchronous .infer(obs) even though iroh's Python
bindings are async-only (see experiments/iroh_probe/iroh_probe.py for the
proven iroh 1.0 API this mirrors).
"""

import asyncio
import concurrent.futures
import logging
import threading

import numpy as np

from ficelle_client import _obsval, msgpack_numpy
from ficelle_client.policy_client import PolicyServerError

logger = logging.getLogger("ficelle.client")

# Mirrors iroh_transport.py's protocol constants (repo root, server side) —
# keep these 4 + ALPN in sync with that file by hand; do not import it here.
ALPN = b"ficelle/1"
OP_META = 0x00
OP_INFER = 0x01
ST_OK = 0x00
ST_ERR = 0x01
MAX_FRAME = 8_000_000

_CONNECT_TIMEOUT_S = 30.0


class IrohPolicyClient:
    """Connects to a ficelle server over iroh (QUIC + NAT hole-punching),
    reads its metadata, then serves .infer(obs) -> {"actions": (n_action_steps,
    action_dim) float32, "server_timing": {...}} — the same contract as
    PolicyClient (websocket transport), just over a different wire.

    Only ONE IrohPolicyClient (or one iroh endpoint) may be live per process:
    iroh-ffi's uniffi event loop is process-global, so a second concurrent one
    registering itself makes the second registration win and stalls the first.
    """

    def __init__(self, ticket: str, jpeg_quality: int | None = None,
                 infer_timeout: float = 5.0, relay_url: str | None = None,
                 resize: bool = False):
        try:
            import iroh
        except ImportError:
            raise ValueError(
                "iroh transport needs the extra: pip install 'ficelle-client[iroh]'"
            ) from None
        self._iroh = iroh
        self._infer_timeout = infer_timeout
        self._packer = msgpack_numpy.Packer()
        self._encode_frame = None
        self._resize = resize
        self._resizer = None
        self.last_request_bytes: int | None = None
        self._ep = None
        self._conn = None
        self._closed = False
        if jpeg_quality is not None:
            _obsval.check_jpeg_quality(jpeg_quality)
            self._encode_frame = _obsval.pick_jpeg_encoder(jpeg_quality)
        if resize:
            self._resizer = _obsval.pick_resizer()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        try:
            self.metadata = self._call(self._aconnect(ticket, relay_url), timeout=_CONNECT_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            self.close()
            raise PolicyServerError(
                f"could not connect to the iroh server within {_CONNECT_TIMEOUT_S:.0f}s — "
                "check the ticket is current and the server is running"
            ) from None
        except Exception:
            self.close()
            raise
        if jpeg_quality is not None and "jpeg" not in self.metadata.get("image_encodings", ["raw"]):
            self.close()
            raise ValueError(
                "server does not support jpeg encoding (metadata lacks 'jpeg' in "
                "image_encodings) — upgrade the server or drop jpeg_quality"
            )
        if resize and not _obsval.resolve_resize(self.metadata):
            self.close()
            raise ValueError(
                "server does not advertise resize (its policy doesn't resize internally)"
            )
        if resize and jpeg_quality is None:
            logger.info(
                "resize without jpeg_quality: raw resized frames are still large "
                "over the network — jpeg_quality=90 is the recommended combo"
            )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._iroh.iroh_ffi.uniffi_set_event_loop(self._loop)
        except AttributeError:
            pass  # newer bindings may set this implicitly
        self._loop.run_forever()

    def _call(self, coro, timeout: float | None = None):
        """Run a coroutine on the background loop, block this (any) thread
        for the result. On timeout the coroutine is left running on the loop
        thread (its result is simply discarded) — the caller never hangs."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    async def _aconnect(self, ticket: str, relay_url: str | None) -> dict:
        iroh = self._iroh
        kwargs = {"preset": iroh.preset_n0()}
        if relay_url is not None:
            kwargs["relay_mode"] = iroh.RelayMode.custom_from_urls([relay_url])
        self._ep = await iroh.Endpoint.bind(iroh.EndpointOptions(**kwargs))
        addr = iroh.EndpointTicket.from_string(ticket).endpoint_addr()
        self._conn = await self._ep.connect(addr, ALPN)
        reply = await self._arequest(bytes([OP_META]))
        return msgpack_numpy.unpackb(reply)

    async def _arequest(self, payload: bytes) -> bytes:
        bi = await self._conn.open_bi()
        send, recv = bi.send(), bi.recv()
        await send.write_all(payload)
        await send.finish()
        return await recv.read_to_end(MAX_FRAME)

    def infer(self, obs: dict) -> dict:
        """One observation dict in, one action chunk out. See server metadata
        (self.metadata['observations']) for the expected keys/shapes/dtypes."""
        _obsval.validate_obs(self.metadata, obs)
        if self._resize:
            obs = _obsval.resize_frames(obs, self.metadata["observations"], self._resizer)
        if self._encode_frame is not None:
            obs = _obsval.encode_jpeg(self.metadata, obs, self._encode_frame)
        payload = bytes([OP_INFER]) + self._packer.pack(obs)
        self.last_request_bytes = len(payload)
        try:
            reply = self._call(self._arequest(payload), timeout=self._infer_timeout)
        except concurrent.futures.TimeoutError:
            raise PolicyServerError(
                f"inference timed out after {self._infer_timeout:.1f}s (link degraded or server stuck)"
            ) from None
        status, body = reply[0], reply[1:]
        if status == ST_ERR:
            raise PolicyServerError(body.decode())
        reply_obj = msgpack_numpy.unpackb(body)
        if isinstance(reply_obj.get("actions"), np.ndarray):
            # msgpack-decoded arrays are read-only views of the receive buffer;
            # robot code mutates chunks (clipping, scaling) — hand back a copy
            reply_obj["actions"] = np.array(reply_obj["actions"])
        return reply_obj

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _aclose():
            if self._conn is not None:
                try:
                    self._conn.close(0, b"")
                except Exception:  # noqa: BLE001 — best-effort, we're tearing down anyway
                    pass
            if self._ep is not None:
                await self._ep.close()

        try:
            self._call(_aclose(), timeout=5.0)
        except Exception:  # noqa: BLE001 — best-effort, we're tearing down anyway
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
