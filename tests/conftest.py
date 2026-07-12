import asyncio
import socket
import threading
import time

import numpy as np
import pytest

from errors import ObservationError


class FakeHost:
    """Stands in for policy_host.PolicyHost in transport/client tests:
    no lerobot, no checkpoint, instant answers."""

    metadata = {
        "protocol": "ficelle/1",
        "checkpoint": "fake/checkpoint",
        "policy_type": "fake",
        "device": "cpu",
        "n_obs_steps": 1,
        "n_action_steps": 10,
        "chunk_size": 10,
        "action_dim": 2,
        "fps": 30.0,
        "observations": {
            "observation.images.cam0": {"kind": "image", "dtype": "uint8",
                                        "frame_shape": [96, 96, 3], "wire_shape": [96, 96, 3]},
            "observation.state": {"kind": "state", "dtype": "float32",
                                  "frame_shape": [2], "wire_shape": [2]},
        },
    }

    def infer(self, obs):
        expected = set(self.metadata["observations"])
        unknown = {k for k in obs if k.startswith("observation.")} - expected
        if unknown:
            raise ObservationError(
                f"unknown observation key(s) {sorted(unknown)} — expected {sorted(expected)}"
            )
        if obs.get("task") == "reject me":       # hook for server-side-error tests
            raise ObservationError("task 'reject me' rejected as requested")
        if obs.get("task") == "crash":           # hook for internal-error tests
            raise RuntimeError("fake internal crash")
        return {"actions": np.zeros((10, 2), dtype=np.float32)}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """Runs a FicelleServer in a daemon thread. .stop() shuts it down cleanly
    so tests can restart a server on the same port (restart-resilience tests)."""

    def __init__(self, host_obj, port: int | None = None, api_key: str | None = None):
        from serve import FicelleServer

        self.port = port if port is not None else free_port()
        self._server = FicelleServer(host_obj, "127.0.0.1", self.port, api_key=api_key)
        self._loop = asyncio.new_event_loop()
        self._stop_event: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._wait_listening()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        self._stop_event = asyncio.Event()
        server_task = asyncio.ensure_future(self._server.run())
        stop_task = asyncio.ensure_future(self._stop_event.wait())
        await asyncio.wait([server_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    def _wait_listening(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"server did not start listening on port {self.port}")

    def stop(self):
        while self._stop_event is None:
            time.sleep(0.01)
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)


@pytest.fixture
def fake_host():
    return FakeHost()


@pytest.fixture
def server(fake_host):
    st = ServerThread(fake_host)
    yield st
    st.stop()
