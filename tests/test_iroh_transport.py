"""Server-half tests for the iroh transport: a raw iroh client (mirroring
experiments/iroh_probe/iroh_probe.py) drives IrohPolicyServer in-process, over
loopback only (RelayMode.disabled(), dial server.addr() directly — no network,
no relays, no n0 discovery)."""

import asyncio
import pathlib
import subprocess
import sys
import threading
import time

import iroh
import numpy as np
import pytest

from iroh_transport import ALPN, MAX_FRAME, OP_INFER, OP_META, ST_ERR, ST_OK, IrohPolicyServer
import ficelle_client
from ficelle_client import IrohPolicyClient, PolicyServerError, msgpack_numpy
from policy_host import PolicyHost
from tests.test_policy_host import StubConfig, StubPolicy, identity


def _set_event_loop() -> None:
    try:
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    except AttributeError:
        pass


def _stub_host() -> PolicyHost:
    return PolicyHost(StubPolicy(StubConfig()), identity, identity,
                      checkpoint="stub/ckpt", device="cpu", fps=None)


def _good_obs() -> dict:
    return {
        "observation.images.cam0": np.zeros((2, 96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros((2, 2), dtype=np.float32),
        "task": "pick",
    }


async def _client_endpoint() -> "iroh.Endpoint":
    opts = iroh.EndpointOptions(relay_mode=iroh.RelayMode.disabled())
    return await iroh.Endpoint.bind(opts)


async def _request(server: IrohPolicyServer, payload: bytes) -> bytes:
    """Dial the running server directly on loopback and return the raw reply."""
    _set_event_loop()
    client_ep = await _client_endpoint()
    try:
        conn = await client_ep.connect(server.addr(), ALPN)
        bi = await conn.open_bi()
        send, recv = bi.send(), bi.recv()
        await send.write_all(payload)
        await send.finish()
        return await recv.read_to_end(MAX_FRAME)
    finally:
        await client_ep.close()


async def _run_against_server(payload: bytes) -> bytes:
    """Start a fresh IrohPolicyServer, issue one request, tear down.

    Note: IrohPolicyServer._options() always binds with iroh.preset_n0()
    (n0 public relays + discovery enabled) — it has no disable knob, only a
    relay_url override for custom relays. This bind call itself does not
    require live connectivity (verified: it succeeds offline), and the
    client dials server.addr() directly over loopback, so the request/reply
    exchanged in this test never touches the network or n0's relays.
    """
    server = IrohPolicyServer(_stub_host(), relay_url=None)
    server_task = asyncio.ensure_future(server.run())
    try:
        await server.wait_ready()
        return await _request(server, payload)
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def test_meta_opcode_returns_metadata():
    async def body():
        reply = await _run_against_server(bytes([OP_META]))
        meta = msgpack_numpy.unpackb(reply)
        assert meta["protocol"] == "ficelle/1"

    asyncio.run(body())


def test_infer_opcode_returns_actions():
    async def body():
        payload = bytes([OP_INFER]) + msgpack_numpy.packb(_good_obs())
        reply = await _run_against_server(payload)
        status, body_bytes = reply[0], reply[1:]
        assert status == ST_OK
        result = msgpack_numpy.unpackb(body_bytes)
        assert result["actions"].shape == (8, 11)

    asyncio.run(body())


def test_infer_bad_obs_returns_status_err_one_line():
    async def body():
        obs = _good_obs()
        obs["observation.state"] = np.zeros((2, 3), dtype=np.float32)  # wrong shape
        payload = bytes([OP_INFER]) + msgpack_numpy.packb(obs)
        reply = await _run_against_server(payload)
        status, body_bytes = reply[0], reply[1:]
        assert status == ST_ERR
        text = body_bytes.decode()
        assert text.startswith("ficelle: ")
        assert "\n" not in text
        assert "observation.state" in text

    asyncio.run(body())


def test_bind_failure_surfaces():
    """A startup failure (here: an invalid relay_url, rejected by iroh while
    building EndpointOptions before bind) must raise out of wait_ready(),
    not hang it forever — _ready has to be set on the failure path too."""
    async def body():
        server = IrohPolicyServer(_stub_host(), relay_url="not a url")
        server_task = asyncio.ensure_future(server.run())
        with pytest.raises(iroh.iroh_ffi.IrohError):
            await server.wait_ready()
        with pytest.raises(iroh.iroh_ffi.IrohError):
            await server_task

    asyncio.run(body())


def test_unknown_opcode_errors():
    async def body():
        reply = await _run_against_server(bytes([0x09]))
        status, body_bytes = reply[0], reply[1:]
        assert status == ST_ERR
        text = body_bytes.decode()
        assert text.startswith("ficelle: ")
        assert "\n" not in text

    asyncio.run(body())


# ---------------------------------------------------------------------------
# Client-half tests: the SYNC IrohPolicyClient against a real IrohPolicyServer.
#
# The server runs in a SEPARATE PROCESS (tests/_iroh_stub_server.py), not a
# background thread: iroh-ffi's uniffi_set_event_loop is process-global, so
# co-hosting a server loop and the client's loop in one process makes the
# second registration win and deadlocks the first. Separate processes is also
# how real deployments and experiments/iroh_probe run. Each test spawns the
# stub server, reads its printed TICKET, drives the in-process client, then
# terminates the subprocess.
# ---------------------------------------------------------------------------

_STUB_SERVER = str(pathlib.Path(__file__).with_name("_iroh_stub_server.py"))


class StubServerProcess:
    """Spawn tests/_iroh_stub_server.py, block until it prints its ticket."""

    def __init__(self, *extra_args: str, ticket_timeout: float = 30.0):
        self._proc = subprocess.Popen(
            [sys.executable, _STUB_SERVER, *extra_args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.ticket = self._read_ticket(ticket_timeout)

    def _read_ticket(self, timeout: float) -> str:
        """Read stdout on a helper thread so a wedged spawn can't hang the
        suite forever — if no TICKET line arrives in `timeout`, kill the
        process and fail with whatever it wrote to stderr."""
        result: dict = {}

        def reader():
            for line in self._proc.stdout:
                if line.startswith("TICKET "):
                    result["ticket"] = line[len("TICKET "):].strip()
                    return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        if "ticket" not in result:
            self._proc.kill()
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"stub iroh server never printed a ticket in {timeout}s:\n{stderr}")
        return result["ticket"]

    def stop(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)


def _stacked_obs() -> dict:
    # Gradient content (not i.i.d. noise): JPEG needs spatial correlation to
    # compress at all — mirrors tests/test_policy_client.py's helper.
    xx, yy = np.meshgrid(np.arange(96, dtype=np.uint8) * 2, np.arange(96, dtype=np.uint8) * 2)
    frame0 = np.stack([xx, yy, np.full_like(xx, 32)], axis=-1)
    frame1 = np.stack([np.full_like(xx, 200), xx[::-1], yy], axis=-1)
    return {
        "observation.images.cam0": np.stack([frame0, frame1]),
        "observation.state": np.zeros((2, 2), dtype=np.float32),
        "task": "pick",
    }


def test_client_metadata_and_infer():
    srv = StubServerProcess()
    try:
        with IrohPolicyClient(srv.ticket) as client:
            assert client.metadata["policy_type"] == "stub"
            reply = client.infer(_good_obs())
            assert reply["actions"].shape == (8, 11)
    finally:
        srv.stop()


def test_client_validation_before_send():
    # Same message as the websocket client (tests/test_policy_client.py's
    # test_client_side_validation) — both go through the shared _obsval module.
    srv = StubServerProcess()
    try:
        with IrohPolicyClient(srv.ticket) as client:
            obs = _good_obs()
            obs["observation.state"] = np.zeros((2, 3), dtype=np.float32)
            with pytest.raises(ValueError, match=r"shape \[2, 3\], expected \[2, 2\]"):
                client.infer(obs)
    finally:
        srv.stop()


def test_client_server_error_raises():
    # obs passes client-side validation (correct keys/shapes/dtypes) but the
    # stub host rejects it server-side (--reject-task raises ObservationError
    # on task=="reject me") — the one-line server text reaches the client as a
    # PolicyServerError.
    srv = StubServerProcess("--reject-task")
    try:
        with IrohPolicyClient(srv.ticket) as client:
            obs = _good_obs()
            obs["task"] = "reject me"
            with pytest.raises(PolicyServerError, match="reject me"):
                client.infer(obs)
    finally:
        srv.stop()


def test_client_jpeg_roundtrip():
    # host._policy.seen_batch is unreachable across the process boundary, so
    # the echo-image stub returns the decoded cam0 per-channel mean in the
    # action chunk. A known gradient obs (jpeg_quality=90) must round-trip to
    # matching means within JPEG tolerance — still catches an RGB/BGR swap
    # end-to-end across the real transport.
    srv = StubServerProcess("--echo-image")
    try:
        with IrohPolicyClient(srv.ticket, jpeg_quality=90) as client:
            obs = _stacked_obs()
            reply = client.infer(obs)
            assert client.last_request_bytes is not None
            assert client.last_request_bytes < obs["observation.images.cam0"].nbytes / 5
            returned_means = reply["actions"][0, :3]  # per-channel mean in [0,1]
            expected = obs["observation.images.cam0"].astype(float).mean(axis=(0, 1, 2)) / 255.0
            assert np.allclose(returned_means, expected, atol=0.02), (
                f"channel means {returned_means} vs expected {expected} — "
                "JPEG parity failure (possible RGB/BGR swap)"
            )
    finally:
        srv.stop()


def test_infer_timeout():
    srv = StubServerProcess("--sleep-infer", "2.0")
    try:
        with IrohPolicyClient(srv.ticket, infer_timeout=0.2) as client:
            t0 = time.monotonic()
            with pytest.raises(PolicyServerError, match="timed out after 0.2s"):
                client.infer(_good_obs())
            elapsed = time.monotonic() - t0
            assert elapsed < 1.5, f"infer() should not hang past the timeout, took {elapsed:.2f}s"
    finally:
        srv.stop()


def test_open_client_dispatch(monkeypatch):
    calls = {}

    class FakeIrohClient:
        def __init__(self, ticket, **kw):
            calls["iroh"] = (ticket, kw)

    class FakePolicyClient:
        def __init__(self, addr, **kw):
            calls["ws"] = (addr, kw)

    monkeypatch.setattr(ficelle_client, "IrohPolicyClient", FakeIrohClient)
    monkeypatch.setattr(ficelle_client, "PolicyClient", FakePolicyClient)

    iroh_client = ficelle_client.open_client("endpointv1abcdef")
    assert isinstance(iroh_client, FakeIrohClient)
    assert calls["iroh"][0] == "endpointv1abcdef"

    ws_client = ficelle_client.open_client("127.0.0.1:8000")
    assert isinstance(ws_client, FakePolicyClient)
    assert calls["ws"][0] == "127.0.0.1:8000"
