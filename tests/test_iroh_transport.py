"""Server-half tests for the iroh transport: a raw iroh client (mirroring
experiments/iroh_probe/iroh_probe.py) drives IrohPolicyServer in-process, over
loopback only (RelayMode.disabled(), dial server.addr() directly — no network,
no relays, no n0 discovery)."""

import asyncio
import logging
import pathlib
import subprocess
import sys
import time

import iroh
import numpy as np
import pytest

from errors import POLICY_GENERATION_ERROR
from iroh_transport import ALPN, MAX_FRAME, OP_INFER, OP_META, ST_ERR, ST_OK, IrohPolicyServer
import ficelle_client
from ficelle_client import IrohPolicyClient, PolicyServerError, msgpack_numpy
from policy_host import PolicyHost
from tests._helpers import gradient_frame, read_ticket, stacked_obs
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


async def _run_against_server(payload: bytes, host=None) -> bytes:
    """Start a fresh IrohPolicyServer, issue one request, tear down.

    Note: IrohPolicyServer._options() always binds with iroh.preset_n0()
    (n0 public relays + discovery enabled) — it has no disable knob, only a
    relay_url override for custom relays. This bind call itself does not
    require live connectivity (verified: it succeeds offline), and the
    client dials server.addr() directly over loopback, so the request/reply
    exchanged in this test never touches the network or n0's relays.
    """
    server = IrohPolicyServer(host if host is not None else _stub_host(), relay_url=None)
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


def test_wire_constants_match_client():
    """Guards the hand-synced wire contract: ficelle_client/iroh_client.py
    re-declares ALPN/opcodes/status/MAX_FRAME instead of importing them from
    this module (it must not import iroh_transport.py — see that module's
    docstring, it pulls in torch-adjacent server deps). If the two ever drift,
    client and server silently talk past each other."""
    from ficelle_client import iroh_client

    for name in ("ALPN", "OP_META", "OP_INFER", "ST_OK", "ST_ERR", "MAX_FRAME"):
        server_val = globals()[name]
        client_val = getattr(iroh_client, name)
        assert server_val == client_val, f"{name}: server={server_val!r} client={client_val!r}"


def test_unknown_opcode_errors():
    async def body():
        reply = await _run_against_server(bytes([0x09]))
        status, body_bytes = reply[0], reply[1:]
        assert status == ST_ERR
        text = body_bytes.decode()
        assert text.startswith("ficelle: ")
        assert "\n" not in text

    asyncio.run(body())


def test_assertion_error_gets_friendly_message():
    # Same case as test_server_transport.py's websocket counterpart: an
    # upstream lerobot AssertionError (degenerate generation on an
    # observation that already passed ficelle's own validation) must get the
    # same actionable one-liner as the websocket transport, not the generic
    # "internal server error (AssertionError)" text.
    class AssertingStubPolicy(StubPolicy):
        def predict_action_chunk(self, batch):
            raise AssertionError("fake upstream assertion")

    async def body():
        host = PolicyHost(AssertingStubPolicy(StubConfig()), identity, identity,
                          checkpoint="stub/ckpt", device="cpu", fps=None)
        payload = bytes([OP_INFER]) + msgpack_numpy.packb(_good_obs())
        reply = await _run_against_server(payload, host=host)
        status, body_bytes = reply[0], reply[1:]
        assert status == ST_ERR
        assert body_bytes.decode() == POLICY_GENERATION_ERROR

    asyncio.run(body())



@pytest.mark.parametrize("finish_request", [False, True], ids=["mid-request", "mid-inference"])
def test_client_disconnect_is_not_a_server_error(caplog, finish_request):
    """A client that goes away (killed, Ctrl-C, network drop) while its
    request is still being read, or while inference runs, is routine: the
    server must not log it as an internal error, nor crash the stream task
    trying to send an error reply on the dead stream."""
    class SlowHost:
        metadata = {"protocol": "ficelle/1"}

        def infer(self, obs):
            time.sleep(0.5)
            return {"actions": np.zeros((8, 11), dtype=np.float32)}

    async def body():
        _set_event_loop()
        server = IrohPolicyServer(SlowHost(), relay_url=None)
        server_task = asyncio.ensure_future(server.run())
        try:
            await server.wait_ready()
            client_ep = await _client_endpoint()
            conn = await client_ep.connect(server.addr(), ALPN)
            send = (await conn.open_bi()).send()
            await send.write_all(bytes([OP_INFER]) + msgpack_numpy.packb(_good_obs()))
            if finish_request:
                await send.finish()
            await asyncio.sleep(0.2)
            conn.close(0, b"")
            await client_ep.close()
            await asyncio.sleep(1.0)  # let the server notice and the stream task end
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(body())
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, [r.getMessage() for r in errors]

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
        return read_ticket(self._proc, timeout, "stub iroh server")

    def stop(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)


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
            obs = stacked_obs()
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


def test_client_resize_shrinks_and_infers():
    # mirrors tests/test_policy_client.py's test_resize_shrinks_and_infers:
    # ResizeStubConfig advertises resize_to=[48, 48]; caller keeps passing
    # full (2, 96, 96, 3) frames, the client downscales+jpeg-encodes before
    # send over the real iroh transport (subprocess stub server, --resize).
    srv = StubServerProcess("--resize")
    try:
        with IrohPolicyClient(srv.ticket, resize=True, jpeg_quality=90) as client:
            obs = stacked_obs()
            reply = client.infer(obs)
            assert reply["actions"].shape == (8, 11)
            assert client.last_request_bytes < obs["observation.images.cam0"].nbytes / 10
    finally:
        srv.stop()


def test_infer_recovers_after_timeout():
    # Remote-robustness property: a timed-out request must not wedge the client.
    # The stub sleeps 0.7s on the FIRST infer only; with infer_timeout=0.5 the
    # first call times out, then the SECOND call on the SAME client succeeds
    # (fresh open_bi per request; the abandoned first request doesn't wedge the
    # background loop). Timing note: PolicyHost serializes inference under a
    # lock, so the slow first request holds it for 0.7s; the second request's
    # own 0.5s deadline (starting ~0.5s in, right after the first times out)
    # therefore extends past that 0.7s lock release, so it gets served.
    srv = StubServerProcess("--slow-first", "0.7")
    try:
        with IrohPolicyClient(srv.ticket, infer_timeout=0.5) as client:
            t0 = time.monotonic()
            with pytest.raises(PolicyServerError, match="timed out after 0.5s"):
                client.infer(_good_obs())
            assert time.monotonic() - t0 < 1.2
            reply = client.infer(_good_obs())  # must not hang or time out
            assert reply["actions"].shape == (8, 11)
    finally:
        srv.stop()


def test_infer_reconnects_after_connection_lost():
    # Remote-robustness property: a dropped QUIC connection (wifi blip, NAT
    # rebinding, idle timeout) must not kill the client for good — the next
    # infer() dials again and succeeds. The drop is simulated by closing the
    # client's own connection under it.
    srv = StubServerProcess()
    try:
        with IrohPolicyClient(srv.ticket) as client:
            assert client.infer(_good_obs())["actions"].shape == (8, 11)

            async def _drop():
                client._conn.close(0, b"")
            client._call(_drop())
            reply = client.infer(_good_obs())
            assert reply["actions"].shape == (8, 11)
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


# ---------------------------------------------------------------------------
# Real-checkpoint e2e over iroh: drives the REAL production entrypoint
# (`serve.py --transport iroh`, not the stub server used above) against a
# real diffusion checkpoint, mirroring tests/test_real_diffusion.py's
# websocket e2e test (test_end_to_end_over_websocket) but over the iroh
# transport. Same process-separation reasoning as the client-half tests
# above: iroh-ffi's uniffi_set_event_loop is process-global, so the server
# runs as a subprocess while IrohPolicyClient (its own thread + loop) runs
# in-process. Slow (checkpoint download + load); marked `checkpoint`.
# ---------------------------------------------------------------------------

_REAL_CKPT = "SteveNguyen/diffusion_grabette_simu_release"
_SERVE_PY = str(pathlib.Path(__file__).resolve().parent.parent / "serve.py")


class ServePyProcess:
    """Spawn `serve.py --transport iroh --checkpoint <checkpoint>`, block
    until it prints its ticket. Mirrors StubServerProcess above but drives
    the real CLI entrypoint with a real checkpoint (HF download + model load
    happen before the ticket is printed), hence the generous default
    timeout."""

    def __init__(self, checkpoint: str, ticket_timeout: float = 120.0):
        self._proc = subprocess.Popen(
            [sys.executable, _SERVE_PY, "--transport", "iroh", "--checkpoint", checkpoint],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.ticket = self._read_ticket(ticket_timeout)

    def _read_ticket(self, timeout: float) -> str:
        return read_ticket(self._proc, timeout, "serve.py", reap_timeout=10)

    def stop(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)


def _real_ckpt_obs(meta: dict) -> dict:
    """Synthetic observation dict matching the checkpoint's metadata: image
    keys get stacked gradient frames (so raw-vs-jpeg size is a meaningful
    comparison), everything else gets random floats."""
    rng = np.random.default_rng(0)
    obs = {"task": ""}
    for key, spec in meta["observations"].items():
        shape = spec["wire_shape"]
        if spec["dtype"] == "uint8":
            h, w = shape[-3], shape[-2]
            obs[key] = np.broadcast_to(gradient_frame(h, w), shape).copy()
        else:
            obs[key] = rng.standard_normal(shape).astype(np.float32)
    return obs


@pytest.mark.checkpoint
def test_real_diffusion_checkpoint_over_iroh():
    srv = ServePyProcess(_REAL_CKPT)
    try:
        with IrohPolicyClient(srv.ticket) as client:
            meta = client.metadata
            assert meta["policy_type"] == "diffusion"
            assert meta["n_obs_steps"] == 2
            assert (meta["n_action_steps"], meta["action_dim"]) == (8, 11)
            # guards against a lerobot attr rename (resize_shape) silently
            # disabling resize for this checkpoint without any test noticing
            image_specs = [s for s in meta["observations"].values() if s["kind"] == "image"]
            assert image_specs and any(s.get("resize_to") for s in image_specs), (
                "expected at least one image observation to advertise resize_to"
            )

            obs = _real_ckpt_obs(meta)
            reply = client.infer(obs)
            actions = reply["actions"]
            assert actions.shape == (8, 11)
            assert actions.dtype == np.float32
            assert np.isfinite(actions).all()
            raw_bytes = client.last_request_bytes

        # Fresh connection (a new ticket-dial) for the jpeg side of the
        # comparison, same server process.
        with IrohPolicyClient(srv.ticket, jpeg_quality=90) as jpeg_client:
            jpeg_client.infer(obs)
            jpeg_bytes = jpeg_client.last_request_bytes

        assert raw_bytes is not None and jpeg_bytes is not None
        assert jpeg_bytes < raw_bytes, (
            f"jpeg request ({jpeg_bytes} bytes) should be smaller than raw "
            f"({raw_bytes} bytes) for gradient image content"
        )
    finally:
        srv.stop()
