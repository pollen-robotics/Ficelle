"""Server-half tests for the iroh transport: a raw iroh client (mirroring
experiments/iroh_probe/iroh_probe.py) drives IrohPolicyServer in-process, over
loopback only (RelayMode.disabled(), dial server.addr() directly — no network,
no relays, no n0 discovery)."""

import asyncio

import iroh
import numpy as np
import pytest

from iroh_transport import ALPN, MAX_FRAME, OP_INFER, OP_META, ST_ERR, ST_OK, IrohPolicyServer
from ficelle_client import msgpack_numpy
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
