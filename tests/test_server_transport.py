import urllib.request

import numpy as np
import pytest
import websockets.exceptions
import websockets.sync.client

from errors import POLICY_GENERATION_ERROR
from ficelle_client import msgpack_numpy
from tests.conftest import ServerThread


def _connect(port, api_key=None):
    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    return websockets.sync.client.connect(
        f"ws://127.0.0.1:{port}", compression=None, max_size=None, additional_headers=headers
    )


def _good_obs(task=""):
    return {
        "observation.images.cam0": np.zeros((96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros(2, dtype=np.float32),
        "task": task,
    }


def test_metadata_sent_on_connect(server):
    with _connect(server.port) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
    assert meta["protocol"] == "ficelle/1"
    assert meta["n_action_steps"] == 10


def test_infer_roundtrip(server):
    with _connect(server.port) as ws:
        ws.recv()  # metadata
        ws.send(msgpack_numpy.packb(_good_obs()))
        reply = msgpack_numpy.unpackb(ws.recv())
    assert reply["actions"].shape == (10, 2)
    assert reply["actions"].dtype == np.float32
    assert reply["server_timing"]["total_ms"] > 0


def test_validation_error_is_one_line_text_and_keeps_connection(server):
    with _connect(server.port) as ws:
        ws.recv()
        bad = _good_obs()
        bad["observation.images.wrong"] = bad.pop("observation.images.cam0")
        ws.send(msgpack_numpy.packb(bad))
        reply = ws.recv()
        assert isinstance(reply, str)
        assert reply.startswith("ficelle: ")
        assert "\n" not in reply.strip()
        assert "observation.images.wrong" in reply
        # connection still usable after a validation error
        ws.send(msgpack_numpy.packb(_good_obs()))
        assert isinstance(ws.recv(), bytes)


def test_internal_error_sends_text_then_closes(server):
    with _connect(server.port) as ws:
        ws.recv()
        ws.send(msgpack_numpy.packb(_good_obs(task="crash")))
        reply = ws.recv()
        assert isinstance(reply, str) and "internal server error" in reply
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            ws.recv()


def test_assertion_error_gets_friendly_message_and_closes(server):
    # An upstream lerobot AssertionError (degenerate generation on an
    # observation that already passed ficelle's own validation) must get the
    # actionable one-liner, not the generic "internal server error" text —
    # but the connection still closes, same as any other internal error.
    with _connect(server.port) as ws:
        ws.recv()
        ws.send(msgpack_numpy.packb(_good_obs(task="crash assert")))
        reply = ws.recv()
        assert reply == POLICY_GENERATION_ERROR
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            ws.recv()


def test_non_dict_payload_rejected(server):
    with _connect(server.port) as ws:
        ws.recv()  # metadata
        ws.send(msgpack_numpy.packb([1, 2, 3]))
        reply = ws.recv()
        assert isinstance(reply, str)
        assert reply.startswith("ficelle: ")
        assert "observation dict" in reply
        # connection still usable after a validation error
        ws.send(msgpack_numpy.packb(_good_obs()))
        assert isinstance(ws.recv(), bytes)


def test_server_survives_client_disconnect(server):
    for _ in range(2):  # connect, use, hard-drop, repeat
        ws = _connect(server.port)
        ws.recv()
        ws.send(msgpack_numpy.packb(_good_obs()))
        ws.recv()
        ws.close()


def test_api_key(fake_host):
    st = ServerThread(fake_host, api_key="sekret")
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus) as excinfo:
            with _connect(st.port):
                pass
        assert excinfo.value.response.status_code == 403
        with _connect(st.port, api_key="sekret") as ws:
            assert msgpack_numpy.unpackb(ws.recv())["protocol"] == "ficelle/1"
    finally:
        st.stop()


def test_healthz(server):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz") as resp:
        assert resp.status == 200
