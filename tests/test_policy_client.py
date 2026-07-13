import numpy as np
import pytest

from ficelle_client import PolicyClient, PolicyServerError
from policy_host import PolicyHost
from tests._helpers import stacked_obs
from tests.conftest import FakeHost, ServerThread
from tests.test_policy_host import ResizeStubConfig, StubConfig, StubPolicy, identity


def _good_obs(task=""):
    return {
        "observation.images.cam0": np.zeros((96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros(2, dtype=np.float32),
        "task": task,
    }


@pytest.fixture
def client(server):
    with PolicyClient("127.0.0.1", server.port) as c:
        yield c


def test_metadata_read_on_connect(client):
    assert client.metadata["policy_type"] == "fake"
    assert client.metadata["n_action_steps"] == 10


def test_infer_returns_actions(client):
    reply = client.infer(_good_obs())
    assert reply["actions"].shape == (10, 2)
    assert reply["actions"].dtype == np.float32
    assert reply["actions"].flags.writeable


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda o: o.pop("observation.state"), "missing observation key 'observation.state'"),
        (lambda o: o.update({"observation.images.cam1": o["observation.images.cam0"]}),
         "unknown observation key 'observation.images.cam1'"),
        (lambda o: o.update({"observation.state": np.zeros(3, dtype=np.float32)}),
         r"shape \[3\], expected \[2\]"),
        (lambda o: o.update({"observation.state": np.zeros(2, dtype=np.float64)}),
         "dtype float64, expected float32"),
        (lambda o: o.update({"observation.images.cam0": [[0]]}), "must be a numpy array"),
        (lambda o: o.update({"task": 3}), "'task' must be a string"),
    ],
)
def test_client_side_validation(client, mutate, match):
    obs = _good_obs()
    mutate(obs)
    with pytest.raises(ValueError, match=match):
        client.infer(obs)


def test_server_error_raises_policy_server_error(client):
    # passes client-side validation; FakeHost rejects it server-side
    with pytest.raises(PolicyServerError, match="reject me"):
        client.infer(_good_obs(task="reject me"))


def test_wrong_api_key_raises_actionable_error(fake_host):
    st = ServerThread(fake_host, api_key="sekret")
    try:
        with pytest.raises(PolicyServerError, match="rejected the api key"):
            PolicyClient("127.0.0.1", st.port, api_key="wrong")
    finally:
        st.stop()


def test_client_survives_server_restart(fake_host):
    st = ServerThread(fake_host)
    client = PolicyClient("127.0.0.1", st.port)
    assert client.infer(_good_obs())["actions"].shape == (10, 2)
    port = st.port
    st.stop()
    st2 = ServerThread(fake_host, port=port)  # same port = same address, like a real restart
    try:
        assert client.infer(_good_obs())["actions"].shape == (10, 2)
    finally:
        client.close()
        st2.stop()


def _stub_policy_host():
    return PolicyHost(StubPolicy(StubConfig()), identity, identity,
                      checkpoint="stub/ckpt", device="cpu", fps=None)


def _resize_stub_policy_host():
    return PolicyHost(StubPolicy(ResizeStubConfig()), identity, identity,
                      checkpoint="stub/resize-ckpt", device="cpu", fps=None)


def test_jpeg_end_to_end_against_real_host():
    host = _stub_policy_host()
    st = ServerThread(host)
    try:
        with PolicyClient("127.0.0.1", st.port, jpeg_quality=90) as client:
            obs = stacked_obs()
            reply = client.infer(obs)
            assert reply["actions"].shape == (8, 11)
            assert client.last_request_bytes < obs["observation.images.cam0"].nbytes / 5
            # pixel-level roundtrip through the REAL client encoder and server
            # decoder — catches an RGB/BGR swap on either side (gradient
            # frames make a swap blow the mean error far past 5)
            seen = host._policy.seen_batch["observation.images.cam0"]  # (1, 2, 3, 96, 96), float [0,1]
            decoded = seen[0].permute(0, 2, 3, 1).numpy() * 255
            assert np.mean(np.abs(decoded - obs["observation.images.cam0"].astype(float))) < 5
    finally:
        st.stop()


def test_jpeg_caller_contract_unchanged():
    # caller passes the SAME raw arrays as in raw mode; validation still runs
    st = ServerThread(_stub_policy_host())
    try:
        with PolicyClient("127.0.0.1", st.port, jpeg_quality=90) as client:
            bad = stacked_obs()
            bad["observation.state"] = np.zeros((2, 3), dtype=np.float32)
            with pytest.raises(ValueError, match=r"shape \[2, 3\], expected \[2, 2\]"):
                client.infer(bad)
    finally:
        st.stop()


def test_jpeg_refused_when_server_is_raw_only(fake_host, server):
    # FakeHost metadata has no "image_encodings" -> treat as raw-only
    with pytest.raises(ValueError, match="does not support jpeg"):
        PolicyClient("127.0.0.1", server.port, jpeg_quality=90)


@pytest.mark.parametrize("quality", [0, 101, "high", True])
def test_jpeg_quality_validated(quality):
    with pytest.raises(ValueError, match="jpeg_quality must be an integer between 1 and 100"):
        PolicyClient.__new__(PolicyClient)._check_jpeg_quality(quality)


def test_resize_shrinks_and_infers():
    # ResizeStubConfig advertises resize_to=[48, 48]; caller keeps passing
    # full (2, 96, 96, 3) frames per the unchanged caller contract, the
    # client downscales+jpeg-encodes before send.
    host = _resize_stub_policy_host()
    st = ServerThread(host)
    try:
        with PolicyClient("127.0.0.1", st.port, resize=True, jpeg_quality=90) as client:
            obs = stacked_obs()
            reply = client.infer(obs)
            assert reply["actions"].shape == (8, 11)
            assert client.last_request_bytes < obs["observation.images.cam0"].nbytes / 10
    finally:
        st.stop()


def test_resize_refused_when_unsupported():
    # plain StubConfig has no resize_shape -> no resize_to advertised
    st = ServerThread(_stub_policy_host())
    try:
        with pytest.raises(ValueError, match="does not advertise resize"):
            PolicyClient("127.0.0.1", st.port, resize=True)
    finally:
        st.stop()


def test_resize_caller_contract_unchanged():
    # a wrong full-frame shape must still raise the same validation error,
    # BEFORE any resize is attempted
    st = ServerThread(_resize_stub_policy_host())
    try:
        with PolicyClient("127.0.0.1", st.port, resize=True) as client:
            bad = stacked_obs()
            bad["observation.state"] = np.zeros((2, 3), dtype=np.float32)
            with pytest.raises(ValueError, match=r"shape \[2, 3\], expected \[2, 2\]"):
                client.infer(bad)
    finally:
        st.stop()
