import numpy as np
import pytest

from ficelle_client import PolicyClient, PolicyServerError
from tests.conftest import FakeHost, ServerThread


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
