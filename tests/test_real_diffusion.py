import numpy as np
import pytest
import torch

pytestmark = pytest.mark.checkpoint

CKPT = "SteveNguyen/test_pick_mustard_200_diffusion-best"


@pytest.fixture(scope="module")
def diffusion_host():
    from policy_host import load_policy_host

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_policy_host(CKPT, device, None)


def _synthetic_obs(meta, rng):
    obs = {"task": ""}
    for key, spec in meta["observations"].items():
        if spec["dtype"] == "uint8":
            obs[key] = rng.integers(0, 256, size=spec["wire_shape"], dtype=np.uint8)
        else:
            obs[key] = rng.standard_normal(spec["wire_shape"]).astype(np.float32)
    return obs


def test_metadata_reflects_checkpoint(diffusion_host):
    meta = diffusion_host.metadata
    assert meta["policy_type"] == "diffusion"
    assert meta["n_obs_steps"] == 2
    cam = meta["observations"]["observation.images.cam0"]
    assert cam["kind"] == "image" and cam["dtype"] == "uint8"
    assert cam["wire_shape"][0] == 2  # stacked frame pair
    # fps is best-effort: this checkpoint repo has no train_config.json, so the
    # lookup chain correctly yields None (the key must still always be present)
    assert "fps" in meta
    assert meta["fps"] is None or isinstance(meta["fps"], float)


def test_infer_shapes(diffusion_host):
    meta = diffusion_host.metadata
    reply = diffusion_host.infer(_synthetic_obs(meta, np.random.default_rng(0)))
    assert reply["actions"].shape == (meta["n_action_steps"], meta["action_dim"])
    assert reply["actions"].dtype == np.float32
    assert np.isfinite(reply["actions"]).all()


def test_offline_chunk_matches_select_action_queue_path(diffusion_host):
    """The served path (reset + predict_action_chunk on a time-stacked batch,
    offline mode) must equal lerobot's own select_action queue path — same
    frame pair, same noise. This is the correctness proof for decision 2."""
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import ACTION

    policy = diffusion_host._policy
    pre = diffusion_host._pre
    device = torch.device(diffusion_host._device)
    meta = diffusion_host.metadata
    rng = np.random.default_rng(1)

    obs = _synthetic_obs(meta, rng)
    frames = diffusion_host._split_frames(obs)
    processed = [pre(prepare_observation_for_inference(f, device, task="")) for f in frames]

    action_dim = meta["action_dim"]
    horizon = policy.config.horizon
    noise = torch.randn(1, horizon, action_dim, generator=torch.Generator().manual_seed(0)).to(device)

    # lerobot's own path: two select_action calls (the queue-based eval pattern)
    policy.reset()
    _ = policy.select_action(dict(processed[0]), noise=noise)  # junk chunk from duplicated frame
    policy._queues[ACTION].clear()                             # flush it
    first = policy.select_action(dict(processed[1]), noise=noise)
    rest = [policy._queues[ACTION].popleft() for _ in range(len(policy._queues[ACTION]))]
    queue_chunk = torch.stack([first, *rest], dim=1)           # (1, n_action_steps, dim)

    # ficelle's served path: offline mode on a time-stacked batch
    policy.reset()
    stacked = dict(processed[-1])
    for key in meta["observations"]:
        stacked[key] = torch.stack([processed[0][key], processed[1][key]], dim=1)
    offline_chunk = policy.predict_action_chunk(stacked, noise=noise)

    torch.testing.assert_close(offline_chunk, queue_chunk, rtol=1e-4, atol=1e-5)


def test_end_to_end_over_websocket(diffusion_host):
    import websockets.sync.client
    from ficelle_client import PolicyClient, msgpack_numpy
    from tests.conftest import ServerThread

    st = ServerThread(diffusion_host)
    try:
        with PolicyClient("127.0.0.1", st.port) as client:
            meta = client.metadata
            reply = client.infer(_synthetic_obs(meta, np.random.default_rng(2)))
            assert reply["actions"].shape == (meta["n_action_steps"], meta["action_dim"])
            assert reply["server_timing"]["infer_ms"] > 0
            # acceptance test 3: wrong shape -> one-line actionable error, not a stack trace
            bad = _synthetic_obs(meta, np.random.default_rng(3))
            bad["observation.state"] = bad["observation.state"][:1]
            with pytest.raises(ValueError, match="expected"):
                client.infer(bad)
            # acceptance test 3, server side: bypass the client's pre-validation with
            # a raw websocket so the REAL server must produce the one-line error itself
            with websockets.sync.client.connect(
                f"ws://127.0.0.1:{st.port}", compression=None, max_size=None
            ) as ws:
                ws.recv()  # metadata frame
                bad = _synthetic_obs(meta, np.random.default_rng(4))
                bad["observation.state"] = bad["observation.state"][:1]
                ws.send(msgpack_numpy.packb(bad))
                reply = ws.recv()
                assert isinstance(reply, str)
                assert reply.startswith("ficelle: ")
                assert "\n" not in reply.strip()
                assert "observation.state" in reply
    finally:
        st.stop()
