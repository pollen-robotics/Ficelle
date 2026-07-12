import cv2
import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from errors import ObservationError
from policy_host import PolicyHost


class StubConfig:
    """Just the attributes PolicyHost reads. Mimics a diffusion checkpoint:
    n_obs_steps=2, no chunk_size attribute (falls back to n_action_steps)."""

    type = "stub"
    n_obs_steps = 2
    n_action_steps = 8
    input_features = {
        "observation.images.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(11,))}


class StubPolicy:
    def __init__(self, config):
        self.config = config
        self.reset_calls = 0
        self.seen_batch = None

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch):
        self.seen_batch = batch
        return torch.zeros(1, self.config.n_action_steps,
                           self.config.output_features["action"].shape[0])


def identity(x):
    return x


@pytest.fixture
def host():
    return PolicyHost(StubPolicy(StubConfig()), identity, identity,
                      checkpoint="stub/ckpt", device="cpu", fps=None)


def _good_obs():
    return {
        "observation.images.cam0": np.zeros((2, 96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros((2, 2), dtype=np.float32),
        "task": "pick",
    }


def _jpeg(frame_rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def test_metadata_advertises_encodings(host):
    assert host.metadata["image_encodings"] == ["raw", "jpeg"]


def test_infer_accepts_jpeg_frame_list(host):
    obs = _good_obs()
    raw = np.random.default_rng(0).integers(0, 256, size=(2, 96, 96, 3), dtype=np.uint8)
    obs["observation.images.cam0"] = [_jpeg(raw[0]), _jpeg(raw[1])]
    reply = host.infer(obs)
    assert reply["actions"].shape == (8, 11)
    seen = host._policy.seen_batch["observation.images.cam0"]
    assert tuple(seen.shape) == (1, 2, 3, 96, 96)  # decoded to the raw layout
    # decoded pixels approximate the originals (JPEG lossy, but same image;
    # catches RGB/BGR swaps, which produce a huge channel-wise error)
    decoded = (seen[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    assert np.mean(np.abs(decoded.astype(int) - raw.astype(int))) < 8


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (lambda j: [j], "got 1 JPEG frames, expected 2"),
        (lambda j: [j, "not-bytes"], "expected JPEG bytes, got str"),
        (lambda j: [j, b"garbagegarbage"], "JPEG decode failed"),
    ],
)
def test_jpeg_error_messages(host, frames, match):
    obs = _good_obs()
    j = _jpeg(np.zeros((96, 96, 3), dtype=np.uint8))
    obs["observation.images.cam0"] = frames(j)
    with pytest.raises(ObservationError, match=match) as excinfo:
        host.infer(obs)
    assert "\n" not in str(excinfo.value)


def test_jpeg_wrong_decoded_size_hits_shape_validation(host):
    obs = _good_obs()
    small = np.zeros((48, 48, 3), dtype=np.uint8)
    obs["observation.images.cam0"] = [_jpeg(small), _jpeg(small)]
    with pytest.raises(ObservationError, match=r"shape \[2, 48, 48, 3\], expected \[2, 96, 96, 3\]"):
        host.infer(obs)


def test_metadata(host):
    m = host.metadata
    assert m["protocol"] == "ficelle/1"
    assert m["policy_type"] == "stub"
    assert m["n_obs_steps"] == 2
    assert m["n_action_steps"] == 8
    assert m["chunk_size"] == 8  # fallback when config has no chunk_size
    assert m["action_dim"] == 11
    cam = m["observations"]["observation.images.cam0"]
    assert cam == {"kind": "image", "dtype": "uint8",
                   "frame_shape": [96, 96, 3], "wire_shape": [2, 96, 96, 3]}
    state = m["observations"]["observation.state"]
    assert state == {"kind": "state", "dtype": "float32",
                     "frame_shape": [2], "wire_shape": [2, 2]}


def test_infer_stacks_frames_and_returns_chunk(host):
    reply = host.infer(_good_obs())
    assert reply["actions"].shape == (8, 11)
    assert reply["actions"].dtype == np.float32
    assert reply["server_timing"]["infer_ms"] >= 0
    policy = host._policy
    assert policy.reset_calls == 1  # stateless per request
    # time-stacked offline-mode batch: (B, n_obs_steps, ...)
    assert tuple(policy.seen_batch["observation.images.cam0"].shape) == (1, 2, 3, 96, 96)
    assert tuple(policy.seen_batch["observation.state"].shape) == (1, 2, 2)
    # images converted uint8 HWC -> float CHW in [0, 1]
    assert policy.seen_batch["observation.images.cam0"].dtype == torch.float32


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda o: o.pop("observation.state"), "missing observation key 'observation.state'"),
        (lambda o: o.update({"observation.extra": np.zeros(2, np.float32)}),
         "unknown observation key 'observation.extra'"),
        (lambda o: o.update({"observation.state": np.zeros((2, 3), np.float32)}),
         r"shape \[2, 3\], expected \[2, 2\]"),
        (lambda o: o.update({"observation.images.cam0": np.zeros((96, 96, 3), np.uint8)}),
         "2 frames stacked oldest-first"),
        (lambda o: o.update({"observation.state": np.zeros((2, 2), np.float64)}),
         "dtype float64, expected float32"),
        (lambda o: o.update({"observation.state": [[0.0, 0.0]] * 2}), "must be a numpy array"),
        (lambda o: o.update({"task": 42}), "'task' must be a string"),
    ],
)
def test_validation_one_line_errors(host, mutate, match):
    obs = _good_obs()
    mutate(obs)
    with pytest.raises(ObservationError, match=match) as excinfo:
        host.infer(obs)
    assert "\n" not in str(excinfo.value)


def test_single_obs_policy_with_empty_cameras():
    """pi0fast-shaped stub: n_obs_steps=1, one real camera + one empty_camera slot.
    The empty slot is hidden from clients and zero-filled server-side."""

    class Pi0ishConfig(StubConfig):
        n_obs_steps = 1
        chunk_size = 10
        n_action_steps = 10
        input_features = {
            "observation.images.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.images.empty_camera_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }

    host = PolicyHost(StubPolicy(Pi0ishConfig()), identity, identity,
                      checkpoint="stub/pi0ish", device="cpu", fps=30.0)
    assert "observation.images.empty_camera_0" not in host.metadata["observations"]
    assert host.metadata["chunk_size"] == 10
    assert host.metadata["fps"] == 30.0
    obs = {
        "observation.images.cam0": np.zeros((96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros(2, dtype=np.float32),
        "task": "pick",
    }
    reply = host.infer(obs)
    assert reply["actions"].shape == (10, 11)
    seen = host._policy.seen_batch
    # single-frame batch, no time axis; empty camera zero-filled
    assert tuple(seen["observation.images.cam0"].shape) == (1, 3, 96, 96)
    assert tuple(seen["observation.images.empty_camera_0"].shape) == (1, 3, 96, 96)
    assert torch.all(seen["observation.images.empty_camera_0"] == 0)
    assert seen["task"] == "pick"
