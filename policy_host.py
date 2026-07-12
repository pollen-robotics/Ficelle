"""Everything that touches lerobot: checkpoint loading, the metadata contract,
and turning raw numpy observations into action chunks.

Preprocessing runs HERE, through the checkpoint's own serialized processor
pipeline — the server is the only place that can guarantee train/inference
processing parity (see docs/policy_server_project_brief.md §2.3).
"""

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION

from errors import ObservationError

logger = logging.getLogger("ficelle.policy")

PROTOCOL = "ficelle/1"


def load_policy_host(checkpoint: str, device: str, dtype: str | None) -> "PolicyHost":
    """Load ANY LeRobot policy checkpoint (hub repo id or local dir) for serving.

    CPU-first load, optional dtype cast, then move to device — the memory-tight
    pattern proven on the 3B pi0fast (Grabette's smoke_generation.py).
    """
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    if dtype is not None:
        policy = policy.to(dtype=getattr(torch, dtype))
    policy.eval()
    policy = policy.to(device)
    policy.config.device = device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return PolicyHost(policy, preprocessor, postprocessor, checkpoint=checkpoint,
                      device=device, fps=read_training_fps(checkpoint))


def read_training_fps(checkpoint: str) -> float | None:
    """Best-effort: fps is not in the policy config — it lives in the training
    dataset's info.json, reachable via the checkpoint's train_config.json."""
    try:
        ckpt_dir = Path(checkpoint)
        if ckpt_dir.is_dir():
            train_cfg = json.loads((ckpt_dir / "train_config.json").read_text())
        else:
            train_cfg = json.loads(Path(hf_hub_download(checkpoint, "train_config.json")).read_text())
        repo_id = train_cfg["dataset"]["repo_id"]
        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        return float(json.loads(Path(info_path).read_text())["fps"])
    except Exception as e:  # noqa: BLE001 — metadata nicety, never fatal
        logger.warning(f"could not determine training fps ({e}); metadata will report fps=None")
        return None


class PolicyHost:
    """Wraps a loaded policy + processor pipelines behind the two-member
    contract the websocket transport needs: .metadata and .infer().

    Each request is STATELESS: reset() + predict_action_chunk(). For
    n_obs_steps > 1 the client sends the frame history itself (temporally
    adjacent frames — see docs/plans/2026-07-12-ficelle-v0.md decision 1) and
    we feed diffusion's offline mode with a time-stacked batch.

    Known limit: a diffusion checkpoint with n_obs_steps == 1 would trip an
    upstream shape assert in offline mode (state would need a time axis);
    no such checkpoint is in scope for v0. Likewise, a policy combining empty_camera_*
    slots with n_obs_steps > 1 would carry only the last frame's zero-fill into the
    stacked batch; no such checkpoint is in scope for v0.
    """

    def __init__(self, policy, preprocessor, postprocessor, checkpoint: str,
                 device: str, fps: float | None):
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._device = device
        # one warm model, requests serialized: reset()+predict must not interleave
        self._lock = threading.Lock()
        cfg = policy.config
        self._n_obs_steps = int(getattr(cfg, "n_obs_steps", 1))
        self._n_action_steps = int(cfg.n_action_steps)
        # pi0fast injects zero-filled empty_camera_* slots into input_features;
        # they are our job, not the client's (uint8 zeros -> -1 after SigLIP
        # rescale = training's padding value)
        self._empty_camera_keys = [k for k in cfg.input_features if "empty_camera" in k]
        self._obs_spec = self._build_obs_spec(cfg)
        self.metadata = {
            "protocol": PROTOCOL,
            "checkpoint": checkpoint,
            "policy_type": cfg.type,
            "device": device,
            "n_obs_steps": self._n_obs_steps,
            "n_action_steps": self._n_action_steps,
            "chunk_size": int(getattr(cfg, "chunk_size", 0)) or self._n_action_steps,
            "action_dim": int(cfg.output_features[ACTION].shape[0]),
            "fps": fps,
            "image_encodings": ["raw", "jpeg"],
            "observations": self._obs_spec,
        }

    def _build_obs_spec(self, cfg) -> dict:
        """The client-facing contract: images as uint8 HWC, states as float32."""
        spec = {}
        for name, feat in cfg.input_features.items():
            if name in self._empty_camera_keys:
                continue
            if feat.type is FeatureType.VISUAL:
                c, h, w = feat.shape
                spec[name] = {"kind": "image", "dtype": "uint8", "frame_shape": [h, w, c]}
            else:
                spec[name] = {"kind": "state", "dtype": "float32", "frame_shape": list(feat.shape)}
        for s in spec.values():
            s["wire_shape"] = ([self._n_obs_steps] + s["frame_shape"]
                               if self._n_obs_steps > 1 else list(s["frame_shape"]))
        return spec

    def infer(self, obs: dict) -> dict:
        obs = self._decode_images(obs)
        self._validate(obs)
        task = obs.get("task", "")
        frames = self._split_frames(obs)
        with self._lock:
            processed = []
            for frame in frames:
                # lerobot's own helper: uint8 HWC -> float32 CHW in [0,1],
                # add batch dim, move to device, attach task
                prepared = prepare_observation_for_inference(
                    frame, torch.device(self._device), task=task
                )
                processed.append(self._pre(prepared))
            if self._n_obs_steps == 1:
                batch = processed[0]
            else:
                # time-stack per-frame preprocessed tensors to (B, n_obs_steps, ...)
                # for predict_action_chunk's offline mode (queues empty after reset)
                batch = dict(processed[-1])
                for key in self._obs_spec:
                    batch[key] = torch.stack([p[key] for p in processed], dim=1)
            t0 = time.perf_counter()
            with torch.no_grad():
                self._policy.reset()
                actions = self._policy.predict_action_chunk(batch)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if actions.ndim != 3:
                actions = actions.unsqueeze(0)
            # postprocessor (unnormalize -> cpu) expects (B, action_dim): per step
            steps = [self._post(actions[:, i]) for i in range(actions.shape[1])]
        chunk = np.stack([a.squeeze(0).float().cpu().numpy() for a in steps]).astype(np.float32)
        return {"actions": chunk, "server_timing": {"infer_ms": infer_ms}}

    def _split_frames(self, obs: dict) -> list[dict]:
        """n_obs_steps single-frame numpy dicts, oldest first. Copies are
        deliberate: msgpack-decoded arrays are read-only, torch.from_numpy needs
        writable memory. Empty camera slots are zero-filled here."""
        frames = [{} for _ in range(self._n_obs_steps)]
        for key in self._obs_spec:
            arr = obs[key]
            for i, frame in enumerate(frames):
                frame[key] = np.array(arr[i] if self._n_obs_steps > 1 else arr)
        for key in self._empty_camera_keys:
            c, h, w = self._policy.config.input_features[key].shape
            for frame in frames:
                frame[key] = np.zeros((h, w, c), dtype=np.uint8)
        return frames

    def _decode_images(self, obs: dict) -> dict:
        """Transport-level JPEG support: an image key may arrive as a list of
        per-frame JPEG bytes (oldest first) instead of a raw uint8 array.
        Decode back to the raw layout here; the normal validation and the
        checkpoint's own pipeline then run unchanged — JPEG is lossy
        compression at the transport layer, not a preprocessing change."""
        decoded = dict(obs)
        for key, spec in self._obs_spec.items():
            val = obs.get(key)
            if spec["kind"] != "image" or not isinstance(val, (list, tuple)):
                continue
            if len(val) != self._n_obs_steps:
                raise ObservationError(
                    f"'{key}': got {len(val)} JPEG frames, expected {self._n_obs_steps}"
                )
            frames = []
            for i, buf in enumerate(val):
                if not isinstance(buf, bytes):
                    raise ObservationError(
                        f"'{key}' frame {i}: expected JPEG bytes, got {type(buf).__name__}"
                    )
                img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ObservationError(f"'{key}' frame {i}: JPEG decode failed")
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            decoded[key] = np.stack(frames) if self._n_obs_steps > 1 else frames[0]
        return decoded

    def _validate(self, obs: dict) -> None:
        spec = self._obs_spec
        if "task" in obs and not isinstance(obs["task"], str):
            raise ObservationError(f"'task' must be a string, got {type(obs['task']).__name__}")
        for key in obs:
            if key != "task" and key not in spec:
                raise ObservationError(
                    f"unknown observation key '{key}' — expected {sorted(spec)} (+ optional 'task')"
                )
        for key, s in spec.items():
            if key not in obs:
                raise ObservationError(
                    f"missing observation key '{key}' — expected {sorted(spec)} (+ optional 'task')"
                )
            arr = obs[key]
            if not isinstance(arr, np.ndarray):
                raise ObservationError(f"'{key}' must be a numpy array, got {type(arr).__name__}")
            if list(arr.shape) != s["wire_shape"]:
                hint = (f" ({self._n_obs_steps} frames stacked oldest-first)"
                        if self._n_obs_steps > 1 else "")
                raise ObservationError(
                    f"'{key}' has shape {list(arr.shape)}, expected {s['wire_shape']}{hint}"
                )
            if str(arr.dtype) != s["dtype"]:
                raise ObservationError(f"'{key}' has dtype {arr.dtype}, expected {s['dtype']}")
