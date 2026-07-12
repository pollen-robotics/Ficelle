# Ficelle JPEG Transport Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Optional client-side JPEG encoding of image observations — ~15× smaller requests (4 MiB → ~250 KB for the 2×720×960 simu checkpoint) — with zero change to how callers build observations and zero change to server-side preprocessing (parity preserved: JPEG is lossy compression at the transport layer only).

**Architecture:** `PolicyClient(..., jpeg_quality=N)` validates the caller's raw observation as today, then encodes each image key to a list of per-frame JPEG `bytes` before packing. `PolicyHost.infer` sniffs image values: ndarray → unchanged raw path; list of bytes → decode via cv2 back to the raw layout, then the existing validation + pipeline run untouched. Metadata advertises `"image_encodings": ["raw", "jpeg"]` so clients fail fast against non-JPEG servers.

**Tech Stack:** cv2 server-side (already present via lerobot). Client: soft-import cv2, fallback PIL, actionable error if neither — base deps stay exactly `websockets, msgpack, numpy`.

## Global Constraints

- Client package keeps exactly three hard dependencies. JPEG needs cv2 or PIL via soft-import chosen once at `PolicyClient.__init__` (fail fast, one-line error naming both install options).
- Color convention: observations are RGB. cv2 encodes/decodes BGR — every cv2 encode is preceded by `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` and every decode followed by `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. PIL is RGB-native (no conversion). A color-swap bug here silently degrades policies — the e2e test must assert approximate pixel equality after a roundtrip, not just shapes.
- Wire compatibility: raw ndarray observations must work exactly as before (all existing tests keep passing unmodified). JPEG frames per image key = list of `n_obs_steps` bytes objects (list of 1 when n_obs_steps == 1), oldest first.
- One-line actionable `ObservationError`s for: wrong frame count, non-bytes frame, undecodable bytes, decoded-dimension mismatch (the existing shape check covers this after decode).
- Every failure client-side (jpeg requested, server raw-only; no codec installed; bad quality value) is a one-line ValueError before any network send.
- No commits in the GRABETTE repo (working-tree edit only). ficelle commits on branch `v0-prototype`.

## File structure

- Modify: `policy_host.py` (decode + metadata), `client/ficelle_client/policy_client.py` (encode), `examples/random_client.py` (`--jpeg_quality`, payload-size print), `README.md` (short section), `tests/test_policy_host.py`, `tests/test_policy_client.py`
- Modify (GRABETTE, uncommitted): `.../openarm_gripette_simu/examples/evaluate.py` (`--jpeg_quality` passthrough)

---

### Task 1: Server decode path + metadata advertisement

**Files:** Modify `policy_host.py`; Test additions in `tests/test_policy_host.py`.

**Interfaces:**
- Produces: metadata gains top-level `"image_encodings": ["raw", "jpeg"]`. `PolicyHost.infer(obs)` accepts, for each image key, EITHER the raw uint8 ndarray (unchanged) OR a list of `n_obs_steps` JPEG bytes. Task 2's client and Task 3's probe rely on exactly this.

- [ ] **Step 1: Write the failing tests** (extend `tests/test_policy_host.py`; reuse `StubConfig`/`StubPolicy`/`identity`/`host` fixture and `_good_obs` as-is)

```python
import cv2


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
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_policy_host.py -v`; expected: new tests FAIL (`KeyError: 'image_encodings'`, ObservationError about list not ndarray, etc.), old tests pass.

- [ ] **Step 3: Implement in `policy_host.py`**

Add `import cv2` to the imports. In `PolicyHost.__init__`, add to the metadata dict (after `"observations"`): `"image_encodings": ["raw", "jpeg"],`. At the top of `infer()`, before `self._validate(obs)`, insert `obs = self._decode_images(obs)` and add the method:

```python
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
```

- [ ] **Step 4: Run to verify green** — `uv run pytest tests/test_policy_host.py -v` then full fast suite `uv run pytest`; all pass, pristine.
- [ ] **Step 5: Verify no commit needed to FakeHost** — transport/client tests still pass untouched (raw path unchanged). Leave changes uncommitted for review, then commit on approval per SDD flow.

---

### Task 2: Client-side encoding

**Files:** Modify `client/ficelle_client/policy_client.py`; Test additions in `tests/test_policy_client.py`.

**Interfaces:**
- Consumes: Task 1's server decode + `"image_encodings"` metadata.
- Produces: `PolicyClient(host, port=8000, api_key=None, connect_retry_s=5.0, jpeg_quality=None)`. When `jpeg_quality` is an int 1–100: caller-facing obs contract UNCHANGED (raw arrays validated exactly as today), then image keys are encoded to `[bytes, ...]` before packing. Also `client.last_request_bytes: int | None` (payload size of the most recent infer, for diagnostics — Task 3's probe prints it).

- [ ] **Step 1: Write the failing tests** (extend `tests/test_policy_client.py`; note these run against `FakeHost`, which is raw-only and does NOT decode — so the e2e jpeg test builds a real `PolicyHost` around `StubPolicy` from `tests/test_policy_host.py` and serves it with `ServerThread`)

```python
from policy_host import PolicyHost
from tests.test_policy_host import StubConfig, StubPolicy, identity


def _stub_policy_host():
    return PolicyHost(StubPolicy(StubConfig()), identity, identity,
                      checkpoint="stub/ckpt", device="cpu", fps=None)


def _stacked_obs():
    return {
        "observation.images.cam0": np.random.default_rng(0).integers(
            0, 256, size=(2, 96, 96, 3), dtype=np.uint8),
        "observation.state": np.zeros((2, 2), dtype=np.float32),
        "task": "pick",
    }


def test_jpeg_end_to_end_against_real_host():
    st = ServerThread(_stub_policy_host())
    try:
        with PolicyClient("127.0.0.1", st.port, jpeg_quality=90) as client:
            raw_size_estimate = _stacked_obs()["observation.images.cam0"].nbytes
            reply = client.infer(_stacked_obs())
            assert reply["actions"].shape == (8, 11)
            assert client.last_request_bytes < raw_size_estimate / 5  # big win or bust
    finally:
        st.stop()


def test_jpeg_caller_contract_unchanged():
    # caller passes the SAME raw arrays as in raw mode; validation still runs
    st = ServerThread(_stub_policy_host())
    try:
        with PolicyClient("127.0.0.1", st.port, jpeg_quality=90) as client:
            bad = _stacked_obs()
            bad["observation.state"] = np.zeros((2, 3), dtype=np.float32)
            with pytest.raises(ValueError, match=r"shape \[2, 3\], expected \[2, 2\]"):
                client.infer(bad)
    finally:
        st.stop()


def test_jpeg_refused_when_server_is_raw_only(fake_host, server):
    # FakeHost metadata has no "image_encodings" -> treat as raw-only
    with pytest.raises(ValueError, match="does not support jpeg"):
        PolicyClient("127.0.0.1", server.port, jpeg_quality=90)


@pytest.mark.parametrize("quality", [0, 101, "high"])
def test_jpeg_quality_validated(quality):
    with pytest.raises(ValueError, match="jpeg_quality must be an integer between 1 and 100"):
        PolicyClient.__new__(PolicyClient)._check_jpeg_quality(quality)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_policy_client.py -v`; new tests fail (TypeError on `jpeg_quality` kwarg), old ones pass.

- [ ] **Step 3: Implement in `policy_client.py`**

```python
    def __init__(self, host: str, port: int | None = 8000, api_key: str | None = None,
                 connect_retry_s: float = 5.0, jpeg_quality: int | None = None):
        ...existing body unchanged, plus:
        self._jpeg_quality = jpeg_quality
        self._encode_frame = None
        self.last_request_bytes: int | None = None
        if jpeg_quality is not None:
            self._check_jpeg_quality(jpeg_quality)
            self._encode_frame = self._pick_jpeg_encoder(jpeg_quality)
        self._ws, self.metadata = self._connect()
        if jpeg_quality is not None and "jpeg" not in self.metadata.get("image_encodings", ["raw"]):
            raise ValueError(
                f"server at {self._uri} does not support jpeg encoding "
                "(metadata lacks 'jpeg' in image_encodings) — upgrade the server or drop jpeg_quality"
            )

    @staticmethod
    def _check_jpeg_quality(quality) -> None:
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError(f"jpeg_quality must be an integer between 1 and 100, got {quality!r}")

    @staticmethod
    def _pick_jpeg_encoder(quality: int):
        """cv2 if available, else PIL — chosen once, fail fast with install hint."""
        try:
            import cv2

            def encode(frame):  # frame: uint8 HWC RGB
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                return buf.tobytes()

            return encode
        except ImportError:
            pass
        try:
            import io

            from PIL import Image

            def encode(frame):
                out = io.BytesIO()
                Image.fromarray(frame).save(out, format="JPEG", quality=quality)
                return out.getvalue()

            return encode
        except ImportError:
            raise ValueError(
                "jpeg_quality requires an image codec: pip install opencv-python-headless (or pillow)"
            ) from None
```

In `infer()`, after `self._validate(obs)` and before packing:

```python
        if self._encode_frame is not None:
            obs = self._encode_jpeg(obs)
        payload = self._packer.pack(obs)
        self.last_request_bytes = len(payload)
```

and the method:

```python
    def _encode_jpeg(self, obs: dict) -> dict:
        """Encode validated raw image arrays to per-frame JPEG bytes (oldest
        first). Callers keep passing raw arrays — this is transport-only."""
        encoded = dict(obs)
        for key, spec in self.metadata["observations"].items():
            if spec["kind"] != "image":
                continue
            arr = obs[key]
            frames = arr if arr.ndim == 4 else arr[None]
            encoded[key] = [self._encode_frame(np.ascontiguousarray(f)) for f in frames]
        return encoded
```

(Note `_connect` ordering: `_pick_jpeg_encoder`/`_check_jpeg_quality` run BEFORE `_connect` so a missing codec fails instantly, not after a connect wait.)

- [ ] **Step 4: Run to verify green** — `uv run pytest tests/test_policy_client.py tests/test_policy_host.py -v`, then full `uv run pytest`; pristine.
- [ ] **Step 5: Commit** both tasks' changes on `v0-prototype` after review (SDD flow).

---

### Task 3: Probe flag, README, Grabette passthrough, acceptance

**Files:** Modify `examples/random_client.py`, `README.md`; GRABETTE `evaluate.py` (uncommitted).

- [ ] **Step 1: `random_client.py`** — add `parser.add_argument("--jpeg_quality", type=int, default=None, help="encode images as JPEG at this quality (1-100) before sending")`; pass to `PolicyClient(...)`; add to the final summary line: `f" | request payload ~{client.last_request_bytes / 1024:.0f} KiB"`.
- [ ] **Step 2: README** — in section 3 (client usage), add a short "Bandwidth: JPEG encoding" subsection: what `jpeg_quality=90` does, the measured reduction, the codec soft-dependency, and that quality <85 may start affecting policy behavior (untested — measure on your task). Add a troubleshooting row for the raw-only-server error message.
- [ ] **Step 3: GRABETTE `evaluate.py`** (working-tree only): add `--jpeg_quality` arg (same help), pass to `PolicyClient(host, port, jpeg_quality=args.jpeg_quality)` in the remote branch.
- [ ] **Step 4: Acceptance (controller runs):** serve `SteveNguyen/diffusion_grabette_simu_release`; probe raw vs `--jpeg_quality 90` (expect payload 4 MiB → ~250 KB, wire+codec overhead drop); sim (headless, `PROXIMAL_CMD_SIGN=-1`) + remote eval `--num_episodes 15 --max_steps 600 --n_action_steps 8 --fps 30 --jpeg_quality 90`: success in the local-parity band (≥ ~70% at n=15), stale frames < the 5–8% seen with raw.

## Self-review

- Caller contract preserved (validation before encoding; test pins it). Raw path byte-identical (no existing test touched). Color convention double-converted on both sides with an approximate-pixel-equality e2e test. Errors all one-line. Client deps unchanged (soft imports only). GRABETTE untouched by git.
