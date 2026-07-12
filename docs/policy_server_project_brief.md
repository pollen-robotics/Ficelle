# Project brief: Ficelle — generic LeRobot policy server ("serve any policy, anywhere")

*Ficelle: the puppeteer's string — the thin link that carries motion from
the hand that decides to the puppet that moves. Name verified available on
PyPI and unclaimed in the GitHub tooling namespace (2026-07-12).*

*Handoff brief for a dedicated development session. Written 2026-07-12 from
the Grabette project, where all context below was established. Companion
document: `docs/policy_serving_survey.md` (the tooling survey this design is
based on — read it first).*

## 1. Why this exists

We need to run large policies (a 3B pi0-FAST fine-tune, later possibly
π0.5-class models) for a real-robot control loop when the robot's computer
(laptop, RTX 4070 8 GB) cannot run them at useful speed, but a remote RTX
5090 (different network, behind NAT) can. The policy call must cross the
internet with low, predictable latency.

The 2026-07 tooling survey found a genuine ecosystem gap:

- **LeRobot's own async-inference stack** rejects `pi0fast` (hardcoded
  `SUPPORTED_POLICIES`), couples its client to lerobot `Robot` hardware
  classes (unusable from a custom eval loop), and pickles uncompressed
  full-resolution frames over unauthenticated gRPC — with a published
  unauthenticated-RCE CVE (CVE-2026-25874).
- **openpi's server** (Physical Intelligence) has the *right* protocol
  (websocket + msgpack_numpy, dependency-light client) but can only load
  openpi-format checkpoints.
- Every other VLA project hand-rolls the same ~100-line server
  (GR00T: ZeroMQ+msgpack; OpenVLA: FastAPI+json-numpy).

**The product**: a small, clean, generic tool that serves ANY LeRobot policy
checkpoint (diffusion, ACT, pi0, pi0fast, smolvla, …) over the
openpi-style protocol, with a self-describing metadata contract, callable
from any Python client that has observations as numpy arrays. Done well,
this is publishable open source and a credible upstream contribution.

## 2. Design decisions (already made — don't relitigate without new facts)

1. **Protocol: websocket + msgpack_numpy** (openpi-compatible in spirit).
   One request per action chunk: client sends an observation dict
   (numpy images + state + task string), server replies
   `{"actions": np.ndarray[chunk, action_dim]}`. Binary msgpack, compression
   off (images should be resized client-side; optionally JPEG-encoded).
   Rationale: transport overhead is noise vs a 3B forward pass; every
   alternative (gRPC schemas, Triton, Ray) buys nothing measurable and costs
   integration effort. **Never pickle** (that's how lerobot got its CVE).
2. **Self-describing metadata call** (our addition over openpi): on connect
   (or via a `{"type": "metadata"}` request) the server reports: model repo
   id, policy type, expected observation keys and shapes/dtypes, action dim,
   chunk size, n_action_steps, training fps, and a protocol version.
   Client validates before the first inference; mismatches produce a clear
   error naming the offending key. (Grabette rule: every failure must be
   self-explaining — users of this tool won't have a debugging assistant.)
3. **Preprocessing runs server-side** using the checkpoint's own lerobot
   pipeline (`make_pre_post_processors`) — the client sends RAW observations
   (uint8 HWC images at the policy's input resolution, float state, task
   string). This is the single most important correctness decision: the
   server is the only place that can guarantee train/inference processing
   parity. Client-side duties are limited to resize (to the resolution the
   metadata announces) and optional JPEG encoding.
4. **Network layer: out of scope — assume a flat network.** Tailscale (or
   headscale/NetBird for full-FOSS) gives robot↔server a direct WireGuard
   path through NAT; the server binds to the tailnet IP. Document this;
   don't build VPN logic. Support an optional `api_key` (openpi has one) as
   defense in depth.
5. **Latency handling is the CLIENT's business, kept simple**: execute the
   received chunk open-loop, request the next chunk a few actions before
   exhaustion, optionally blend overlapping actions (weighted average,
   ~15 lines — pattern from lerobot's `aggregate_fn`). Real-Time Chunking
   (RTC) does NOT apply to autoregressive policies (pi0fast); it becomes
   relevant only for flow-based policies (pi0/pi0.5/smolvla) — design the
   client so a smarter scheduler can be swapped in later, don't build it now.

## 3. Deliverables (v0 scope)

```
ficelle/
  pyproject.toml                # uv project; deps: lerobot (pinned), websockets, msgpack, numpy
  serve.py                      # the server CLI (~150 lines)
  client/                       # pip-installable, dependency-light (NO torch/lerobot)
    policy_client.py            # PolicyClient(host, port).infer(obs) -> dict; .metadata
    msgpack_numpy.py            # vendored (pattern: openpi packages/openpi-client)
  examples/
    random_client.py            # sends synthetic obs, prints latency stats
  README.md                     # non-expert grade: setup incl. Tailscale, troubleshooting table
```

Server CLI sketch:
```bash
uv run python serve.py --checkpoint SteveNguyen/grabette_pick3_pi0fast \
    --host 100.x.y.z --port 8000 [--api_key ...] [--device cuda] [--dtype bfloat16]
```

v0 acceptance test (definition of "works properly"):
1. Serve `SteveNguyen/test_pick_mustard_200_diffusion-best` (small, fast) and
   `SteveNguyen/grabette_pick3_pi0fast` (3B; on the 5090).
2. `random_client.py` from another machine over Tailscale: correct action
   shapes, p50/p95 round-trip latency printed; diffusion ≲150 ms, pi0fast
   ≲700 ms per chunk over internet (5090).
3. Wrong obs key / wrong shape / wrong api_key each produce a one-line
   actionable error, not a stack trace.
4. Server survives client disconnect/reconnect; client survives server
   restart (auto-retry like openpi's).

## 4. Integration contract with Grabette (the first real consumer)

Grabette's eval loop is
`grabette/integrations/openarm/openarm_gripette_simu/examples/evaluate.py`,
run in SYNC mode (`--n_action_steps 4` for diffusion; pi0fast chunk 10).
Integration = a `--policy_addr host:port` mode where the local
`select_action` path is replaced at CHUNK granularity by `client.infer()`.

What the client must send per inference (from the eval loop's existing data):
- `observation.images.cam0`: uint8 HWC RGB frame (the camera stream already
  provides this; resize client-side to metadata resolution)
- `observation.state`: float32 gripper joints `[proximal, distal]` (2,)
- `task`: string (multi-task pi0fast: "pick up the mustard bottle" /
  "pick up the red can" / "pick up the cup")

Policy-specific caveats the server must absorb (all verified this week):
- **Diffusion (grabette recipe)**: `n_obs_steps=2` — the policy conditions
  on an observation PAIR. The lerobot queue-based API handles this via two
  `select_action` calls (see `run_episode_async.make_batch` two-call pattern
  in evaluate.py). Server-side, either accept a 2-frame observation or
  maintain per-connection observation history. Decide explicitly; document.
- **pi0fast**: single obs; `empty_cameras=2` slots are zero-filled by the
  processor pipeline (do NOT require the client to send them); load with
  `PreTrainedConfig.from_pretrained` then `policy.to(bfloat16)` (CPU-load
  first if memory-tight); the checkpoint's own processor pipeline loads fine
  for inference (unlike training, where it had to be rebuilt — see
  `integrations/Pi0Fast/train_pi0fast.py` docstring for that story).

## 5. Hard-won context the new session should not rediscover

- **lerobot version pinning discipline**: the pi0fast checkpoint
  (`SteveNguyen/grabette_pick3_pi0fast`) was trained on lerobot main at
  commit `e40b58a8dfa9e7b86918c374791599d070518d11` (pinned in
  `grabette/integrations/Pi0Fast/pyproject.toml`). The SERVER must use a
  compatible revision — train/infer version skew is how pi0fast generation
  breaks (double-<bos> saga; see `docs/policy_comparison_gripette.md` and
  the pi0fast-minimal-repro project). The client, by design, has no lerobot
  dependency at all.
- **The FAST action tokenizer** for the pi0fast model is
  `SteveNguyen/fast_tokenizer_grabette_pick3` (fit on our data); it loads via
  the checkpoint config. The original `physical-intelligence/fast` repo does
  NOT load on transformers v5 (Hub packaging bug) — if a tokenizer load
  error appears, that's the first suspect.
- **Inference speed expectations** (for latency budgets): pi0fast 3B
  autoregressive ≈ 0.4–0.5 s/chunk on a 5090, ~1.5–2 s on a laptop 4070 —
  memory-bandwidth-bound decode (~7 GB weights read per token). Diffusion
  (~100M) ≈ 60 ms/chunk on the 4070. Prefill note: 2 of 3 camera slots are
  zero-padding (empty_cameras=2) — a future training without them would cut
  pi0fast prefill ~3×; not this project's concern but explains the numbers.
- **Working style** (owner: Steve, robotics engineer at Pollen): uv for
  everything (`uv run`, pyproject per project, `[tool.uv] package=false` for
  scripts-only dirs); plan and align before implementing; simplicity first —
  no speculative abstractions; READMEs written for non-experts with
  troubleshooting tables (users won't have a coding assistant); every
  failure guarded or self-explaining; he handles git commits himself.
- **Reference patterns to copy**: openpi
  `packages/openpi-client/src/openpi_client/websocket_client_policy.py` and
  `.../base_policy.py`, plus their `serve_policy.py` /
  `websocket_policy_server.py` (github.com/Physical-Intelligence/openpi) —
  clean, minimal, proven. Grabette's `integrations/Pi0Fast/smoke_generation.py`
  shows exactly how to load + run inference on the pi0fast checkpoint.

## 6. Milestones

1. **Day 1**: server + client + random_client latency test on localhost with
   the diffusion checkpoint. Metadata contract working.
2. **Day 2**: pi0fast on the 5090, Tailscale link from the laptop,
   acceptance tests 1–4 green.
3. **Then, back in the Grabette session**: `--policy_addr` mode in
   evaluate.py (grabette-side change, ~50 lines) and the first remote-served
   arm run.
4. **Later / stretch**: JPEG image encoding flag, overlap-blending client
   scheduler, README polish for open-sourcing, upstream conversation with
   the LeRobot team (they know our contact — the pi0fast double-bos report
   came from us).
