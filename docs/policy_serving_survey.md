> Design background: a point-in-time survey (2026-07) of the VLA
> policy-serving landscape, kept for its technical value. Note: pi0-FAST,
> the driving use case throughout, is currently broken upstream in LeRobot
> (free-running generation collapses — a model/training issue, not ficelle).

# Remote VLA policy inference serving — tooling survey (2026-07)

*Research for serving a pi0-FAST fine-tune from a remote GPU (RTX 5090) to
a robot eval loop across the internet. Verified against lerobot upstream/main
source and live GitHub.*

**Scenario:** lerobot pi0-FAST fine-tune (~3B) on a single RTX 5090 box;
Python robot client on another network with observations already as numpy
arrays, needing ~10-action chunks back.

## Executive summary

- **LeRobot ships a real gRPC async-inference stack** (`src/lerobot/async_inference/`),
  with genuine chunk-overlap aggregation — but its stock server **rejects
  `pi0fast`** (not in `SUPPORTED_POLICIES`), it pickles uncompressed full-res
  frames over unauthenticated gRPC (**CVE-2026-25874, unauthenticated RCE**),
  and the client is hard-coupled to lerobot `Robot` hardware classes.
- **openpi's stack is the cleanest client-side design** (websocket +
  msgpack_numpy, `infer(obs) -> {"actions": chunk}`, 5-dependency client),
  but its server only loads openpi-format checkpoints — not a lerobot
  fine-tune. Its *protocol*, however, is trivially reimplementable
  server-side around a lerobot policy.
- **RTC does not apply to pi0-FAST.** The paper is "Real-Time Execution of
  Action Chunking Flow Policies" (Black, Galliker, Levine; arXiv:2506.07339,
  NeurIPS 2025); it is flow-matching inpainting. lerobot's `rtc_config`
  field in the pi0fast config is **dead plumbing** — zero `rtc_processor`
  call sites in `modeling_pi0_fast.py` (verified on upstream/main), vs. live
  integration in pi0/pi0.5/SmolVLA.
- **Triton/Ray Serve/LitServe: overkill, unanimously.** Every major VLA
  codebase rolls its own ~100–300-line server (openpi: websocket+msgpack;
  lerobot: gRPC; NVIDIA GR00T: ZeroMQ+msgpack; OpenVLA: FastAPI+json-numpy).
- **Tailscale (WireGuard) is the right network layer** — and effectively
  *mandatory* given both stock servers are unauthenticated/unencrypted.
- **Recommendation:** ~50-line custom websocket+msgpack server wrapping the
  lerobot `PI0FastPolicy.predict_action_chunk()`, openpi-client (or a
  30-line equivalent) on the robot side, over Tailscale with a verified
  direct connection.

## 1. LeRobot async inference (PolicyServer / RobotClient)

**What it is.** In `huggingface/lerobot` main since PR #1196 (2025-07-10,
v0.3.2), now `src/lerobot/async_inference/` (`policy_server.py`,
`robot_client.py`, `configs.py`, `helpers.py`). Docs:
https://huggingface.co/docs/lerobot/async and
https://huggingface.co/blog/async-robot-inference.

**Protocol.** gRPC (`src/lerobot/transport/services.proto`, service
`AsyncInference`) — but all payloads are opaque `bytes` containing **Python
pickle** of `TimedObservation` / `list[TimedAction]` / `RemotePolicyConfig`.
Images go **raw and uncompressed** at full camera resolution (resizing is
server-side), streamed in 2 MB chunks. No TLS, no auth
(`grpc.insecure_channel`). `grpcio` hard-pinned to 1.73.1. Launch:
`python -m lerobot.async_inference.policy_server --host=... --port=8080`.

**Chunk handling — the genuinely valuable part.** Client keeps an action
queue, requests a new chunk when `queue_size/chunk_size <=
chunk_size_threshold` (default 0.5), and merges incoming chunks with the
queued one on overlapping timesteps via a pluggable `aggregate_fn`
(`weighted_average` 0.3·old+0.7·new default, `latest_only`, `average`,
`conservative`). Chunk-boundary blending, not full temporal ensembling.

**Latency.** Good architecture (hides inference behind chunk execution), but
uncompressed full-res pickled frames are the internet bottleneck: two
640×480×3 frames ≈ 1.8 MB/obs → ~1 s on a 10–20 Mbps uplink. Fine on LAN.

**Blockers for our case.**
1. `SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet",
   "pi0", "pi05", "groot"]` (`async_inference/constants.py`) — server raises
   at handshake for `pi0fast`. One-line patch on a fork, but stock rejects it.
2. **CVE-2026-25874** (CVSS 9.3, Apr 2026, unpatched ≤0.5.1):
   `pickle.loads()` of attacker bytes over unauthenticated gRPC =
   unauthenticated RCE (https://chocapikk.com/posts/2026/lerobot-pickle-rce/).
   Never expose publicly.
3. **Custom eval loop:** `RobotClient.__init__` instantiates and connects a
   lerobot `Robot` — no numpy-observation injection point. Workarounds: a
   mock `Robot`, or talking to the `AsyncInferenceStub` directly (~40 lines),
   with the caveat that the wire format is pickled lerobot classes (client
   and server versions must match).

## 2. openpi serving stack (serve_policy.py + openpi-client)

**Confirmed:** `scripts/serve_policy.py` runs a `WebsocketPolicyServer`
(port 8000); client in `packages/openpi-client`. Docs:
https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md

**Protocol & client API.** WebSocket (`websockets.sync.client`, compression
disabled) + **msgpack with numpy support** (`msgpack_numpy`):

```python
client = WebsocketClientPolicy(host, port)   # also takes api_key
result = client.infer({"observation/image": img, "observation/state": state, "prompt": task})
chunk = result["actions"]                    # numpy action chunk
```

Auto-reconnects every 5 s. `ActionChunkBroker` replays a cached chunk and
re-queries when exhausted (synchronous, no overlap blending). Dependencies:
`dm-tree, msgpack, numpy>=1.22,<2.0, pillow, websockets>=11` — no torch/jax.
(The `numpy<2` pin is easily relaxed in a vendored copy.)

**Blocker:** the server loads openpi-format checkpoints only — cannot serve
a lerobot fine-tune. The valuable artifact is the *protocol*: a custom
server speaking it is ~50 lines, after which the polished client works
unchanged. This is what Physical Intelligence runs for DROID/ALOHA evals.

## 3. Real-Time Chunking (RTC)

**Paper:** "Real-Time Execution of Action Chunking Flow Policies" — Black,
Galliker, Levine (Physical Intelligence), arXiv:2506.07339, NeurIPS 2025.
Blog: https://www.pi.website/research/real_time_chunking

**Mechanism:** generate chunk *n+1* while executing chunk *n*; the new chunk
is treated as **inpainting** — actions that will execute during the
inference delay are frozen, the overlap is softly guided toward the previous
chunk's tail via a guidance term **in the flow-matching denoising loop**.
Inference-time only. Maintains performance under +100–300 ms injected
latency. **Requires iterative denoising — does not apply to autoregressive
pi0-FAST decoding.**

**lerobot implementation:** real (PR #1698, v0.5.0, `src/lerobot/policies/rtc/`,
docs https://huggingface.co/docs/lerobot/en/rtc) — for **pi0, pi0.5, SmolVLA
only**, via the `lerobot-rollout --inference.type=rtc` path. Verified on
upstream/main: `pi0_fast` carries the `rtc_config` field and even
instantiates an `RTCProcessor`, but there are **zero `rtc_processor.` call
sites** in `modeling_pi0_fast.py` — the field is inert for pi0fast.

**Implication:** with pi0-FAST, latency handling = chunk-overlap blending or
open-loop chunk execution with early re-query. RTC becomes available if we
ever switch to a pi0/pi0.5/SmolVLA fine-tune.

## 4. Generic serving frameworks

**Triton** (~0.1–0.4 ms overhead, but dynamic batching/model repos are dead
weight for one robot + one warm model; NVIDIA itself serves GR00T with a
hand-rolled ZeroMQ+msgpack server), **Ray Serve** (~1–2 ms, carries cluster
infrastructure for autoscaling we'll never use), **LitServe**
(JSON-over-HTTP, base64 images, zero robotics track record): all overkill.
The community norm is a small bespoke server: openpi → websocket+msgpack;
lerobot → gRPC+pickle; GR00T → ZeroMQ+msgpack (`allow_pickle=False`);
OpenVLA → FastAPI+json-numpy. Transport overhead is noise vs. a 3B forward
pass. One lesson worth importing: **serialization hygiene — msgpack/protobuf,
never pickle** (see lerobot's CVE).

## 5. Network layer

**Tailscale (WireGuard)**: NAT-to-NAT traversal (~80–94% direct, DERP relay
fallback), stable IPs + MagicDNS, crypto-only overhead on direct paths
(sub-ms; internet RTT 10–40 ms dominates). It also **closes the
unauth-RCE/no-TLS exposure of both stock policy servers** — a private
overlay is effectively mandatory, not optional. Gotcha: silent DERP relay
fallback adds a big detour — verify `tailscale ping` reports `direct` before
trusting latency. Alternatives: headscale/NetBird (self-hosted control
plane), plain WireGuard (needs one reachable endpoint); ngrok/Cloudflare
Tunnel are the wrong tool (public URL, +15–45 ms). WebRTC is standard in
commercial teleop but used by no VLA inference stack.

## Recommendation matrix

| Option | Works with our pi0fast checkpoint? | Custom-loop fit | Internet fit | Effort | Verdict |
|---|---|---|---|---|---|
| A. lerobot PolicyServer + stock RobotClient | No (rejects `pi0fast`; client needs lerobot `Robot`) | Poor | Poor (full-res pickle; RCE) | Low if it worked | Rejected as-is |
| B. lerobot PolicyServer patched + custom gRPC client | Yes (1-line fork patch) | OK (~40 lines vs the stub) | Mediocre (pickle wire format, version-locked; grpcio pin) | Medium | Fallback; buys the `aggregate_fn` machinery |
| C. openpi serve_policy + openpi-client | No (server can't load lerobot checkpoints) | Excellent client | Good | High (checkpoint conversion) | Rejected — wrong server |
| **D. Custom server speaking the openpi protocol** ⭐ | Yes — wrap `PI0FastPolicy.predict_action_chunk()` in `websockets`+`msgpack_numpy` | Excellent — `client.infer(obs)` with numpy | Good (client-side resize/JPEG; optional api_key) | ~50 lines server, ~0–30 client | **Recommended** |
| E. Triton / Ray / LitServe | Python backend only | Awkward | Fine but pointless | High | Overkill |

**Concrete plan for D:**
1. Server (5090): load the fine-tune with lerobot, run the pre/post
   processors per request, `predict_action_chunk` → numpy, reply
   `{"actions": ...}` via msgpack_numpy over `websockets` (pattern
   copy-paste from openpi's `websocket_policy_server.py`).
2. Client (robot side): vendor openpi's `websocket_client_policy.py` +
   `msgpack_numpy.py` (two small files) or install `openpi-client`. Resize
   images to policy input resolution **client-side** before sending.
3. Latency hiding without RTC: execute the 10-action chunk open-loop,
   request the next chunk 2–3 actions before exhaustion, optionally blend
   the overlap with a weighted average (reimplement lerobot's
   `weighted_average`, ~15 lines).
4. Network: Tailscale between robot machine and 5090; bind the server to
   the tailnet IP only; verify `tailscale ping` = `direct`.

## Key sources

- lerobot async inference: https://huggingface.co/docs/lerobot/async ·
  https://huggingface.co/blog/async-robot-inference · `src/lerobot/async_inference/`
- lerobot pickle RCE: https://chocapikk.com/posts/2026/lerobot-pickle-rce/ ·
  https://thehackernews.com/2026/04/critical-cve-2026-25874-leaves-hugging.html
- openpi remote inference: https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md
- RTC: https://arxiv.org/abs/2506.07339 · https://www.pi.website/research/real_time_chunking ·
  https://github.com/huggingface/lerobot/pull/1698 · https://huggingface.co/docs/lerobot/en/rtc ·
  follow-ups: https://arxiv.org/abs/2512.05964, https://arxiv.org/abs/2606.13355
- Community practice: https://github.com/NVIDIA/Isaac-GR00T ·
  https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py ·
  VLAgents survey https://arxiv.org/abs/2601.11250
- Network: https://tailscale.com/docs/reference/connection-types ·
  https://build.nvidia.com/spark/tailscale
