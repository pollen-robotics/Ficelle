# ficelle

Serve any [LeRobot](https://github.com/huggingface/lerobot) policy checkpoint
over the network to a robot, without touching pickle.

## 1. What is this

ficelle is a small server + client pair for running LeRobot policy inference
remotely: the server loads any LeRobot checkpoint (diffusion, pi0fast, ...) on
a GPU machine and serves it over a websocket; the robot-side client is a
3-dependency (`websockets`, `msgpack`, `numpy`) package with no `torch` and no
`lerobot`, so it drops into any robot control loop. The wire protocol is
copied from [openpi](https://github.com/Physical-Intelligence/openpi): on
connect the server pushes a metadata frame describing exactly what
observations it expects, then it's request (observation) / reply (action
chunk) over binary msgpack frames. Everything is serialized with msgpack —
**never pickle** (lerobot's stock policy server has a pickle deserialization
RCE, CVE-2026-25874; ficelle's vendored codec exists specifically to avoid
that class of bug).

## 2. Server setup

Runs on the GPU machine that has the checkpoint. Requires `uv`
(https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url>
cd ficelle
uv sync
```

Start the server:

```bash
uv run python serve.py --checkpoint SteveNguyen/test_pick_mustard_200_diffusion-best
```

`--checkpoint` accepts either a Hugging Face Hub repo id (downloaded and
cached automatically) or a local path to a checkpoint directory (the
`pretrained_model` folder lerobot writes during training).

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | *(required)* | HF Hub repo id or local checkpoint dir |
| `--host` | `0.0.0.0` | bind address — see [Remote access over Tailscale](#4-remote-access-over-tailscale) for why you usually want your tailnet IP here instead |
| `--port` | `8000` | listening port |
| `--api_key` | `None` | optional shared secret; clients must pass the same `--api_key` |
| `--device` | `cuda` if available, else `cpu` | inference device |
| `--dtype` | `None` (checkpoint's native dtype) | `float32` or `bfloat16` — **use `--dtype bfloat16` for pi0fast** (a 3B-parameter model; float32 weights alone are ~12 GB) |

Example for a pi0fast checkpoint:

```bash
uv run python serve.py --checkpoint SteveNguyen/grabette_pick3_pi0fast --dtype bfloat16
```

On startup the server logs four things — read them before wiring up a
client:

```
loaded diffusion from SteveNguyen/test_pick_mustard_200_diffusion-best on cuda
expected observations: observation.images.cam0 uint8[2, 96, 96, 3], observation.state float32[2, 2] (+ optional 'task')
replies: 8 x 11 actions per request | training fps: None
serving on ws://0.0.0.0:8000
```

- Line 1 confirms which checkpoint loaded and onto which device.
- Line 2 is the observation contract: the exact keys, dtypes, and shapes
  every request must match (the `[2, ...]` leading dimension here means this
  checkpoint has `n_obs_steps=2` — see [Client usage](#3-client-usage)).
- Line 3 is the reply shape (`n_action_steps x action_dim`) and the training
  fps, if it could be determined (see the `fps: None` row in
  [Troubleshooting](#7-troubleshooting)).
- Line 4 confirms the address it's actually listening on — this must be
  reachable from the robot (see [Remote access over
  Tailscale](#4-remote-access-over-tailscale)).

The server keeps one policy loaded and serves requests one at a time
(concurrent connections queue); it never asks the client to pickle anything
and never unpickles anything itself.

## 3. Client usage

Install the client (numpy + websockets + msgpack only, no GPU needed):

```bash
# once published:
pip install "ficelle-client @ git+<this-repo-url>#subdirectory=client"

# from a local checkout instead:
pip install ./client
```

(`<this-repo-url>` is a placeholder — this repository hasn't settled on its
final public URL yet.)

Minimal usage:

```python
from ficelle_client import PolicyClient

client = PolicyClient("100.x.y.z", 8000)          # construct: connects, reads metadata
print(client.metadata["observations"])            # inspect the expected obs contract
obs = {
    "observation.images.cam0": cam0_frames,        # uint8, HWC, at metadata's resolution
    "observation.state": state_frames,             # float32
    "task": "pick up the mustard bottle",          # optional, default ""
}
reply = client.infer(obs)
actions = reply["actions"]                         # (n_action_steps, action_dim) float32
```

`client.metadata["observations"]` is a dict keyed by observation name, e.g.:

```python
{
  "observation.images.cam0": {"kind": "image", "dtype": "uint8",
                               "frame_shape": [96, 96, 3],      # per-frame HWC
                               "wire_shape": [2, 96, 96, 3]},   # what you actually send
  "observation.state":       {"kind": "state", "dtype": "float32",
                               "frame_shape": [2], "wire_shape": [2, 2]},
}
```

Three rules that apply to every checkpoint:

- **Images are `uint8`, HWC, at the checkpoint's training resolution** — read
  `frame_shape` (or `wire_shape`'s trailing dims) and resize/crop
  client-side; the server does not resize for you.
- **States are `float32`.**
- **If `client.metadata["n_obs_steps"] > 1`**, every array's `wire_shape` gets
  a leading `n_obs_steps` axis: send that many *temporally adjacent* frames
  (camera-period spacing, not one frame repeated), stacked **oldest first**.
  A checkpoint with `n_obs_steps=2` above wants
  `observation.images.cam0.shape == (2, 96, 96, 3)` — frame `[0]` is the
  older one.

`client.infer()` validates your observation dict against `client.metadata`
before sending anything over the wire, so a wrong key/shape/dtype fails
immediately with a message naming the exact key (see
[Troubleshooting](#7-troubleshooting)). If the connection drops (e.g. the
server restarted), the next `infer()` call reconnects automatically.

### Bandwidth: JPEG encoding

To reduce request payload size on bandwidth-constrained links (e.g. over
Tailscale or real robots with poor connectivity), pass `jpeg_quality` to
the client:

```python
client = PolicyClient("100.x.y.z", 8000, jpeg_quality=90)
# Observations still passed as raw uint8 numpy arrays; the client
# encodes images as JPEG before sending, and the server decodes
# before its normal preprocessing pipeline — preservation parity
# is maintained for inputs downstream of image loading.
```

Measured compression on a real checkpoint (2 frames × 720 × 960 images):
raw request ~4 MiB → ~250 KB with `jpeg_quality=90` (~15× reduction). Wire +
codec overhead depend on your link and policy; measure on your task.

Quality below ~85 may start affecting policy behavior (untested — benchmark
on your validation set). The client soft-depends on `opencv-python`(-headless)
or `pillow`; if neither is installed, `client.infer()` raises a one-line error
naming the missing codec.

## 4. Remote access over Tailscale

For running the server on a GPU box and the client on a robot on a different
network, [Tailscale](https://tailscale.com/) is the simplest way to get a
private, low-latency link between them without exposing the server to the
public internet.

1. Install Tailscale on **both** the GPU machine and the robot, and join them
   to the same tailnet.
2. Bind the server to its **tailnet IP** (not `0.0.0.0`) so `--host` matches
   what `tailscale ip` reports on that machine:

   ```bash
   uv run python serve.py --checkpoint SteveNguyen/grabette_pick3_pi0fast --dtype bfloat16 --host 100.x.y.z
   ```

3. From the robot, point the client at that same tailnet IP:

   ```bash
   uv run python examples/random_client.py --host 100.x.y.z --task "pick up the mustard bottle" -n 30
   ```

4. **Before trusting the latency numbers**, verify the path is direct:

   ```bash
   tailscale ping 100.x.y.z
   ```

   It must say `pong from ... via direct`. If it instead says
   `via DERP <relay>`, Tailscale has fallen back to relaying through a
   Tailscale-operated server (e.g. due to strict NAT/firewalls on one side),
   which silently adds ~50–200 ms of round-trip latency — see the
   [Troubleshooting](#7-troubleshooting) table.

5. Tailscale already authenticates which devices can reach the server, but
   pass `--api_key <secret>` on the server (and the matching `api_key=` on
   the client) as defense in depth — it costs nothing and stops any other
   device on the tailnet from calling your policy server.

## 5. iroh transport (zero-config remote)

Tailscale ([§4](#4-remote-access-over-tailscale)) needs both machines joined
to a tailnet ahead of time. `--transport iroh` is the zero-config
alternative: no VPN, no account, no shared network to set up — the server
prints a one-line **ticket** that fully describes how to reach it, and the
client dials that ticket directly. Under the hood it's QUIC with iroh's
built-in NAT hole-punching (see `experiments/iroh_probe/` for the proof of
concept this is built on).

Start the server with `--transport iroh` instead of the default:

```bash
uv run python serve.py --checkpoint SteveNguyen/test_pick_mustard_200_diffusion-best --transport iroh
```

It logs the usual four lines, then a ticket instead of a `ws://` address:

```
============================================================
iroh ticket: endpointv1...
============================================================
```

Give that ticket to the client — `open_client` dispatches on its shape (an
iroh ticket vs. a `host:port` string), so the same call site works for
either transport:

```python
from ficelle_client import open_client

client = open_client("endpointv1...", jpeg_quality=90)  # ticket -> IrohPolicyClient
# client = open_client("100.x.y.z:8000")                # host:port -> PolicyClient (unchanged)
print(client.metadata["observations"])
reply = client.infer(obs)                                # same .infer() contract either way
```

`--policy_addr` on GRABETTE's `evaluate.py` accepts either form unchanged for
the same reason: pass a ticket or a `host:port` and it does the right thing.

Install the client's iroh extra (the plain `pip install ficelle-client` from
[§3](#3-client-usage) does not pull in `iroh`):

```bash
pip install "ficelle-client[iroh] @ git+<this-repo-url>#subdirectory=client"
# from a local checkout instead:
pip install "./client[iroh]"
```

Notes:

- **First connect may take a moment.** iroh tries a direct (hole-punched)
  path first and falls back to a relay if NAT traversal fails; the very
  first request after construction can be slower while that negotiation
  settles. Subsequent requests reuse the established connection.
- **Relays**: by default iroh uses n0's public relay infrastructure only as
  a fallback/rendezvous — pass `--relay-url <url>` on the server (and
  `relay_url=<url>` on `open_client`/`IrohPolicyClient`) to point both ends
  at a self-hosted relay instead.
- **`infer_timeout`**: `IrohPolicyClient(ticket, infer_timeout=5.0)` (default
  5s) bounds how long a single `.infer()` call waits before raising
  `PolicyServerError` — a degraded link or a stuck server otherwise hangs
  the caller forever. A timed-out request doesn't wedge the client: the next
  `.infer()` call still works.

## 6. Latency expectations

Measured on real checkpoints (server infer time is what
`reply["server_timing"]["infer_ms"]` reports; RTT is what
`examples/random_client.py` measures end-to-end, including msgpack
serialization and network):

| Checkpoint | Server GPU | Server infer / chunk | RTT (localhost) |
|---|---|---|---|
| diffusion, ~100M params | RTX 4070 (laptop) | ~52 ms | p50 ~68 ms (includes ~16 ms serialization) |
| pi0fast, 3B params (`--dtype bfloat16`) | RTX 5090 | ~0.4–0.5 s | — |

Add ~10–40 ms if client and server talk over the internet via a **direct**
Tailscale connection (see [§4](#4-remote-access-over-tailscale)); a DERP
relay adds far more (~50–200 ms) and should be fixed, not budgeted around.

Measure your own setup instead of trusting this table — checkpoint size,
image resolution, and GPU all matter:

```bash
uv run python examples/random_client.py --host <server-host> -n 50
```

This prints, per request, round-trip time and the server's self-reported
infer time, then a p50/p95 summary and `rtt - infer` (an estimate of your
network + serialization overhead).

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ficelle: unknown observation key '...'` | Your observation dict has a key the checkpoint doesn't expect (typo, wrong camera name, stray key). | Print `client.metadata["observations"]` and match keys exactly. |
| `ficelle: missing observation key '...'` | Your observation dict is missing a key the checkpoint requires. | Same as above — `client.metadata["observations"]` lists every required key. |
| `ficelle: '...' has shape [...], expected [...]` | Wrong resolution, missing/extra `n_obs_steps` stacking axis, or wrong array shape. | Resize/crop images and reshape arrays client-side to match `frame_shape`/`wire_shape` from the metadata. |
| `ficelle: '...' has dtype ..., expected uint8/float32` | Sent e.g. `float64` state or `int32` image. | Cast client-side: images to `np.uint8`, states to `np.float32`. |
| `PolicyServerError: server at ws://... rejected the api key — check --api_key on both sides` (raw curl/HTTP clients instead see a `403` body of `ficelle: invalid or missing api key`) | Server was started with `--api_key` but the client didn't pass a matching one (or vice versa). | Pass the same `api_key=...` to `PolicyClient(...)` as `--api_key` on the server. |
| `ficelle: server at ws://... does not support jpeg encoding` | The server is running an older version of ficelle that predates JPEG support. | Upgrade the server to the current version, or drop `jpeg_quality` from the client. |
| Client logs `waiting for server at ...` forever | Wrong host/port, a firewall in the way, or the server is bound to a different interface than the client is reaching (e.g. server on `0.0.0.0` on a machine whose tailnet IP differs from what you passed). | Confirm the server's startup log ("serving on ws://...") matches the address the client is dialing; check firewalls; on Tailscale confirm both machines see each other with `tailscale status`. |
| Server fails to start, error mentions `AutoProcessor` / tokenizer loading | Known `transformers` v5 + `physical-intelligence/fast` Hub packaging bug: the tokenizer repo referenced by the checkpoint's config isn't resolved correctly unless the checkpoint points at its **own** fitted tokenizer repo. | Check the checkpoint's config references its own tokenizer (e.g. `SteveNguyen/fast_tokenizer_grabette_pick3`), not a shared/generic one. |
| Latency much higher than [§6](#6-latency-expectations) despite the server logging fast `infer_ms` | Network path, not the policy. Most commonly a Tailscale DERP relay instead of a direct connection. | Run `tailscale ping <server-ip>` — if it says `via DERP`, fix connectivity (NAT/firewall) so Tailscale can establish a direct path; see [§4](#4-remote-access-over-tailscale). |
| Metadata shows `"fps": None` | The checkpoint has no `train_config.json` (or it points at a training dataset that's unreachable), so ficelle couldn't look up the training frame rate. | Harmless — `fps` is informational only and isn't used by the server or client for anything. |
| `ValueError: iroh transport needs the extra: pip install 'ficelle-client[iroh]'` | `open_client`/`IrohPolicyClient` was called with a ticket but the `iroh` package isn't installed. | Install the extra — see [§5](#5-iroh-transport-zero-config-remote). |
| iroh client raises an error building the endpoint from the ticket string (malformed/truncated ticket) | The ticket was cut off, mistyped, or copy-pasted from the wrong log line — it's a long opaque string and easy to truncate. | Re-copy the full `iroh ticket: endpointv1...` line the server logs on startup; don't split it across lines. |
| `PolicyServerError: inference timed out after ...s (link degraded or server stuck)` | Either NAT hole-punching fell back to a slow path, the server is overloaded/stuck, or `infer_timeout` is too tight for this link. | Retry (the client isn't wedged — the next `.infer()` still works); if it persists, raise `infer_timeout`, check the server log, or set `--relay-url` to a closer self-hosted relay. |

## 8. Development

```bash
uv run pytest              # fast suite: no checkpoint download, no GPU needed
uv run pytest -m checkpoint  # slow suite: downloads and runs a real checkpoint (wants a GPU)
```

Layout:

- `serve.py` — websocket transport (`FicelleServer`) and the CLI (`main()`).
  Stays free of `torch`/`lerobot` imports at module import time so the fast
  test suite doesn't pay model-stack import cost.
- `policy_host.py` — everything that touches lerobot: checkpoint loading,
  the metadata contract, and turning raw numpy observations into action
  chunks (`load_policy_host`, `PolicyHost`).
- `errors.py` — `ObservationError`, the one exception type that turns into a
  one-line `ficelle: ...` message sent back to the client instead of a stack
  trace.
- `client/ficelle_client/` — the pip-installable client package:
  `policy_client.py` (`PolicyClient`, `PolicyServerError`),
  `iroh_client.py` (`IrohPolicyClient`, the `[iroh]` extra), `__init__.py`
  (`open_client`, dispatching on ticket vs. `host:port`), and
  `msgpack_numpy.py` (the vendored, pickle-free msgpack codec shared by both
  client and server).
- `examples/random_client.py` — the latency probe used in [§6](#6-latency-expectations).
- `iroh_transport.py` — iroh transport (`IrohPolicyServer`), the alternative
  to `serve.py`'s websocket transport described in
  [§5](#5-iroh-transport-zero-config-remote).
