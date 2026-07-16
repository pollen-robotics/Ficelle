```
______________           ___________     
___  ____/__(_)_____________  /__  /____ 
__  /_   __  /_  ___/  _ \_  /__  /_  _ \
_  __/   _  / / /__ /  __/  / _  / /  __/
/_/      /_/  \___/ \___//_/  /_/  \___/ 
   serve any LeRobot policy, anywhere,
   over one thin, low-latency string.
```

Serve any [LeRobot](https://github.com/huggingface/lerobot) policy checkpoint
over the network to a robot, without touching pickle.

## What is this

ficelle is a small server + client pair for running LeRobot policy inference
remotely: the server loads any LeRobot checkpoint on a GPU machine and serves
it to a robot on the other end. The wire protocol is copied from
[openpi](https://github.com/Physical-Intelligence/openpi) — a metadata frame
on connect, then request (observation) / reply (action chunk) over binary
msgpack — **never pickle**. The robot-side client is dependency-light
(`websockets`, `msgpack`, `numpy`; no `torch`, no `lerobot`), so it drops into
any control loop. Two transports are supported: a plain websocket, and
[iroh](https://iroh.computer/) for zero-config remote access without a VPN.

## Install

Server (needs [`uv`](https://docs.astral.sh/uv/)):

```bash
git clone git@github.com:SteveNguyen/Ficelle.git
cd ficelle
uv sync
```

Client (robot side, no GPU needed):

```bash
pip install ficelle-client            # websocket transport only
pip install "ficelle-client[iroh]"    # + iroh transport
```

## Serve

```bash
uv run python serve.py --checkpoint <hf-repo-or-local-path> --transport iroh
# or, over a websocket (needs Tailscale/VPN for remote access):
uv run python serve.py --checkpoint <hf-repo-or-local-path> --host 100.x.y.z
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | *(required)* | HF Hub repo id or local checkpoint dir |
| `--transport` | `websocket` | `websocket` or `iroh` (recommended for remote access) |
| `--host` / `--port` | `0.0.0.0` / `8000` | websocket bind address/port |
| `--api_key` | `None` | optional shared secret (websocket only, not supported with iroh) |
| `--device` | `cuda` if available, else `cpu` | inference device |
| `--dtype` | checkpoint's native dtype | `float32` or `bfloat16` — use `bfloat16` for large models |
| `--relay_url` | `None` | iroh only: self-hosted relay URL (else n0 public relays) |

`--checkpoint` accepts a Hugging Face Hub repo id (downloaded and cached
automatically) or a local checkpoint directory. On startup the server prints
the checkpoint it loaded, the exact observation contract it expects, and
either a `ws://` address or an iroh ticket to give to the client.

### Startup & the first request

On launch, after loading the checkpoint the server runs one warmup inference
before printing its ticket / "ready to serve" — so by the time you have an
address to dial, the first *real* request is already fast. See
[`docs/remote-access.md#startup--the-first-request`](docs/remote-access.md#startup--the-first-request)
for the details (why it's safe, when it's skipped, and the `infer_timeout`
fallback).

## Client

```python
from ficelle_client import open_client

client = open_client(addr, jpeg_quality=90)  # addr: "host:port" or an iroh ticket
print(client.metadata["observations"])       # the exact obs contract
reply = client.infer(obs)                    # obs: uint8 HWC images + float32 state + task
actions = reply["actions"]                   # (n_action_steps, action_dim) float32
```

`open_client` dispatches on the shape of `addr`, so the same call site works
for either transport. Observations are always raw `uint8` HWC images and
`float32` state — `jpeg_quality=` and `resize=` are transport-only options
(they change how frames go over the wire, not what you send or what the
policy sees).

## Remote access

[iroh](https://iroh.computer/) is the recommended zero-config remote path:
the client dials the ticket the server printed, iroh handles NAT
hole-punching, and there's a free relay fallback when a direct path isn't
possible. Always pass `jpeg_quality=90` when using iroh — the relay fallback
is bandwidth-throttled and raw frames get slow. See
[`docs/remote-access.md`](docs/remote-access.md) for the full story,
including the websocket-over-Tailscale VPN alternative.

## Latency

| Checkpoint | Setup | Per-chunk latency |
|---|---|---|
| Diffusion (~100M params) | localhost, laptop GPU, 8-action chunk | ~52 ms server inference |
| π0.5 (flow VLA), fp32 | RTX 5090, 50×11 chunk, laptop client through NAT, iroh + `jpeg_quality=90` | 175–190 ms per chunk request |
| π0-FAST | — | currently broken upstream in LeRobot (free-running generation collapses — a model/training issue, not ficelle); not a current example |

Notes:

- The π0.5 number amortizes to ~12 ms/action at a 15-action replan prefix
  (~8 ms client-side queue pops between replans).
- **Always use JPEG over iroh**: raw frames measured ~0.6 s per request on a
  direct path but **3.7–5 s when iroh fell back to the relay** (~1 Mbps
  throttle).
- **Cold start**: the server now pays the ~30 s CUDA/model warmup at startup
  (see [Startup & the first request](#startup--the-first-request)), not on
  the first request. If warmup is skipped for a checkpoint, the first
  request pays it instead — set the client's `infer_timeout` to **at least
  60 s** for VLA models as a fallback. Subsequent requests are fast either
  way.
- CPU serving is ~8 s/request (dev/testing only, not for real-time control).

## Troubleshooting

Common errors and fixes are in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Development

```bash
uv run pytest                # fast suite: no checkpoint download, no GPU needed
uv run pytest -m checkpoint   # slow suite: downloads and runs a real checkpoint (wants a GPU)
```

Layout:

- `serve.py` — websocket transport (`FicelleServer`) and the CLI (`main()`).
- `policy_host.py` — everything that touches lerobot: checkpoint loading, the
  metadata contract, and running inference (`load_policy_host`, `PolicyHost`).
- `iroh_transport.py` — the iroh transport (`IrohPolicyServer`).
- `errors.py` — `ObservationError`, the one exception type that turns into a
  one-line `ficelle: ...` message sent back to the client instead of a stack
  trace.
- `client/ficelle_client/` — the pip-installable client package
  (`policy_client.py`, `iroh_client.py`, `__init__.py`'s `open_client`, and
  the vendored `msgpack_numpy.py` codec).
- `examples/random_client.py` — the latency probe used above.

## Design & background

- [`docs/policy_server_project_brief.md`](docs/policy_server_project_brief.md)
  — the original design brief and decisions behind ficelle.
- [`docs/policy_serving_survey.md`](docs/policy_serving_survey.md) — a
  point-in-time survey of the VLA policy-serving landscape.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). The vendored
`client/ficelle_client/msgpack_numpy.py` is Apache-2.0 from
[openpi](https://github.com/Physical-Intelligence/openpi) — see its file
header for attribution.
