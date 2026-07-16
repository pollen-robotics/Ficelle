# Remote access

Back to the [README](../README.md). This is the full story on getting the
client and server talking across networks; the README only covers the
recommended default (`--transport iroh`).

## Startup & the first request

A large model's first forward pass triggers CUDA kernel autotuning and other
lazy initialization that can take 30+ seconds. Historically that cost landed
on the first client request. Instead, on launch the server loads the
checkpoint and then runs one **warmup inference** — a synthetic all-zeros
observation through the exact same code path as a real request — before
printing its `ws://` address or iroh ticket / logging "ready to serve". By
the time you have something to dial, the model is already warm, so the
first real request is fast too.

This is:

- **Behavior-neutral**: warmup runs under `eval()`/`no_grad()` (no weight or
  running-stat change), `reset()` runs first exactly as it does for every
  real request (no queue state leaks forward), and the model's RNG state is
  saved and restored around the call — so the model's stochastic sampling
  (flow-matching/diffusion noise draws) is provably unaffected.
- **Best-effort**: some policies reject a synthetic all-zeros probe (e.g.
  pi0fast's free-running generation asserts on degenerate input). If warmup
  is skipped, the server logs a warning and starts anyway — the first real
  request then pays the warmup cost itself, exactly as it did before this
  feature existed. This is why the `infer_timeout ≥ 60s` guidance (see
  [Cold start](#cold-start) below) still applies as a fallback.
- **Slower startup**: expect launch to take extra seconds (up to the ~30 s
  cold-start cost) for large models — the ticket / "ready to serve" line
  only appears after warmup finishes.

## iroh (recommended, zero-config)

`--transport iroh` needs no VPN, no account, and no shared network set up
ahead of time: the server prints a one-line **ticket** that fully describes
how to reach it, and the client dials that ticket directly. Under the hood
it's QUIC with iroh's built-in NAT hole-punching (see `experiments/iroh_probe/`
for the proof of concept this is built on).

```bash
uv run python serve.py --checkpoint <repo-or-path> --transport iroh
```

It logs the usual startup lines, then a ticket instead of a `ws://` address:

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
# client = open_client("100.x.y.z:8000")                # host:port -> PolicyClient
reply = client.infer(obs)                                # same .infer() contract either way
```

Install the client's iroh extra (the plain `pip install ficelle-client`
doesn't pull in `iroh`):

```bash
pip install "ficelle-client[iroh] @ git+<REPLACE-WITH-REPO-URL>#subdirectory=client"
# from a local checkout instead:
pip install "./client[iroh]"
```

### Hole-punching and relay fallback

iroh tries a direct (hole-punched) path first and falls back to a relay if
NAT traversal fails. In practice this direct path succeeds roughly 90% of
the time; the remaining connections fall back to n0's public relay
infrastructure, which is free but rate-limited. Pass `--relay_url <url>` on
the server (and `relay_url=<url>` on `open_client`/`IrohPolicyClient`) to
point both ends at a self-hosted relay instead of the public one.

**Always use JPEG over iroh.** Raw, uncompressed frames took ~0.6 s per
request on a direct path but **3.7–5 s when iroh fell back to the relay**
(the free relay throttles to roughly 1 Mbps). Passing `jpeg_quality=90`
keeps requests fast on both the direct and relayed paths, so it's not
optional — always set it when using iroh.

### Cold start

As of [Startup & the first request](#startup--the-first-request) above, the
~30 s CUDA/model warmup for large models is normally paid once at server
launch, not on the first client request. If warmup was skipped for a given
checkpoint (see above), the first request pays that cost instead — this
isn't a network problem. Keep the client's `infer_timeout` at at least 60
seconds for VLA-class models as a fallback:

```python
client = IrohPolicyClient(ticket, jpeg_quality=90, infer_timeout=60.0)
```

Subsequent requests reuse the warmed-up model and the established
connection, and are fast.

### One client per process

Because iroh-ffi's uniffi event loop is process-global, only **one**
`IrohPolicyClient` (or one iroh endpoint) may be live per process — a second
concurrent one will stall the first. If you need multiple concurrent iroh
connections, run them in separate processes.

### Access control

With the iroh transport, possession of the printed ticket is the only
access control — there is no `--api_key` support over iroh (see
[websocket over Tailscale](#websocket-over-tailscale-vpn-alternative) below
if you need a shared-secret gate). Keep the ticket secret, prefer a
trusted/private network, and don't paste it anywhere public.

## websocket over Tailscale (VPN alternative)

If you already run a [Tailscale](https://tailscale.com/) tailnet, or want a
persistent private network beyond just ficelle, or need `--api_key` support,
run the default websocket transport over Tailscale instead of iroh.

1. Install Tailscale on **both** the GPU machine and the robot, and join them
   to the same tailnet.
2. Bind the server to its **tailnet IP** (not `0.0.0.0`) so `--host` matches
   what `tailscale ip` reports on that machine:

   ```bash
   uv run python serve.py --checkpoint <repo-or-path> --host 100.x.y.z
   ```

3. From the robot, point the client at that same tailnet IP:

   ```bash
   uv run python examples/random_client.py --host 100.x.y.z --task "pick up the mustard bottle" -n 30
   ```

4. **Before trusting latency numbers**, verify the path is direct:

   ```bash
   tailscale ping 100.x.y.z
   ```

   It must say `pong from ... via direct`. If it instead says
   `via DERP <relay>`, Tailscale has fallen back to relaying through a
   Tailscale-operated server (e.g. due to strict NAT/firewalls on one side),
   which silently adds ~50–200 ms of round-trip latency — see
   [troubleshooting.md](troubleshooting.md).

5. Tailscale already authenticates which devices can reach the server, but
   pass `--api_key <secret>` on the server (and the matching `api_key=` on
   the client) as defense in depth — it costs nothing and stops any other
   device on the tailnet from calling your policy server. Note: `--api_key`
   is **not** supported with `--transport iroh` — over iroh, access control
   is ticket possession, not a shared secret.
