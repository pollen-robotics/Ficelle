# iroh transport probe (spike)

Throwaway experiment to decide whether ficelle's remote transport should move
from websockets(+Tailscale) to **iroh** (QUIC + built-in NAT hole-punching,
zero-config "dial a key", free public relay fallback). Not part of ficelle's
shipped code. See `docs/` discussion for why iroh over Tailscale/WebRTC.

It models ficelle's real traffic: one QUIC bidirectional stream per request,
an observation-sized payload up, a small action-chunk-sized payload down.

## Run it

Two machines on different networks (the real test — e.g. robot laptop + 5090,
**no Tailscale**):

```bash
# GPU box
uv run python iroh_probe.py serve
# -> prints a `ticket` string; copy it

# robot laptop
uv run python iroh_probe.py probe <ticket> --profile jpeg -n 100
```

Profiles: `control` (~1 KB), `jpeg` (~250 KB, a JPEG-encoded 2-frame obs),
`raw` (~4 MB, an uncompressed obs). `--payload-bytes N` overrides.
`--sweep` runs several sizes (1/25/100/250/1000 KB) back-to-back on one
connection and prints the payload→latency curve — the fastest way to see
where your link sits. `--relay-url https://your.relay/` points fallback at a
self-hosted relay (`RelayMode.custom_from_urls`) once we deploy one.

Loopback (one machine, two terminals) validates the code but always reports a
direct local path — real direct-vs-relay and latency numbers need two networks.

## Loopback baseline (2026-07-13, single machine — code validation only)

| profile | payload up | rtt p50 | effective Hz |
|---|---|---|---|
| control | 1 KB | 0.4 ms | ~2200 |
| jpeg | 250 KB | 24.8 ms | ~40 |
| raw | 4 MB | 381 ms | ~2.6 |

Findings already visible: the ~4 MB raw obs is a non-starter (~2.6 Hz even on
loopback — a third independent confirmation that JPEG/resize matters); JPEG-sized
requests clear the 30 Hz target with margin. The per-request bidi-stream pattern
plus the 4 MB throughput ceiling (~10 MB/s on loopback) suggests a
receive-window / stream-reuse tuning question (`conn.set_receive_window`,
persistent stream) to explore during the real integration — flagged, not solved.

## First cross-machine run (2026-07-13, real internet, direct IPv6 path)

Hole-punched relay→DIRECT in **203 ms**, stayed direct, 0 flips, ~0.03% loss.
Base path RTT ~43 ms. Round-trip is payload-bound, not latency-bound:

| request payload | rtt p50 (real link) | rtt p50 (loopback) |
|---|---|---|
| 1 KB | ~0.4 ms* | 0.4 ms |
| 250 KB | 94 ms | 23 ms |
| 1 MB | — | 92 ms |

(*run `--sweep` on the real link to fill this in.) At 250 KB the round trip is
~94 ms (≈2–3 RTTs to push ~200 packets through the congestion window), i.e.
~10 Hz raw request rate. For chunked sync control that's plenty (a chunk of 8
at 30 fps is requested only ~3.75 Hz), but each refill blocks ~94 ms + server
inference — so shrinking the payload (resized/higher-compression JPEG, ~25 KB →
~1 RTT) is the highest-leverage, behavior-safe latency win. Async chunk
replacement is NOT an option (breaks these grasp policies — see project notes).

## iroh 1.0 Python API notes (confirmed at runtime, for the integration)

- `import iroh`; asyncio-based; `iroh.iroh_ffi.uniffi_set_event_loop(loop)` before awaited calls.
- `ep = await iroh.Endpoint.bind(iroh.EndpointOptions(preset=iroh.preset_n0(), alpns=[ALPN]))`.
- Pairing: `iroh.EndpointTicket.from_addr(ep.addr())` → str → `from_string(s)` → `.endpoint_addr()`.
- Accept: `inc = await ep.accept_next(); conn = await (await inc.accept()).connect()`.
- Connect: `conn = await ep.connect(addr, ALPN)`.
- Stream: `bi = await conn.open_bi()/accept_bi(); bi.send()/bi.recv()`;
  `await send.write_all(bytes); await send.finish(); data = await recv.read_to_end(max)`.
- Direct-vs-relay: `conn.paths()` → `PathSnapshot(id, is_ip, is_relay, is_selected, remote_addr, rtt_ms, stats)`;
  the `is_selected` path with `is_ip and not is_relay` is a direct hole-punched path.
- `conn.rtt` (QUIC RTT), `conn.stats()` → `ConnectionStats(udp_tx/rx_datagrams, _bytes, lost_packets, lost_bytes)`.
- Relays: `RelayMode.disabled()/default_mode()/staging()/custom_from_urls([...])`; `preset_n0_disable_relay()`.
