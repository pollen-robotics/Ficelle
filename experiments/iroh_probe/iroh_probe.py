"""Standalone iroh 1.0 transport probe for the ficelle remote-serving use case.

This is a throwaway experiment, NOT part of ficelle's shipped code. It answers
the questions we need before committing an iroh transport into ficelle:

  1. Does iroh get a DIRECT (hole-punched) path between the robot laptop and
     the GPU box, or does it fall back to a relay? How long until direct?
  2. What is the request/reply round-trip latency for ficelle-sized payloads
     (an observation up, an action chunk down), at p50/p95/p99?
  3. What effective request rate (Hz) can we sustain vs the 30 Hz control loop?

It models ficelle's actual traffic: the `probe` side sends an observation-sized
payload and the `serve` side replies with a small action-chunk-sized payload,
one QUIC bidirectional stream per request (= one control tick).

Usage (two machines, e.g. laptop + 5090 over the internet, NO Tailscale):

    # on the GPU box
    uv run python iroh_probe.py serve

    # copy the printed ticket, then on the laptop
    uv run python iroh_probe.py probe <ticket> --profile jpeg -n 100

Loopback smoke test (one machine, two terminals) validates the code but will
always report a direct/local path with ~0 latency — real numbers need two
machines on different networks.
"""

import argparse
import asyncio
import statistics
import time

import iroh

ALPN = b"ficelle/iroh-probe/0"

# ficelle-shaped payload profiles: (request = observation up, reply = actions down)
PROFILES = {
    "control": 1_000,       # tiny control message
    "jpeg": 250_000,        # ~JPEG-encoded 2-frame observation
    "raw": 4_050_000,       # ~raw uint8 2x720x960x3 observation
}
REPLY_BYTES = 1_000         # ~an 8x11 float32 action chunk + framing


def _set_event_loop() -> None:
    # iroh-ffi bridges Rust async to the running asyncio loop; required before
    # any awaited iroh call (see iroh-ffi python/main.py).
    try:
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    except AttributeError:
        pass  # newer bindings may set this implicitly


def _endpoint_options(*, accept: bool, relay_url: str | None):
    kwargs = {"preset": iroh.preset_n0()}  # n0 public relays + discovery
    if accept:
        kwargs["alpns"] = [ALPN]
    if relay_url is not None:
        # point at a self-hosted relay instead of n0's (the future ficelle
        # deployment step) — direct paths still bypass it; it's fallback only.
        kwargs["relay_mode"] = iroh.RelayMode.custom_from_urls([relay_url])
    return iroh.EndpointOptions(**kwargs)


def _fields(obj) -> dict:
    """Read every public attribute of a uniffi record/instance (fields show on
    instances, not the class), calling zero-arg accessors safely."""
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
            out[name] = val() if callable(val) else val
        except Exception as e:  # noqa: BLE001
            out[name] = f"<err {e}>"
    return out


def describe_paths(conn) -> str:
    """Fully introspective dump of the connection's paths — iroh 1.0's Python
    PathSnapshot fields aren't documented yet, so we print the real structure
    at runtime. This is how we lock the exact API for the ficelle integration."""
    lines = []
    try:
        lines.append(f"conn.rtt = {conn.rtt() if callable(conn.rtt) else conn.rtt}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"conn.rtt unavailable: {e}")
    try:
        paths = conn.paths()
        lines.append(f"paths ({len(paths)}):")
        for i, p in enumerate(paths):
            lines.append(f"  [{i}] {_fields(p)}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"paths() unavailable: {e}")
    return "\n".join(lines)


def looks_direct(conn) -> bool | None:
    """Is any active path a direct (non-relay) UDP path? Returns None if the
    exposed fields don't let us tell (then rely on the describe_paths dump).
    A relayed path carries a relay url; a direct path carries a UDP socket addr.
    Field names are resolved defensively against the runtime PathSnapshot."""
    # iroh 1.0 PathSnapshot fields (confirmed at runtime): is_ip, is_relay,
    # is_selected, remote_addr, rtt_ms. The SELECTED path is the one in use;
    # a direct hole-punched path has is_ip=True/is_relay=False.
    try:
        paths = conn.paths()
    except Exception:  # noqa: BLE001
        return None
    if not paths:
        return None
    selected = [p for p in paths if _fields(p).get("is_selected")] or paths
    f = _fields(selected[0])
    if f.get("is_relay"):
        return False
    if f.get("is_ip"):
        return True
    return None


async def serve(relay_url: str | None) -> None:
    ep = await iroh.Endpoint.bind(_endpoint_options(accept=True, relay_url=relay_url))
    ticket = iroh.EndpointTicket.from_addr(ep.addr())
    print(f"node id: {ep.id()}")
    print(f"ticket:  {ticket}")
    print("waiting for a connection...", flush=True)

    reply = bytes(REPLY_BYTES)
    while True:
        incoming = await ep.accept_next()
        if incoming is None:
            break
        accepting = await incoming.accept()
        conn = await accepting.connect()
        print(f"\nconnection from {conn.remote_id()}")
        print(describe_paths(conn), flush=True)
        served = 0
        try:
            while True:
                bi = await conn.accept_bi()
                recv, send = bi.recv(), bi.send()
                # read the whole observation (client finishes its send side)
                await recv.read_to_end(8_000_000)
                await send.write_all(reply)
                await send.finish()
                served += 1
        except Exception as e:  # noqa: BLE001 — connection closed ends the loop
            print(f"connection closed after {served} requests ({type(e).__name__})",
                  flush=True)


SWEEP_SIZES = [1_000, 25_000, 100_000, 250_000, 1_000_000]


async def probe(ticket_str: str, payload_bytes: int, count: int, warmup: int,
                relay_url: str | None, sweep: list[int] | None) -> None:
    ep = await iroh.Endpoint.bind(_endpoint_options(accept=False, relay_url=relay_url))
    ticket = iroh.EndpointTicket.from_string(ticket_str)
    addr = ticket.endpoint_addr()
    print(f"dialing {addr.id()} ...", flush=True)
    conn = await ep.connect(addr, ALPN)
    print(f"connected to {conn.remote_id()}")

    async def one_request(payload: bytes) -> float:
        t0 = time.perf_counter()
        bi = await conn.open_bi()
        send, recv = bi.send(), bi.recv()
        await send.write_all(payload)
        await send.finish()
        await recv.read_to_end(1_000_000)
        return (time.perf_counter() - t0) * 1000.0

    # observe path/type right after connect and time-to-direct
    t_connect = time.perf_counter()
    print("\n--- connection paths at t=0 ---")
    print(describe_paths(conn), flush=True)
    time_to_direct = 0.0 if looks_direct(conn) else None

    # warmup (also gives hole-punch a moment to upgrade relay -> direct)
    for _ in range(warmup):
        await one_request(bytes(1000))
        if time_to_direct is None and looks_direct(conn):
            time_to_direct = time.perf_counter() - t_connect

    def stats_line(label: str, rtts: list[float]) -> str:
        q = lambda n: statistics.quantiles(rtts, n=100)[n - 1] if len(rtts) > 1 else rtts[0]
        mean = statistics.mean(rtts)
        return (f"  {label:>10} | p50 {q(50):6.1f}  p95 {q(95):6.1f}  p99 {q(99):6.1f} ms"
                f" | {1000.0/mean:6.1f} Hz")

    sizes = sweep if sweep else [payload_bytes]
    results = []
    for size in sizes:
        payload = bytes(size)
        rtts = [await one_request(payload) for _ in range(count)]
        if time_to_direct is None and looks_direct(conn):
            time_to_direct = time.perf_counter() - t_connect
        results.append((size, rtts))

    print("\n--- connection paths after run ---")
    print(describe_paths(conn))
    try:
        print(f"stats: {conn.stats()}")
    except Exception as e:  # noqa: BLE001
        print(f"stats() unavailable: {e}")

    direct = looks_direct(conn)
    print("\n============================================================")
    print(f"  iroh probe: {count} req/size, {REPLY_BYTES/1000:.0f} KB reply down")
    print(f"  path:            {'DIRECT' if direct else 'RELAY' if direct is False else 'UNKNOWN (see dump)'}")
    print(f"  time to direct:  {time_to_direct*1000:.0f} ms" if time_to_direct is not None
          else "  time to direct:  n/a")
    print(f"  conn.rtt:        {conn.rtt() if callable(conn.rtt) else conn.rtt} ms")
    print("  ---- round-trip by request payload (30 Hz control target) ----")
    for size, rtts in results:
        print(stats_line(f"{size/1000:.0f}KB", rtts))
    print("============================================================")
    await ep.close()


async def main() -> None:
    _set_event_loop()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="bind an endpoint, print a ticket, echo replies")
    ps.add_argument("--relay-url", default=None, help="self-hosted relay URL (else n0 public relays)")

    pp = sub.add_parser("probe", help="dial a ticket and measure request/reply latency")
    pp.add_argument("ticket")
    pp.add_argument("--profile", choices=list(PROFILES), default="jpeg",
                    help="payload size profile (default: jpeg ~250KB)")
    pp.add_argument("--payload-bytes", type=int, default=None,
                    help="override the profile with an exact request size in bytes")
    pp.add_argument("-n", "--count", type=int, default=100)
    pp.add_argument("--warmup", type=int, default=5)
    pp.add_argument("--relay-url", default=None)
    pp.add_argument("--sweep", action="store_true",
                    help="run several payload sizes back-to-back on one connection "
                         "and print the payload->latency curve (ignores --profile)")
    args = parser.parse_args()

    if args.cmd == "serve":
        await serve(args.relay_url)
    else:
        size = args.payload_bytes if args.payload_bytes is not None else PROFILES[args.profile]
        await probe(args.ticket, size, args.count, args.warmup, args.relay_url,
                    SWEEP_SIZES if args.sweep else None)


if __name__ == "__main__":
    asyncio.run(main())
