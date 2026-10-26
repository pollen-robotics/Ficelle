"""ficelle: serve any LeRobot policy checkpoint over iroh (QUIC + NAT hole-
punching, see experiments/iroh_probe/).

Protocol ("ficelle/1" over the ALPN below), one request per QUIC bidirectional
stream:
- client opens a bi stream, writes a single framed request, finishes its send
  side: `bytes([opcode]) + payload`
    - OP_META: no payload; reply is one msgpack frame, the metadata dict
    - OP_INFER: payload is a msgpack observation dict; reply is
      `bytes([status]) + msgpack frame` where status is ST_OK (body is the
      {"actions": ..., "server_timing": {...}} dict) or ST_ERR (body is a
      one-line "ficelle: ..." UTF-8 error message)
- an unknown opcode is answered the same way as an observation error: ST_ERR
  and a one-line message.
"""

import asyncio
import logging

import iroh

from errors import POLICY_GENERATION_ERROR, ObservationError
from ficelle_client import msgpack_numpy

logger = logging.getLogger("ficelle")

ALPN = b"ficelle/1"
OP_META = 0x00
OP_INFER = 0x01
ST_OK = 0x00
ST_ERR = 0x01
MAX_FRAME = 8_000_000


def _set_event_loop() -> None:
    # iroh-ffi bridges Rust async to the running asyncio loop; required before
    # any awaited iroh call (see experiments/iroh_probe/iroh_probe.py).
    try:
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    except AttributeError:
        pass  # newer bindings may set this implicitly


async def _reply(send, data: bytes) -> None:
    """Write one reply and finish the stream, best-effort: if the client has
    already gone away (process killed, connection lost mid-inference) there is
    no one left to tell, so that is logged, not raised."""
    try:
        await send.write_all(data)
        await send.finish()
    except iroh.iroh_ffi.IrohError as e:
        logger.info(f"client went away before the reply was sent ({e.message()})")


async def handle_stream(bi, policy_host) -> None:
    """Serve one request on one bidirectional stream, then close it.

    Reusable by tests: takes anything with a bi.recv()/bi.send() pair shaped
    like an iroh BiStream, and a policy_host with .metadata/.infer().
    """
    recv, send = bi.recv(), bi.send()
    try:
        req = await recv.read_to_end(MAX_FRAME)
    except iroh.iroh_ffi.IrohError as e:
        # read_to_end(MAX_FRAME) raises message() == "TooLong" when the
        # client's request exceeds the cap — report it like any other bad
        # request. Anything else means the stream/connection died mid-request
        # (client killed or disconnected): nothing to answer, not a server bug.
        if e.message() == "TooLong":
            await _reply(send, bytes([ST_ERR])
                         + "ficelle: request too large (limit 8MB) — enable jpeg_quality/resize "
                           "to shrink observations".encode())
        else:
            logger.info(f"client went away mid-request ({e.message()})")
        return
    try:
        if not req:
            raise ObservationError("empty request")
        op, payload = req[0], req[1:]
        if op == OP_META:
            reply = msgpack_numpy.packb(policy_host.metadata)
        elif op != OP_INFER:
            raise ObservationError(f"unknown opcode {op:#04x} (expected META=0 or INFER=1)")
        else:
            obs = msgpack_numpy.unpackb(payload)
            # inference runs in a thread so the event loop keeps serving other
            # streams during a long (0.5 s pi0fast) forward pass — same reasoning
            # as the websocket server's asyncio.to_thread (see serve.py).
            result = await asyncio.to_thread(policy_host.infer, obs)
            reply = bytes([ST_OK]) + msgpack_numpy.packb(result)
    except ObservationError as e:
        reply = bytes([ST_ERR]) + f"ficelle: {e}".encode()
    except AssertionError:
        logger.error("policy failed to generate actions", exc_info=True)
        reply = bytes([ST_ERR]) + POLICY_GENERATION_ERROR.encode()
    except Exception as e:  # noqa: BLE001 — protect the accept loop, report to client
        logger.exception("internal error handling iroh stream")
        reply = bytes([ST_ERR]) + f"ficelle: internal server error ({type(e).__name__})".encode()
    await _reply(send, reply)


class IrohPolicyServer:
    """iroh transport around a policy host: any object with .metadata (dict)
    and .infer(obs dict) -> reply dict (raising ObservationError on bad
    client input) — the same contract FicelleServer (websocket) wraps."""

    def __init__(self, policy_host, relay_url: str | None = None):
        self._policy_host = policy_host
        self._relay_url = relay_url
        self._ep: iroh.Endpoint | None = None
        self._ready = asyncio.Event()
        self._bind_error: Exception | None = None
        self._tasks: set[asyncio.Task] = set()  # strong refs so background tasks aren't GC'd mid-flight

    def _options(self) -> "iroh.EndpointOptions":
        kwargs = {"preset": iroh.preset_n0(), "alpns": [ALPN]}
        if self._relay_url is not None:
            kwargs["relay_mode"] = iroh.RelayMode.custom_from_urls([self._relay_url])
        return iroh.EndpointOptions(**kwargs)

    async def wait_ready(self) -> None:
        """Resolve once the endpoint is bound and addr()/ticket() are valid, or
        raise if startup failed (bind errors would otherwise hang this forever,
        since _ready was never set)."""
        await self._ready.wait()
        if self._bind_error is not None:
            raise self._bind_error

    def addr(self):
        return self._ep.addr()

    def ticket(self) -> str:
        return str(iroh.EndpointTicket.from_addr(self._ep.addr()))

    def _spawn(self, coro) -> None:
        """Schedule a background task, keeping a strong reference so it can't
        be garbage-collected mid-flight (asyncio only holds a weak ref)."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Drop the strong ref, and log any exception the task raised so it
        isn't silently swallowed (also silences asyncio's cosmetic "Task
        exception was never retrieved" warning)."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("background iroh task failed", exc_info=exc)

    async def run(self) -> None:
        _set_event_loop()
        try:
            self._ep = await iroh.Endpoint.bind(self._options())
        except Exception as e:
            self._bind_error = e
            self._ready.set()
            raise
        self._ready.set()
        logger.info(f"iroh node id: {self._ep.id()}")
        while True:
            incoming = await self._ep.accept_next()
            if incoming is None:
                break
            self._spawn(self._serve_connection(incoming))

    async def _serve_connection(self, incoming) -> None:
        try:
            conn = await (await incoming.accept()).connect()
        except Exception:  # noqa: BLE001 — a failed handshake shouldn't kill the accept loop
            logger.exception("iroh connection failed to establish")
            return
        logger.info(f"iroh connection from {conn.remote_id()}")
        try:
            while True:
                bi = await conn.accept_bi()
                self._spawn(handle_stream(bi, self._policy_host))
        except Exception:  # noqa: BLE001 — connection closed ends the loop, same as the probe
            logger.info(f"iroh connection from {conn.remote_id()} closed")
