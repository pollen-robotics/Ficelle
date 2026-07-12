"""ficelle: serve any LeRobot policy checkpoint over websockets.

Protocol ("ficelle/1", openpi-style — see README):
- on connect the server sends one msgpack frame: the metadata dict
- then request/reply: client sends a msgpack observation dict, server replies
  one msgpack frame {"actions": ..., "server_timing": {...}}
- a TEXT frame from the server is always a one-line error message; validation
  errors keep the connection open, internal errors close it
"""

import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.exceptions
import websockets.frames

from errors import ObservationError
from ficelle_client import msgpack_numpy

logger = logging.getLogger("ficelle")


class FicelleServer:
    """Websocket transport around a policy host: any object with
    .metadata (dict) and .infer(obs dict) -> reply dict (raising
    ObservationError on bad client input)."""

    def __init__(self, host_obj, host: str, port: int, api_key: str | None = None):
        self._policy_host = host_obj
        self._host = host
        self._port = port
        self._api_key = api_key

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            logger.info(f"serving on ws://{self._host}:{self._port}")
            await server.serve_forever()

    def _process_request(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self._api_key is not None and request.headers.get("Authorization") != f"Api-Key {self._api_key}":
            return connection.respond(http.HTTPStatus.FORBIDDEN, "ficelle: invalid or missing api key\n")
        return None

    async def _handler(self, websocket):
        logger.info(f"connection from {websocket.remote_address} opened")
        try:
            await self._serve_connection(websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        logger.info(f"connection from {websocket.remote_address} closed")

    async def _serve_connection(self, websocket) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy_host.metadata))
        while True:
            request = await websocket.recv()
            start = time.monotonic()
            try:
                obs = msgpack_numpy.unpackb(request)
                if not isinstance(obs, dict):
                    raise ObservationError("request must be a msgpack-encoded observation dict")
                # inference runs in a thread so websocket keepalive pings keep
                # flowing during a long (0.5 s pi0fast) forward pass
                result = await asyncio.to_thread(self._policy_host.infer, obs)
            except ObservationError as e:
                await websocket.send(f"ficelle: {e}")
                continue
            except Exception:
                logger.error(traceback.format_exc())
                await websocket.send("ficelle: internal server error — see server log for the traceback")
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="internal error"
                )
                return
            result.setdefault("server_timing", {})["total_ms"] = (time.monotonic() - start) * 1000.0
            await websocket.send(packer.pack(result))
