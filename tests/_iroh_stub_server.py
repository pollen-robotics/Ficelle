"""Stub IrohPolicyServer runner for the client-half iroh tests, run as a
SEPARATE PROCESS.

Why a subprocess: iroh-ffi's uniffi_set_event_loop is PROCESS-GLOBAL. Running
both IrohPolicyServer and IrohPolicyClient on two asyncio loops in one process
makes the second registration win and the first loop's iroh awaits never
complete (deadlock). Real deployments (and experiments/iroh_probe) always have
server and client in separate processes; this harness matches that.

Prints exactly one line `TICKET <ticket>` to stdout (flushed) once bound, then
serves forever until terminated. Modes:
  (default)         serve the stub metadata/actions
  --sleep-infer S   wrap the policy so predict_action_chunk sleeps S seconds
                    (for the client-side infer_timeout test)
  --echo-image      reply's actions carry the decoded cam0 mean per channel, so
                    the client can assert JPEG pixel-parity across the process
                    boundary (host._policy.seen_batch is unreachable here)
"""

import argparse
import asyncio
import pathlib
import sys
import time

# Run as a bare `python tests/_iroh_stub_server.py` subprocess: only tests/ is
# on sys.path by default, so put the repo root (its parent) first to import the
# server modules the way pytest does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from errors import ObservationError
from iroh_transport import IrohPolicyServer
from policy_host import PolicyHost
from tests.test_policy_host import StubConfig, StubPolicy, identity


class SlowFirstStubPolicy(StubPolicy):
    """Sleeps `sleep_s` on the FIRST infer only, then serves normally — lets a
    test time out the first request and prove the SECOND one still succeeds on
    the same client (the connection/loop must not wedge)."""

    def __init__(self, config, sleep_s: float):
        super().__init__(config)
        self._sleep_s = sleep_s
        self._calls = 0

    def predict_action_chunk(self, batch):
        self._calls += 1
        if self._calls == 1:
            time.sleep(self._sleep_s)
        return super().predict_action_chunk(batch)


class RejectTaskStubPolicy(StubPolicy):
    """Passes client-side validation (metadata is identical both sides) but
    rejects a specific task server-side with a one-line ObservationError —
    which the transport turns into a ST_ERR reply."""

    def predict_action_chunk(self, batch):
        if batch.get("task") == "reject me":
            raise ObservationError("stub rejects task 'reject me'")
        return super().predict_action_chunk(batch)


class SleepStubPolicy(StubPolicy):
    def __init__(self, config, sleep_s: float):
        super().__init__(config)
        self._sleep_s = sleep_s

    def predict_action_chunk(self, batch):
        time.sleep(self._sleep_s)
        return super().predict_action_chunk(batch)


class EchoImageStubPolicy(StubPolicy):
    """Return an action chunk whose first row is the per-channel mean of the
    decoded cam0 (in [0,1]), so the client can verify end-to-end pixel parity
    (an RGB/BGR swap would move these means far apart)."""

    def predict_action_chunk(self, batch):
        self.seen_batch = batch
        cam = batch["observation.images.cam0"]  # (1, n_obs, 3, H, W) float [0,1]
        # mean over batch, time, spatial -> per-channel mean, 3 values
        per_channel = cam.reshape(cam.shape[0], cam.shape[1], 3, -1).mean(dim=(0, 1, 3))
        chunk = torch.zeros(1, self.config.n_action_steps,
                            self.config.output_features["action"].shape[0])
        chunk[0, 0, :3] = per_channel
        return chunk


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-infer", type=float, default=None)
    parser.add_argument("--slow-first", type=float, default=None)
    parser.add_argument("--echo-image", action="store_true")
    parser.add_argument("--reject-task", action="store_true")
    args = parser.parse_args()

    if args.sleep_infer is not None:
        policy = SleepStubPolicy(StubConfig(), args.sleep_infer)
    elif args.slow_first is not None:
        policy = SlowFirstStubPolicy(StubConfig(), args.slow_first)
    elif args.echo_image:
        policy = EchoImageStubPolicy(StubConfig())
    elif args.reject_task:
        policy = RejectTaskStubPolicy(StubConfig())
    else:
        policy = StubPolicy(StubConfig())
    host = PolicyHost(policy, identity, identity,
                      checkpoint="stub/ckpt", device="cpu", fps=None)

    server = IrohPolicyServer(host, relay_url=None)
    server_task = asyncio.ensure_future(server.run())
    await server.wait_ready()
    print(f"TICKET {server.ticket()}", flush=True)
    await server_task


if __name__ == "__main__":
    asyncio.run(_main())
