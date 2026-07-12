"""Synthetic-observation latency probe for a ficelle server.

Reads the server's metadata, sends N random observations that match it, and
prints round-trip latency percentiles. Needs only ficelle-client + numpy —
run it from any machine that can reach the server (e.g. over Tailscale):

    uv run python examples/random_client.py --host 100.x.y.z [-n 50] [--task "pick up the cup"]
"""

import argparse
import logging
import time

import numpy as np

from ficelle_client import PolicyClient


def synthetic_obs(metadata: dict, task: str, rng: np.random.Generator) -> dict:
    obs = {"task": task}
    for key, spec in metadata["observations"].items():
        if spec["dtype"] == "uint8":
            obs[key] = rng.integers(0, 256, size=spec["wire_shape"], dtype=np.uint8)
        else:
            obs[key] = rng.standard_normal(spec["wire_shape"]).astype(np.float32)
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("-n", type=int, default=50, help="number of timed requests")
    parser.add_argument("--task", default="", help="task string to send (for VLA policies)")
    parser.add_argument("--jpeg_quality", type=int, default=None, help="encode images as JPEG at this quality (1-100) before sending")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    with PolicyClient(args.host, args.port, api_key=args.api_key, jpeg_quality=args.jpeg_quality) as client:
        meta = client.metadata
        print(f"policy {meta['policy_type']} from {meta['checkpoint']} on {meta['device']} | "
              f"{meta['n_action_steps']} x {meta['action_dim']} actions per request | "
              f"n_obs_steps {meta['n_obs_steps']} | training fps {meta['fps']}")

        # warmup request: first inference pays cuda/tokenizer warmup, don't time it
        reply = client.infer(synthetic_obs(meta, args.task, rng))
        expected_shape = (meta["n_action_steps"], meta["action_dim"])
        assert reply["actions"].shape == expected_shape, reply["actions"].shape

        rtt, infer = [], []
        for i in range(args.n):
            # generate the observation OUTSIDE the timed bracket: rng over full
            # image stacks costs milliseconds and isn't client/wire/server time
            obs = synthetic_obs(meta, args.task, rng)
            t0 = time.perf_counter()
            reply = client.infer(obs)
            rtt.append((time.perf_counter() - t0) * 1000.0)
            infer.append(reply["server_timing"]["infer_ms"])
            print(f"req {i:3d} | rtt {rtt[-1]:7.1f} ms | server infer {infer[-1]:7.1f} ms")

        rtt_a, infer_a = np.array(rtt), np.array(infer)
        print(f"\n{args.n} requests | rtt p50 {np.percentile(rtt_a, 50):.1f} ms "
              f"p95 {np.percentile(rtt_a, 95):.1f} ms | "
              f"server infer p50 {np.percentile(infer_a, 50):.1f} ms "
              f"p95 {np.percentile(infer_a, 95):.1f} ms | "
              f"network+serialization p50 {np.percentile(rtt_a - infer_a, 50):.1f} ms | "
              f"request payload ~{client.last_request_bytes / 1024:.0f} KiB")
        print(f"actions shape OK: {expected_shape}")


if __name__ == "__main__":
    main()
