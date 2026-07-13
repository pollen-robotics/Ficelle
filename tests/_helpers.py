"""Shared test builders/helpers duplicated (byte-for-byte) across multiple
test modules — consolidated here so there's one copy to keep in sync."""

import subprocess
import threading

import numpy as np


def gradient_frame(h: int, w: int) -> np.ndarray:
    """One (h, w, 3) uint8 frame with spatial correlation — JPEG needs
    gradient content (not i.i.d. noise) to compress at all."""
    xx, yy = np.meshgrid(np.linspace(0, 255, w, dtype=np.uint8),
                         np.linspace(0, 255, h, dtype=np.uint8))
    return np.stack([xx, yy, np.full_like(xx, 128)], axis=-1)


def stacked_obs() -> dict:
    # Gradient content (not i.i.d. noise): JPEG needs spatial correlation to
    # compress at all, so this is what makes the "big win or bust" size
    # assertions in the jpeg/resize tests meaningful.
    xx, yy = np.meshgrid(np.arange(96, dtype=np.uint8) * 2, np.arange(96, dtype=np.uint8) * 2)
    frame0 = np.stack([xx, yy, np.full_like(xx, 32)], axis=-1)
    frame1 = np.stack([np.full_like(xx, 200), xx[::-1], yy], axis=-1)
    return {
        "observation.images.cam0": np.stack([frame0, frame1]),
        "observation.state": np.zeros((2, 2), dtype=np.float32),
        "task": "pick",
    }


def read_ticket(proc: subprocess.Popen, timeout: float, label: str,
               reap_timeout: float | None = None) -> str:
    """Read `proc`'s stdout on a helper thread so a wedged spawn can't hang
    the suite forever — if no TICKET line arrives in `timeout`, kill the
    process and fail with whatever it wrote to stderr. `label` names the
    process in the failure message; pass `reap_timeout` to wait for the
    killed process to exit so a startup failure doesn't leak a zombie."""
    result: dict = {}

    def reader():
        for line in proc.stdout:
            if line.startswith("TICKET "):
                result["ticket"] = line[len("TICKET "):].strip()
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout)
    if "ticket" not in result:
        proc.kill()
        stderr = proc.stderr.read() if proc.stderr else ""
        if reap_timeout is not None:
            proc.wait(timeout=reap_timeout)
        raise RuntimeError(f"{label} never printed a ticket in {timeout}s:\n{stderr}")
    return result["ticket"]
