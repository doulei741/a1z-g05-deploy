"""Offline timing tests for chunk prefetching (stub server, no GPU, no robot).

The client is driven in dry-run mode (no ``--execute``) against
``tests/stub_policy_server.py``, which emulates the real 670 ms recompute. Every
``step`` line the client prints is timestamped by the parent process, so the
per-step interval shows exactly where the robot would freeze.

Run one case by hand::

    python tests/test_chunk_prefetch.py baseline        # current behaviour
    python tests/test_chunk_prefetch.py prefetch        # after phase 1

Or let pytest assert on the same measurements.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "a1z_g05_deploy" / "client" / "dual_arm_client.py"
STUB = Path(__file__).resolve().parent / "stub_policy_server.py"
PYTHON = os.environ.get("G05_CLIENT_PYTHON", "/home/dj/miniconda3/envs/lerobot-a1z/bin/python")
NOMINAL_PERIOD = 1 / 30.0
STALL_THRESHOLD = 0.1          # 3x the 30 Hz period

# Prefetch is currently reverted; the measurements stay available, the
# assertions resume as soon as the client serves local chunks again.
sys.path.insert(0, str(ROOT / "a1z_g05_deploy" / "client"))
import dual_arm_client  # noqa: E402

PREFETCH_REVERTED = getattr(dual_arm_client, "adopt_action_chunk", None) is None
pytestmark = pytest.mark.skipif(PREFETCH_REVERTED,
                                reason="chunk prefetch path is not present")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def clean_env() -> dict:
    env = dict(os.environ)
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                 "http_proxy", "https_proxy"):
        env.pop(name, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def client_config(port: int, steps: int, *, prefetch: bool | None) -> Path:
    config = {
        "server_host": "127.0.0.1",
        "server_port": port,
        "control_frequency": 30,
        "action_steps": 24,
        "max_joint_step": 0.1,
        "max_initial_joint_step": 0.1,
        "request_timeout": 60,
        "max_cycles": steps,
        "task": "stub timing probe",
        "embodiment_type": "galaxea_a1z",
        "image_size": [640, 480],
        "websocket_compression": None,
        "enforce_joint_step": True,
        "smoothing": {"enabled": False},
        "gripper_boundary_tolerance": 0.02,
        "left_can": "can0",
        "right_can": "can1",
        "camera_head": 1,
        "camera_left_wrist": 2,
        "camera_right_wrist": 3,
    }
    if prefetch is not None:
        config["prefetch_next_chunk"] = prefetch
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(config, handle)
    handle.close()
    return Path(handle.name)


def measure(*, steps: int = 180, latency: float = 0.67, action_steps: int = 24,
            no_chunk: bool = False, prefetch: bool | None = True) -> dict:
    """Run one client/stub pair and return the measured per-step intervals."""
    port = free_port()
    stub_args = [PYTHON, str(STUB), "--port", str(port), "--latency", str(latency),
                 "--action-steps", str(action_steps)]
    if no_chunk:
        stub_args.append("--no-chunk")
    stub = subprocess.Popen(stub_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=clean_env())
    try:
        ready = stub.stdout.readline()
        if "stub ready" not in ready:
            raise RuntimeError(f"stub failed to start: {ready}{stub.stdout.read()}")
        config = client_config(port, steps, prefetch=prefetch)
        client = subprocess.Popen([PYTHON, str(CLIENT), "--config", str(config)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, env=clean_env())
        started = time.monotonic()
        events = []
        for line in client.stdout:
            events.append((time.monotonic() - started, line.rstrip("\n")))
        status = client.wait(timeout=180)
        config.unlink()
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stub.kill()

    steps_at = [(t, line) for t, line in events
                if ("dry-run step" in line or "execute step" in line
                    or "observe-only step" in line)]
    intervals = [b[0] - a[0] for a, b in zip(steps_at, steps_at[1:])]
    stalls = [(a[1], round(b[0] - a[0], 3)) for a, b in zip(steps_at, steps_at[1:])
              if b[0] - a[0] > STALL_THRESHOLD]
    errors = [line for _, line in events if "error" in line.lower() or "Traceback" in line]
    return {
        "status": status,
        "steps": len(steps_at),
        "intervals": intervals,
        "mean": statistics.mean(intervals) if intervals else 0.0,
        "median": statistics.median(intervals) if intervals else 0.0,
        "max": max(intervals) if intervals else 0.0,
        "p95": sorted(intervals)[int(0.95 * len(intervals))] if intervals else 0.0,
        "stalls": stalls,
        "stall_seconds": sum(value for _, value in stalls),
        "stall_share": (sum(value for _, value in stalls) / sum(intervals)) if intervals else 0.0,
        "errors": errors,
        "action_steps": action_steps,
        "log": events,
    }


def report(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"status={result['status']} steps={result['steps']} "
          f"mean={result['mean']*1000:.1f}ms median={result['median']*1000:.1f}ms "
          f"p95={result['p95']*1000:.1f}ms max={result['max']*1000:.1f}ms")
    print(f"stalls(>100ms)={len(result['stalls'])} "
          f"total_stall={result['stall_seconds']:.2f}s "
          f"share={result['stall_share']*100:.0f}%")
    if result["stalls"]:
        print("  first stalls:", result["stalls"][:6])
    if result["errors"]:
        print("  errors:", result["errors"][:3])


# ── pytest: the behaviour phase 1 must deliver ────────────────────────

def test_chunk_serving_with_prefetch_has_no_stall():
    result = measure(steps=180, latency=0.67, action_steps=24, prefetch=True)
    report("chunk + prefetch", result)
    assert result["status"] == 0, result["errors"]
    assert not result["stalls"], f"unexpected stalls: {result['stalls'][:5]}"
    assert result["mean"] == pytest.approx(NOMINAL_PERIOD, abs=0.01)


def test_prefetch_can_be_disabled_for_ab_comparison():
    result = measure(steps=120, latency=0.67, action_steps=24, prefetch=False)
    report("chunk, prefetch disabled", result)
    assert result["status"] == 0, result["errors"]
    assert len(result["stalls"]) >= 3


def test_step_mode_server_still_works_without_chunks():
    """serve_policy.py has no action_chunk: the client must keep step serving."""
    result = measure(steps=120, latency=0.67, action_steps=24, no_chunk=True)
    report("no action_chunk (serve_policy.py)", result)
    assert result["status"] == 0, result["errors"]
    assert len(result["stalls"]) >= 3


def test_inference_slower_than_chunk_window_degrades_gracefully():
    """1.2 s > 24 steps x 34 ms: stalls come back but nothing breaks."""
    result = measure(steps=120, latency=1.2, action_steps=24, prefetch=True)
    report("slow inference", result)
    assert result["status"] == 0, result["errors"]
    assert result["stalls"]


CASES = {
    "baseline": dict(steps=180, latency=0.67, action_steps=24, no_chunk=False, prefetch=None),
    "stepmode": dict(steps=180, latency=0.67, action_steps=24, no_chunk=True, prefetch=None),
    "prefetch": dict(steps=180, latency=0.67, action_steps=24, prefetch=True),
    "noprefetch": dict(steps=120, latency=0.67, action_steps=24, prefetch=False),
    "slow": dict(steps=120, latency=1.2, action_steps=24, prefetch=True),
}


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if case == "all":
        for name, kwargs in CASES.items():
            report(name, measure(**kwargs))
    else:
        report(case, measure(**CASES[case]))
