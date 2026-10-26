"""Stub policy server for offline client timing tests (no GPU, no hardware).

Emulates the wire behaviour of ``serve_policy_batched.py --ensemble``:

* handshake ``{action_steps, shape_meta, ensemble}`` (same contract the real
  server sends after the handshake patch),
* accepts ``current_step`` in the observation and jumps the absolute step
  counter like ``EnsembleChunkedPolicyWrapper.set_step`` does,
* recomputes a chunk with a configurable latency whenever the buffer is
  exhausted (this is the 670 ms that stalls the robot today),
* returns ``action_chunk`` (overlapping predictions averaged per absolute
  step) plus the single served ``action`` and ``need_obs``,
* can omit ``action_chunk`` to emulate ``serve_policy.py`` (step serving), and
  can inject a failure on the Nth request.

Action values are a deterministic slow ramp so a served sequence is easy to
eyeball; nothing here is used by the robot path.

Usage::

    python tests/stub_policy_server.py --port 8766 --latency 0.67 \
        --action-steps 24 --horizon 32
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

CLIENT_DIR = Path(__file__).resolve().parents[1] / "a1z_g05_deploy" / "client"
sys.path.insert(0, str(CLIENT_DIR))

from dual_arm_client import PARTS, pack, unpack  # noqa: E402  (wire format shared with the client)

import websockets  # noqa: E402

SHAPE_META = {"action": [{"key": name, "raw_shape": size} for name, size in PARTS.items()]}


def pose(step: int) -> dict:
    """Deterministic, slowly moving absolute target for one absolute step."""
    arm = 0.5 + 0.0005 * step
    return {
        "left_arm": np.full(6, arm, np.float64),
        "left_gripper": np.array([0.5], np.float64),
        "right_arm": np.full(6, arm + 0.01, np.float64),
        "right_gripper": np.array([0.5], np.float64),
    }


class EmulatedEnsemble:
    """Per-connection bookkeeping copied from EnsembleChunkedPolicyWrapper."""

    def __init__(self, *, action_steps: int, horizon: int, latency: float,
                 serve_chunk: bool, fail_at: int | None):
        self.action_steps = action_steps
        self.horizon = horizon
        self.latency = latency
        self.serve_chunk = serve_chunk
        self.fail_at = fail_at
        self.buf: dict[str, dict[int, list]] = {}
        self.gstep = 0
        self.last_recompute_step = 0
        self.steps_since_recompute = 0
        self.requests = 0
        self.recomputes = 0

    # ── buffer helpers ────────────────────────────────────────────────
    def _install(self, start: int) -> None:
        for offset in range(self.horizon):
            target = pose(start + offset)
            for part, value in target.items():
                self.buf.setdefault(part, {})[start + offset] = [np.asarray(value, np.float64)]

    def _mean_at(self, step: int) -> dict:
        served = {}
        for part in PARTS:
            preds = self.buf.get(part, {}).get(step)
            if not preds:
                raise ValueError(f"no prediction for {part} at step {step}")
            served[part] = np.stack(preds).mean(axis=0)
        return served

    def _window(self, start: int) -> dict:
        """Fused execution window (what the real server ships as action_chunk)."""
        window = {}
        for part in PARTS:
            rows = []
            for offset in range(self.action_steps):
                preds = self.buf.get(part, {}).get(start + offset)
                if not preds:
                    break
                rows.append(np.stack(preds).mean(axis=0))
            if rows:
                window[part] = np.stack(rows)
        return window

    def _prune(self, step: int) -> None:
        for part, ts_map in list(self.buf.items()):
            self.buf[part] = {ts: v for ts, v in ts_map.items() if ts >= step}
            if not self.buf[part]:
                self.buf.pop(part)

    # ── request handling ──────────────────────────────────────────────
    async def serve(self, raw_obs: dict) -> dict:
        self.requests += 1
        if self.fail_at is not None and self.requests == self.fail_at:
            raise RuntimeError(f"stub failure injected on request {self.requests}")

        current_step = raw_obs.pop("current_step", None)
        if current_step is not None:
            self.gstep = int(current_step)
            self.steps_since_recompute = self.gstep - self.last_recompute_step

        need_recompute = (not self.buf) or self.steps_since_recompute >= self.action_steps
        if need_recompute:
            await asyncio.sleep(self.latency)          # the real 670 ms GPU forward
            self._install(self.gstep)
            self.recomputes += 1
            self.steps_since_recompute = 0
            self.last_recompute_step = self.gstep

        chunk = self._window(self.gstep)
        action = self._mean_at(self.gstep)
        self._prune(self.gstep)
        self.gstep += 1
        self.steps_since_recompute += 1

        response = {
            "action": action,
            "need_obs": (not self.buf) or self.steps_since_recompute >= self.action_steps,
        }
        if self.serve_chunk:
            response["action_chunk"] = chunk
        return response


def make_handler(args, stats):
    async def handler(ws):
        state = EmulatedEnsemble(
            action_steps=args.action_steps,
            horizon=args.horizon,
            latency=args.latency,
            serve_chunk=not args.no_chunk,
            fail_at=args.fail_at,
        )
        await ws.send(pack({"action_steps": args.action_steps,
                            "shape_meta": SHAPE_META,
                            "ensemble": not args.no_chunk}))
        try:
            async for message in ws:
                raw_obs = unpack(message)
                try:
                    response = await state.serve(raw_obs)
                except Exception as exc:  # emulates the real 500 payload
                    await ws.send(pack({"error": {"code": 500, "message": str(exc)}}))
                    stats.append({"error": type(exc).__name__, "requests": state.requests})
                    continue
                stats.append({"requests": state.requests, "recomputes": state.recomputes,
                              "gstep": state.gstep})
                await ws.send(pack(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        if args.verbose:
            print(f"stub: client done, requests={state.requests} "
                  f"recomputes={state.recomputes}", flush=True)

    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--latency", type=float, default=0.67,
                        help="emulated inference latency per recompute, seconds")
    parser.add_argument("--action-steps", type=int, default=24,
                        help="steps served per recompute (server --action_steps)")
    parser.add_argument("--horizon", type=int, default=32,
                        help="model chunk length in steps")
    parser.add_argument("--no-chunk", action="store_true",
                        help="omit action_chunk, emulating serve_policy.py")
    parser.add_argument("--fail-at", type=int, default=None,
                        help="raise on the Nth request of the connection")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stats: list = []

    async def run():
        stop = asyncio.Event()
        async with websockets.serve(make_handler(args, stats), args.host, args.port,
                                    max_size=None):
            print(f"stub ready on ws://{args.host}:{args.port} "
                  f"(latency={args.latency}s, action_steps={args.action_steps}, "
                  f"chunk={'off' if args.no_chunk else 'on'})", flush=True)
            await stop.wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
