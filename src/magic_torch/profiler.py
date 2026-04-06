"""Lightweight step profiler using CUDA events (or CPU timers as fallback).

Enabled by setting `profile: true` in the YAML config or `MAGIC_PROFILE=1`.

Usage in hot paths:
    from .profiler import prof
    prof.start("sht_inverse")
    # ... work ...
    prof.stop("sht_inverse")

At the end of each step:
    prof.step_done()       # accumulates timings, prints summary every N steps

At the end of the run:
    prof.report()          # prints full summary

Handles repeated start/stop pairs for the same name within a step
(e.g., inside a chunk loop) by accumulating all pairs.
"""

import torch
from collections import defaultdict

from .params import l_profile
from .precision import DEVICE


class _Profiler:
    """CUDA-event profiler with per-step accumulation."""

    def __init__(self):
        self.enabled = l_profile
        self._use_cuda = DEVICE.type == "cuda"
        # List of (start, stop) event pairs per name, accumulated within a step
        self._pending = defaultdict(list)  # name -> [(start, stop), ...]
        self._open = {}  # name -> start event (for the current start/stop pair)
        self._accum = defaultdict(float)  # name -> total ms across all steps
        self._counts = defaultdict(int)
        self._step_count = 0
        self._warmup = 2  # skip first N steps (JIT warmup)
        self._report_every = 100

    def start(self, name: str):
        if not self.enabled:
            return
        if self._use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open[name] = ev
        else:
            import time
            self._open[name] = time.perf_counter()

    def stop(self, name: str):
        if not self.enabled:
            return
        if name not in self._open:
            return
        if self._use_cuda:
            ev_stop = torch.cuda.Event(enable_timing=True)
            ev_stop.record()
            self._pending[name].append((self._open.pop(name), ev_stop))
        else:
            import time
            self._pending[name].append((self._open.pop(name), time.perf_counter()))

    def step_done(self):
        """Call after each time step. Resolves pending events and accumulates."""
        if not self.enabled:
            return
        self._step_count += 1
        if self._step_count <= self._warmup:
            if self._use_cuda:
                torch.cuda.synchronize()
            self._pending.clear()
            self._open.clear()
            return

        if self._use_cuda:
            torch.cuda.synchronize()

        for name, pairs in self._pending.items():
            total_ms = 0.0
            for start, stop in pairs:
                if self._use_cuda:
                    total_ms += start.elapsed_time(stop)
                else:
                    total_ms += (stop - start) * 1000.0
            self._accum[name] += total_ms
            self._counts[name] += 1

        n = self._step_count - self._warmup
        if n > 0 and n % self._report_every == 0:
            self._print_summary(n)

        self._pending.clear()
        self._open.clear()

    def _print_summary(self, n_steps):
        """Print current timing summary."""
        print(f"\n=== Profile ({n_steps} steps, excluding {self._warmup} warmup) ===")
        # Total from top-level timers only (no '.' in name) to avoid double-counting
        top_level_ms = sum(v for k, v in self._accum.items() if '.' not in k)
        items = sorted(self._accum.items(), key=lambda x: -x[1])
        print(f"{'Component':<30} {'Total ms':>10} {'Avg ms/step':>12} {'%':>6} {'Calls':>6}")
        print("-" * 70)
        for name, total in items:
            avg = total / n_steps
            pct = 100.0 * total / top_level_ms if top_level_ms > 0 else 0
            calls = self._counts[name]
            avg_calls = calls / n_steps
            indent = "  " if '.' in name else ""
            print(f"{indent}{name:<28} {total:>10.1f} {avg:>12.3f} {pct:>5.1f}% {avg_calls:>6.1f}")
        print(f"{'TOTAL (top-level)':<30} {top_level_ms:>10.1f} {top_level_ms/n_steps:>12.3f}")
        print()

    def report(self):
        """Print final summary."""
        if not self.enabled:
            return
        n = self._step_count - self._warmup
        if n > 0:
            self._print_summary(n)


# Module-level singleton
prof = _Profiler()
