"""Lightweight phase timing instrumentation.

This module provides a small, dependency-free utility for measuring how long
individual phases of the pipeline take (both CPU-bound stages such as
Analyze/Filter, and accelerator-bound stages such as the model forward pass),
and for persisting those measurements for later analysis (e.g. comparing
NVIDIA CUDA vs. Intel XPU vs. CPU runs).

Two output files are produced, both controlled by ``BOLTZGEN_TIMING_FILE``:

* ``<name>.jsonl`` -- one line per individual timed call. Useful for a deep
  dive into a specific run, but grows with the number of designs/batches.
* ``<name>.csv``   -- one row per ``(pipeline_step, phase)`` combination,
  aggregated (count/total/mean/min/max) in-memory and flushed once per
  process via :func:`flush_rollup`. This is the "top level" view: its size
  only depends on the number of distinct phases, not on how many designs
  were processed, so it stays small and readable no matter the run size.

Granularity
-----------
Every :class:`Timer` is tagged with a ``level``:

* ``"phase"``  (default) -- a business-meaningful phase (e.g. the diffusion
  sampling loop, the Pairformer trunk, a whole pipeline step). Always timed
  and recorded.
* ``"detail"`` -- a fine-grained internal step (e.g. input embedding init,
  the masker). Skipped by default (no sync, no ``perf_counter`` call, so
  effectively free) unless ``BOLTZGEN_TIMING_VERBOSITY=detailed`` is set, in
  which case it behaves exactly like a ``"phase"`` timer.

Usage
-----
CPU-only, business-level phase::

    with Timer("filter.total", gpu=False):
        run_all_filter_steps()

Accelerator phase (syncs the current accelerator before/after so the
measured duration reflects actual device completion, not just kernel
launch time)::

    with Timer("predict.diffusion_sampling", num_sampling_steps=200):
        struct_out = structure_module.sample(...)

Fine-grained internal step, hidden unless verbose::

    with Timer("forward.masker", gpu=False, level="detail"):
        feat_masked = self.masker(batch)

The output filename is controlled by the ``BOLTZGEN_TIMING_FILE`` env var
(see :func:`get_timing_file`/:func:`set_timing_file`), so different
experiments can be routed to different files (e.g. ``timings_a100`` vs
``timings_intel_gpu``) without any code changes. This env var, along with
``BOLTZGEN_PIPELINE_STEP`` (already set by the CLI to identify which
pipeline step -- design/inverse_folding/folding/... -- is currently
running), is inherited by pipeline subprocesses automatically and is used
to tag every record so phases from different pipeline steps aren't mixed
together.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

__all__ = [
    "DEFAULT_TIMING_FILENAME",
    "Timer",
    "get_timing_file",
    "set_timing_file",
    "record_timing",
    "flush_rollup",
    "print_timing_summary",
]

DEFAULT_TIMING_FILENAME = "timings.jsonl"
_ENV_TIMING_FILE = "BOLTZGEN_TIMING_FILE"
_ENV_VERBOSITY = "BOLTZGEN_TIMING_VERBOSITY"
_ENV_PIPELINE_STEP = "BOLTZGEN_PIPELINE_STEP"

_write_lock = threading.Lock()
_rollup_lock = threading.Lock()
# In-memory roll-up of durations, keyed by (pipeline_step, phase). Flushed
# (and cleared) via flush_rollup(), typically once at the end of each
# pipeline step's process.
_ROLLUP: dict[tuple[str, str], list[float]] = defaultdict(list)


def get_timing_file() -> Path:
    """Return the JSONL path timing records are appended to.

    Resolved from the ``BOLTZGEN_TIMING_FILE`` environment variable so the
    filename is changeable per-experiment while still using the same
    tracking mechanism. Defaults to ``timings.jsonl`` in the current
    working directory if unset. The companion roll-up CSV is always the
    same path with a ``.csv`` extension instead.
    """
    return Path(os.environ.get(_ENV_TIMING_FILE, DEFAULT_TIMING_FILENAME))


def set_timing_file(path: os.PathLike | str) -> None:
    """Set the timing file for this process (and any subprocess children).

    Since pipeline steps are frequently launched as subprocesses, setting
    this once (e.g. in the CLI entrypoint) propagates the chosen filename
    to every step and every DDP rank automatically via env inheritance.
    """
    os.environ[_ENV_TIMING_FILE] = str(path)


def _verbosity() -> str:
    return os.environ.get(_ENV_VERBOSITY, "phase")


def _is_active(level: str) -> bool:
    """Whether a timer of the given level should actually measure anything."""
    return level == "phase" or _verbosity() == "detailed"


def _pipeline_step() -> str:
    return os.environ.get(_ENV_PIPELINE_STEP, "")


def _rank() -> int:
    """Best-effort detection of the current distributed rank.

    Checks the environment variables set by torchrun / PyTorch Lightning.
    Returns 0 (and thus no filename suffix) when not running distributed.
    """
    for var in ("RANK", "GLOBAL_RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def _rank_suffixed_path(path: Path) -> Path:
    """Give each rank its own file so concurrent appends never interleave."""
    rank = _rank()
    if rank == 0:
        return path
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _sync() -> None:
    """Block until all queued accelerator work has completed.

    Uses ``torch.accelerator``, the device-agnostic API (torch >= 2.6,
    this repo requires torch >= 2.13), which synchronizes whichever
    accelerator is active (CUDA, Intel XPU, MPS, ...) without any
    per-backend branching.
    """
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def _device_type() -> str:
    """Return the current accelerator's device type, or 'cpu'."""
    if torch.accelerator.is_available():
        return str(torch.accelerator.current_accelerator())
    return "cpu"


def record_timing(phase: str, duration_s: float, level: str = "phase", **metadata: Any) -> dict:
    """Append a single raw record to the JSONL file and update the roll-up.

    Low-level entry point; prefer the :class:`Timer` context manager for
    normal use.
    """
    metadata.setdefault("pipeline_step", _pipeline_step())
    metadata["level"] = level
    record = {
        "phase": phase,
        "duration_s": duration_s,
        "timestamp": time.time(),
        "rank": _rank(),
        **metadata,
    }
    path = _rank_suffixed_path(get_timing_file())
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    with _write_lock:
        with open(path, "a") as f:
            f.write(line + "\n")

    with _rollup_lock:
        _ROLLUP[(metadata["pipeline_step"], phase)].append(duration_s)

    return record


class Timer(contextlib.ContextDecorator):
    """Context manager (and decorator) that times a block and logs it.

    Parameters
    ----------
    phase:
        Identifier for what is being timed, e.g. ``"predict.trunk"``.
    gpu:
        Whether this phase touches the accelerator. When True, the current
        accelerator is synchronized immediately before and after the timed
        block so the measured wall-clock time reflects actual device
        completion rather than just (async) kernel-launch time. Set False
        for purely CPU-bound phases (e.g. Analyze/Filter) to avoid a
        pointless sync call.
    level:
        ``"phase"`` (default) for business-meaningful phases, always timed
        and recorded. ``"detail"`` for fine-grained internals that are
        skipped entirely (no sync, no timing overhead) unless
        ``BOLTZGEN_TIMING_VERBOSITY=detailed`` is set.
    **metadata:
        Arbitrary extra fields (batch/sample id, recycling_steps,
        sampling_steps, diffusion_samples, use_kernels, ...) stored
        alongside the duration in the JSONL for later filtering.
    """

    def __init__(
        self, phase: str, gpu: bool = True, level: str = "phase", **metadata: Any
    ) -> None:
        self.phase = phase
        self.gpu = gpu
        self.level = level
        self.metadata = metadata
        self._active = _is_active(level)
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        if self._active:
            if self.gpu:
                _sync()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        if not self._active:
            return False
        if self.gpu:
            _sync()
        duration = time.perf_counter() - self._start
        metadata = dict(self.metadata)
        if self.gpu:
            metadata.setdefault("device", _device_type())
        record_timing(self.phase, duration, level=self.level, **metadata)
        return False


def flush_rollup(
    csv_path: Optional[os.PathLike | str] = None, print_summary: bool = True
) -> list[dict]:
    """Aggregate and persist the timings collected so far in this process.

    Appends one row per ``(pipeline_step, phase)`` -- with count, total,
    mean, min, and max duration -- to a small CSV file (by default the
    JSONL file's path with a ``.csv`` extension), then clears the
    in-memory roll-up. Call this once at the end of each task's ``run()``
    (Predict/Analyze/Filter) and at the end of the CLI's pipeline loop.

    This keeps the CSV's size proportional to the number of distinct
    phases rather than the number of designs/batches processed.
    """
    with _rollup_lock:
        items = list(_ROLLUP.items())
        _ROLLUP.clear()

    if not items:
        return []

    rows = []
    for (pipeline_step, phase), durations in sorted(
        items, key=lambda kv: -sum(kv[1])
    ):
        n = len(durations)
        total = sum(durations)
        rows.append(
            {
                "pipeline_step": pipeline_step,
                "phase": phase,
                "count": n,
                "total_s": round(total, 3),
                "mean_s": round(total / n, 3),
                "min_s": round(min(durations), 3),
                "max_s": round(max(durations), 3),
            }
        )

    resolved_csv = (
        Path(csv_path)
        if csv_path is not None
        else _rank_suffixed_path(get_timing_file()).with_suffix(".csv")
    )
    if resolved_csv.parent != Path(""):
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pipeline_step", "phase", "count", "total_s", "mean_s", "min_s", "max_s"]
    write_header = not resolved_csv.exists()
    with _write_lock:
        with open(resolved_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    if print_summary:
        _print_rows(rows)

    return rows


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n--- Timing summary ---")
    header = (
        f"{'pipeline_step':<16}{'phase':<32}{'count':>8}"
        f"{'total_s':>10}{'mean_s':>10}{'min_s':>9}{'max_s':>9}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['pipeline_step']:<16}{row['phase']:<32}{row['count']:>8}"
            f"{row['total_s']:>10.3f}{row['mean_s']:>10.3f}"
            f"{row['min_s']:>9.3f}{row['max_s']:>9.3f}"
        )


def print_timing_summary(csv_path: Optional[os.PathLike | str] = None) -> None:
    """Print the aggregated roll-up CSV (across every pipeline step so far).

    Unlike :func:`flush_rollup` (which reports only what happened in the
    *current* process), this reads the persisted CSV file back from disk,
    so it can be called at the very end of a whole ``boltzgen run`` to show
    a complete overview across all pipeline steps/subprocesses.
    """
    resolved = (
        Path(csv_path)
        if csv_path is not None
        else _rank_suffixed_path(get_timing_file()).with_suffix(".csv")
    )
    if not resolved.exists():
        return
    with open(resolved, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("count",):
            row[key] = int(row[key])
        for key in ("total_s", "mean_s", "min_s", "max_s"):
            row[key] = float(row[key])
    rows.sort(key=lambda r: -r["total_s"])
    _print_rows(rows)
