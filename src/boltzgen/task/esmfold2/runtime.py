"""Automatically provision ESMFold2 without changing BoltzGen's environment."""

import os
import subprocess
from pathlib import Path

from boltzgen.task.esmfold2.contract import ESM_VERSION


def resolve_python(
    python: str | None = None,
    *,
    require_cuda: bool = False,
    require_xpu: bool = False,
    require_cpu: bool = False,
) -> str:
    """Return a checked interpreter, provisioning a cached runtime when needed.

    uv owns dependency resolution, environment locking and Python installation.
    Its tool cache is independent of the active environment and current project.
    A caller-supplied interpreter remains available for managed/offline installs.
    """
    if sum((require_cuda, require_xpu, require_cpu)) > 1:
        raise ValueError("The ESMFold2 runtime can target only one accelerator")
    if python is None:
        from uv import find_uv_bin

        requirements = (
            Path(__file__).resolve().parents[2] / "resources/runtime/esmfold2.txt"
        )
        command = [
            find_uv_bin(),
            "tool",
            "run",
            "--isolated",
            "--no-config",
        ]
        if require_xpu:
            command.extend(["--torch-backend", "xpu"])
        elif require_cpu:
            command.extend(["--torch-backend", "cpu"])
        command.extend(
            [
                "--python",
                "3.12",
                "--from",
                f"esm=={ESM_VERSION}",
                "--with-requirements",
                str(requirements),
                "python",
            ]
        )
    probe = (
        "import sys; from importlib.metadata import version; "
        "assert sys.version_info >= (3, 12); "
        f"assert version('esm') == {ESM_VERSION!r}; "
        "import torch; from esm.models.esmfold2 import EsmFold2Model; "
    )
    if require_cuda:
        probe += (
            "assert torch.cuda.is_available(), "
            "'ESMFold2 requires an available CUDA device'; "
            "torch.empty(1, device='cuda').add_(1); torch.cuda.synchronize(); "
        )
    elif require_xpu:
        probe += (
            "assert hasattr(torch, 'xpu') and torch.xpu.is_available(), "
            "'ESMFold2 requires an available Intel XPU'; "
            "torch.empty(1, device='xpu').add_(1); torch.xpu.synchronize(); "
        )
    elif require_cpu:
        probe += "torch.empty(1, device='cpu').add_(1); "
    probe += "print(sys.executable)"
    try:
        if python is None:
            print("Checking the cached ESMFold2 runtime...", flush=True)
            # uv normally refreshes stale index entries even for a cached tool.
            # Try the complete cache first so subsequent runs work offline.
            discovery = ["-I", "-c", "import sys; print(sys.executable)"]
            cached = subprocess.run(
                [command[0], "--offline", *command[1:], *discovery],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if cached.returncode != 0:
                if os.environ.get("UV_OFFLINE", "").upper() in {
                    "1",
                    "ON",
                    "YES",
                    "TRUE",
                }:
                    raise RuntimeError(
                        "ESMFold2 runtime is not cached and UV_OFFLINE is set; "
                        "pre-cache the runtime or supply --esmfold2_python."
                    )
                print(
                    "Preparing the ESMFold2 runtime: downloading Python and dependencies "
                    "as needed (about 6 GB on first use). This may take several minutes. "
                    "The runtime is cached for later runs; UV_CACHE_DIR controls its location.",
                    flush=True,
                )
                cached = subprocess.run(
                    [*command, *discovery],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            python = cached.stdout.strip().splitlines()[-1]
        print(
            f"Checking ESMFold2 dependencies and device using {python}...", flush=True
        )
        # Validate separately: import/CUDA failures in an existing environment
        # must surface directly, never be mistaken for an installer cache miss.
        result = subprocess.run(
            [python, "-I", "-c", probe],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not prepare the ESMFold2 runtime. See the installer/import/device "
            "error above and check the interpreter, dependencies, and device backend. "
            "An uncached installation also needs network access and space in the uv "
            "cache. Managed installations may set --esmfold2_python."
        ) from exc
    resolved = result.stdout.strip().splitlines()[-1]
    print(f"ESMFold2 runtime ready: {resolved}", flush=True)
    return resolved


def worker_command(python: str, manifest: Path, device: str) -> list[str]:
    """Launch only our worker source in the isolated dependency environment."""
    return [
        python,
        "-I",
        str(Path(__file__).with_name("worker.py")),
        str(manifest),
        "--device",
        device,
    ]


def worker_probe_command(python: str, device: str) -> list[str]:
    return [
        python,
        "-I",
        str(Path(__file__).with_name("worker.py")),
        "--device",
        device,
        "--check-runtime",
    ]
