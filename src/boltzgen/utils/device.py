"""Hardware-agnostic device helpers built on ``torch.accelerator``.

This module centralizes the small amount of accelerator-vendor-specific logic
BoltzGen needs so the rest of the codebase can stay hardware-agnostic:

* Thin wrappers over ``torch.accelerator`` that never assume CUDA
  (``accelerator_type``, ``device_count``, ``device_capability_safe``,
  ``empty_cache``).
* A Lightning ``Accelerator``/``Strategy`` pair for Intel XPU, registered
  lazily (and only when ``torch.xpu`` is actually available) since
  PyTorch Lightning does not ship XPU support out of the box.
* ``resolve_trainer_kwargs`` which picks the right Lightning
  ``accelerator``/``strategy`` combination for the detected hardware, used by
  both the training and prediction entry points.

Only CUDA and XPU are actively supported; everything else falls back to CPU.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy, Strategy


def accelerator_type() -> str:
    """Return the current accelerator's device type string.

    Returns ``"cpu"`` when no accelerator (CUDA/XPU/MPS/...) is available.
    """
    acc = torch.accelerator.current_accelerator()
    return acc.type if acc is not None else "cpu"


def device_count() -> int:
    """Number of accelerator devices visible to this process (0 if none)."""
    if not torch.accelerator.is_available():
        return 0
    return torch.accelerator.device_count()


def device_capability_safe() -> Optional[tuple]:
    """Return ``(major, minor)`` compute capability, or ``None`` when not applicable.

    Compute capability is a CUDA-specific concept (used only to decide
    whether to enable the CUDA-only cuEquivariance kernels); on any other
    accelerator (or when nothing is available) this returns ``None`` instead
    of raising. Uses ``torch.cuda`` directly here since
    ``torch.accelerator.get_device_capability()`` does not support the CUDA
    backend in current PyTorch releases.
    """
    if accelerator_type() != "cuda":
        return None
    return torch.cuda.get_device_capability()


def empty_cache() -> None:
    """Release cached accelerator memory, if an accelerator is available."""
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()


def autocast_disabled():
    """Return an autocast-disabled context for the active accelerator."""
    return torch.autocast(device_type=accelerator_type(), enabled=False)


def xpu_precision_plugin(precision: Optional[str]) -> Optional[Any]:
    """Create Lightning AMP support for XPU precision modes."""
    if accelerator_type() != "xpu" or precision not in {"bf16-mixed", "16-mixed"}:
        return None

    from pytorch_lightning.plugins.precision import MixedPrecision

    return MixedPrecision(precision=precision, device="xpu")


def _register_xpu_lightning_support() -> None:
    """Register Lightning ``Accelerator``/``Strategy`` classes for Intel XPU.

    PyTorch Lightning (as of 2.6.x) only ships built-in accelerators for
    cpu/cuda/mps/tpu. This mirrors the small plugin pattern used by other
    Intel-GPU-enabled projects (e.g. Anomalib) to add XPU support without
    depending on Intel Extension for PyTorch, which is no longer required
    since its functionality was upstreamed into ``torch.xpu``.

    No-op (and no import cost) unless ``torch.xpu`` is actually available.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return

    from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry
    from pytorch_lightning.strategies import StrategyRegistry

    if "xpu" in AcceleratorRegistry:
        return  # Already registered (e.g. re-entrant import).

    class XPUAccelerator(Accelerator):
        """Lightning accelerator support for Intel XPU devices."""

        @property
        def name(self) -> str:
            return "xpu"

        @staticmethod
        def setup_device(device: torch.device) -> None:
            if device.type != "xpu":
                msg = f"Device should be xpu, got {device} instead"
                raise RuntimeError(msg)
            torch.xpu.set_device(device)

        @staticmethod
        def parse_devices(devices: Union[str, list, torch.device]) -> list:
            if isinstance(devices, list):
                return devices
            return [devices]

        @staticmethod
        def get_parallel_devices(devices: list) -> list:
            return [torch.device("xpu", idx) for idx in devices]

        @staticmethod
        def auto_device_count() -> int:
            return torch.xpu.device_count()

        @staticmethod
        def is_available() -> bool:
            return hasattr(torch, "xpu") and torch.xpu.is_available()

        @staticmethod
        def get_device_stats(device: Union[str, torch.device]) -> dict:
            del device  # Unused.
            return {}

        def teardown(self) -> None:
            """No extra teardown needed for XPU."""

    class SingleXPUStrategy(SingleDeviceStrategy):
        """Lightning strategy for training/inference on a single XPU device.

        Needed because the generic ``"auto"`` strategy resolves the root
        device to ``cpu`` for accelerators Lightning doesn't know natively;
        this strategy pins the device explicitly to ``xpu:<idx>``.
        """

        strategy_name = "xpu_single"

        def __init__(
            self,
            device: str = "xpu:0",
            accelerator: Optional[Accelerator] = None,
            checkpoint_io: Optional[Any] = None,
            precision_plugin: Optional[Any] = None,
        ) -> None:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                msg = "`SingleXPUStrategy` requires an available XPU device."
                raise RuntimeError(msg)
            super().__init__(
                accelerator=accelerator,
                device=device,
                checkpoint_io=checkpoint_io,
                precision_plugin=precision_plugin,
            )

    AcceleratorRegistry.register(
        XPUAccelerator().name,
        XPUAccelerator,
        description="Accelerator support for Intel XPU devices",
    )
    StrategyRegistry.register(
        SingleXPUStrategy.strategy_name,
        SingleXPUStrategy,
        description="Strategy for training/inference on a single Intel XPU device",
    )


_register_xpu_lightning_support()


def resolve_trainer_kwargs(
    devices: Union[int, list],
    *,
    ddp_kwargs: Optional[dict] = None,
) -> tuple:
    """Pick the Lightning ``accelerator``/``strategy`` for the current hardware.

    Parameters
    ----------
    devices:
        Number of devices (int) or explicit device list, as passed to
        ``pl.Trainer(devices=...)``.
    ddp_kwargs:
        Extra kwargs forwarded to ``DDPStrategy`` when multi-device CUDA
        training is requested (e.g. ``find_unused_parameters``, ``timeout``).

    Returns
    -------
    (accelerator, strategy):
        ``accelerator`` is a Lightning-recognized string (``"cuda"``,
        ``"xpu"``, or ``"auto"``). ``strategy`` is either a Lightning
        ``Strategy`` instance or the string ``"auto"``.

    Notes
    -----
    Multi-device XPU (distributed) training is not yet supported; only a
    single XPU device is used even if more are requested.
    """
    num_devices = len(devices) if isinstance(devices, list) else devices
    acc_type = accelerator_type()

    if acc_type == "cuda":
        strategy: Union[str, Strategy] = "auto"
        if num_devices > 1:
            strategy = DDPStrategy(**(ddp_kwargs or {}))
        return "cuda", strategy

    if acc_type == "xpu":
        return "xpu", "xpu_single"

    return "auto", "auto"
