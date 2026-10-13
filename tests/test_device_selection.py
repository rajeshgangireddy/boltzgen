"""Hydra device selection at the native Lightning strategy and Trainer seam."""
# ruff: noqa: CPY001, INP001

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning.strategies import StrategyRegistry

from boltzgen.task.predict import predict as native_predict
from boltzgen.task.train import train as native_train
from boltzgen.utils import device


@pytest.mark.parametrize("backend", ["cuda", "xpu"])
def test_accelerator_seeding_keeps_lightning_worker_contract(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        device,
        "seed_everything",
        lambda seed, *, workers: calls.append((seed, workers)),
    )

    device.seed_everything_for_device(23, requested_accelerator=backend)

    assert calls == [(23, True)]


@pytest.mark.parametrize(
    ("backend", "index"), [("cuda", 0), ("cuda", 1), ("xpu", 0), ("xpu", 1)]
)
def test_resolver_uses_selected_hydra_device(
    monkeypatch: pytest.MonkeyPatch, backend: str, index: int
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        StrategyRegistry,
        "get",
        lambda name, **kwargs: (name, kwargs["device"]),
    )

    accelerator, strategy = device.resolve_trainer_kwargs(
        ListConfig([index]), requested_accelerator=backend
    )

    assert accelerator == backend
    if backend == "cuda":
        assert strategy == "auto"
    elif index:
        assert strategy == ("xpu_single", f"xpu:{index}")
    else:
        assert strategy == "xpu_single"


@pytest.mark.parametrize("backend", ["cuda", "xpu"])
@pytest.mark.parametrize("hydra_config", [False, True])
def test_prediction_passes_plain_hydra_devices_to_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    hydra_config: bool,
) -> None:
    received: dict = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        StrategyRegistry,
        "get",
        lambda name, **kwargs: (name, kwargs["device"]),
    )
    monkeypatch.setattr(
        native_predict.Boltz,
        "load_from_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(eval=lambda: None),
    )

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

    monkeypatch.setattr(native_predict, "Trainer", FakeTrainer)
    trainer = {"accelerator": backend, "devices": ListConfig([1])}
    task = native_predict.Predict(
        data=SimpleNamespace(predict_set=[object()]),
        writer=object(),
        checkpoint="unused.ckpt",
        output=str(tmp_path),
        name="prediction",
        recycling_steps=1,
        sampling_steps=1,
        trainer=OmegaConf.create(trainer) if hydra_config else trainer,
    )

    task.run(run_prediction=False)

    assert type(received["devices"]) is list
    assert received["devices"] == [1]
    assert received["accelerator"] == backend
    assert received["strategy"] == (
        "auto" if backend == "cuda" else ("xpu_single", "xpu:1")
    )


@pytest.mark.parametrize("backend", ["cuda", "xpu"])
@pytest.mark.parametrize("hydra_config", [False, True])
def test_training_passes_plain_hydra_devices_to_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    hydra_config: bool,
) -> None:
    received: dict = {}
    resolver_devices: list = []
    resolver_options: dict = {}
    monkeypatch.setattr(native_train, "TrainingDataModule", lambda _data: object())

    def capture_resolver(devices: list, **kwargs: object) -> tuple[str, str]:
        resolver_devices.append(devices)
        resolver_options.update(kwargs)
        return backend, "auto"

    monkeypatch.setattr(
        native_train,
        "resolve_trainer_kwargs",
        capture_resolver,
    )
    monkeypatch.setattr(
        native_train.pl,
        "Trainer",
        lambda **kwargs: (
            received.update(kwargs)
            or SimpleNamespace(fit=lambda *_args, **_kwargs: None)
        ),
    )
    trainer = {"accelerator": backend, "devices": ListConfig([1])}
    task = native_train.Training(
        data=native_train.DataConfig.__new__(native_train.DataConfig),
        model=object(),
        output=str(tmp_path),
        name="train",
        disable_checkpoint=True,
        trainer=OmegaConf.create(trainer) if hydra_config else trainer,
    )

    task.run(config=None)

    assert type(resolver_devices[0]) is list
    assert resolver_devices == [[1]]
    assert type(received["devices"]) is list
    assert received["devices"] == [1]
    assert received["accelerator"] == backend
    assert resolver_options["requested_accelerator"] == backend
