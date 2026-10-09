"""Direct, offline Python interface to the BoltzGen design pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import omegaconf
import pandas as pd
import torch
import yaml

from boltzgen._pipeline import (
    BinderDesignPipeline,
    config_dir,
    merge_directories,
    parse_additional_filters,
    parse_size_buckets,
    protocol_configs,
    run_task,
)
from boltzgen.data import const
from boltzgen.task.esmfold2.contract import (
    ESMC_REVISION,
    ESM_VERSION,
    MODEL_REVISION,
    REDESIGN_SCORE_KEY,
    SCORE_DIR,
    SCORE_KEY,
    fingerprint,
    load_result,
)
from boltzgen.utils.timing import flush_rollup, set_timing_file

__all__ = [
    "BoltzGenEngine",
    "PipelineRequest",
    "PipelinePlan",
    "PipelineRun",
    "StageResult",
    "ArtifactRef",
    "SelectedDesign",
    "PipelineValidationError",
    "PipelineResumeError",
    "PipelineStageError",
]

_MANIFEST = "pipeline-manifest.json"
_VERSION = 2
_FILTER_RUNS = "filter_runs"
_ACTIVE_RUN = threading.Lock()
_POLYMER = frozenset(protocol_configs) - {"protein-small_molecule"}
_DESIGN_FOLDING = {"protein-anything", "protein-small_molecule"}
_INT_OPTIONS = {
    "design": {"recycling_steps", "sampling_steps"},
    "inverse_folding": {"recycling_steps", "sampling_steps"},
    "folding": {"recycling_steps", "sampling_steps", "diffusion_samples"},
    "design_folding": {"recycling_steps", "sampling_steps", "diffusion_samples"},
    "affinity": {"recycling_steps", "sampling_steps", "diffusion_samples"},
    "esmfold2_scoring": {"num_loops", "sampling_steps", "diffusion_samples"},
    "filtering": {"top_budget", "num_liability_plots"},
}
_BOOL_OPTIONS = {
    "filtering": {"filter_bindingsite", "filter_cysteine", "filter_target_aligned"}
}
_CHOICE_OPTIONS = {
    "analysis": {"liability_peptide_type": {"linear", "cyclic"}},
    "filtering": {"peptide_type": {"linear", "cyclic"}},
}


class PipelineValidationError(ValueError):
    """The request cannot be run as specified."""


class PipelineResumeError(PipelineValidationError):
    """Saved inputs or stage artifacts no longer match the request."""


class PipelineStageError(RuntimeError):
    """A stage failed; completed earlier stages remain resumable."""

    def __init__(
        self, protocol: str, stage: str, cause: Exception, result: StageResult
    ) -> None:
        self.protocol = protocol
        self.stage = stage
        self.cause = cause
        self.result = result
        super().__init__(f"{protocol}/{stage}: {cause}")


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    design_spec: Path
    output_dir: Path
    protocol: str = "protein-anything"
    num_designs: int = 10000
    budget: int = 30
    device: str = "cpu"
    precision: str | None = None
    seed: int | None = None
    design_checkpoints: tuple[Path, ...] = ()
    folding_checkpoint: Path | None = None
    solublempnn_checkpoint: Path | None = None
    boltzif_checkpoint: Path | None = None
    affinity_checkpoint: Path | None = None
    moldir: Path | None = None
    esmfold2_python: Path | None = None
    inverse_fold_model: str | None = None
    skip_inverse_folding: bool = False
    only_inverse_fold: bool = False
    diffusion_batch_size: int | None = None
    step_scale: float | None = None
    noise_scale: float | None = None
    inverse_fold_num_sequences: int = 1
    inverse_fold_avoid: str | None = None
    solublempnn_sampling_temperature: float = 0.1
    esmfold2_acceleration: str = "auto"
    use_kernels: str = "auto"
    num_workers: int = 1
    analysis_processes: int = 1
    alpha: float | None = None
    filter_biased: bool = True
    refolding_rmsd_threshold: float | None = None
    metrics_override: Mapping[str, float | None] = field(default_factory=dict)
    additional_filters: tuple[str, ...] = ()
    size_buckets: tuple[str, ...] = ()
    step_options: Mapping[
        str, Mapping[str, bool | int | float | str | tuple[str, ...]]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "design_spec",
            "output_dir",
            "moldir",
            "folding_checkpoint",
            "solublempnn_checkpoint",
            "boltzif_checkpoint",
            "affinity_checkpoint",
            "esmfold2_python",
        ):
            value = getattr(self, name)
            if value is not None:
                path = Path(value).expanduser()
                object.__setattr__(
                    self,
                    name,
                    path.absolute() if name == "esmfold2_python" else path.resolve(),
                )
        object.__setattr__(
            self,
            "design_checkpoints",
            tuple(
                Path(path).expanduser().resolve() for path in self.design_checkpoints
            ),
        )
        object.__setattr__(self, "additional_filters", tuple(self.additional_filters))
        object.__setattr__(self, "size_buckets", tuple(self.size_buckets))
        object.__setattr__(
            self, "metrics_override", MappingProxyType(dict(self.metrics_override))
        )
        object.__setattr__(
            self,
            "step_options",
            MappingProxyType(
                {
                    stage: MappingProxyType(dict(options))
                    for stage, options in self.step_options.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    request: PipelineRequest
    stages: tuple[str, ...]
    input_files: tuple[ArtifactRef, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    name: str
    path: Path
    sha256: str
    source_id: str | None = None
    source_run: Path | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    status: str
    files: tuple[ArtifactRef, ...]
    elapsed_seconds: float
    device: str
    design_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedDesign:
    id: str
    source_id: str
    source_run: Path
    cif: Path
    rank: int
    before_refolding_cif: Path | None = None
    esmfold2_cif: Path | None = None
    esmfold2_json: Path | None = None
    passes_filters: bool = False
    affinity_npz: Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineRun:
    stages: tuple[StageResult, ...]
    resume_handle: Path
    output_dir: Path
    completed: bool
    final_csv: Path | None
    selected: tuple[SelectedDesign, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                digest.update(_sha256(child).encode())
        return digest.hexdigest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    git = root / ".git"
    if not git.exists():
        try:
            receipt = distribution("boltzgen").read_text("direct_url.json")
        except PackageNotFoundError:
            receipt = None
        if receipt is not None:
            try:
                revision = json.loads(receipt)["vcs_info"]["commit_id"]
            except (KeyError, TypeError, ValueError):
                revision = None
            if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                return revision.lower()
        return "installed-package"
    if git.is_file():
        git = Path(git.read_text().strip().split("gitdir: ", 1)[1])
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    common = (
        (git / (git / "commondir").read_text().strip()).resolve()
        if (git / "commondir").exists()
        else git
    )
    ref_path = common / ref
    return ref_path.read_text().strip() if ref_path.is_file() else "installed-package"


def _implementation_hash() -> str:
    package = Path(__file__).resolve().parent
    sources = sorted(package.rglob("*.py")) + sorted(
        path for path in (package / "resources").rglob("*") if path.is_file()
    )
    return _digest({str(p.relative_to(package)): _sha256(p) for p in sources})


def _device(request: PipelineRequest, *, available: bool = False) -> tuple[str, int]:
    if not isinstance(request.device, str):
        raise PipelineValidationError("device must be cpu, cuda[:index] or xpu[:index]")
    match = re.fullmatch(r"(cpu|cuda|xpu)(?::(\d+))?", request.device)
    if match is None:
        raise PipelineValidationError(
            "device must be cpu, cuda[:index] or xpu[:index]; multiple devices are unsupported"
        )
    backend, number = match.groups()
    index = int(number or 0)
    if backend == "cpu" and number is not None:
        raise PipelineValidationError("CPU accepts only the single device cpu")
    if available and backend != "cpu":
        accelerator = torch.cuda if backend == "cuda" else getattr(torch, "xpu", None)
        if (
            accelerator is None
            or not accelerator.is_available()
            or index >= accelerator.device_count()
        ):
            raise PipelineValidationError(
                f"Selected device {request.device} is unavailable; no CPU fallback is permitted"
            )
    return backend, index


def _graph(request: PipelineRequest) -> tuple[str, ...]:
    if request.protocol not in protocol_configs:
        raise PipelineValidationError(f"Unknown protocol {request.protocol!r}")
    if request.skip_inverse_folding and request.only_inverse_fold:
        raise PipelineValidationError(
            "only_inverse_fold cannot be combined with skip_inverse_folding"
        )
    if (
        request.protocol == "protein-small_molecule"
        and request.inverse_fold_model == "solublempnn"
    ):
        raise PipelineValidationError(
            "protein-small_molecule requires ligand-conditioned BoltzIF"
        )
    graph = []
    if not request.only_inverse_fold:
        graph.append("design")
    if not request.skip_inverse_folding:
        graph.append("inverse_folding")
    graph.append("folding")
    if request.protocol in _DESIGN_FOLDING:
        graph.append("design_folding")
    graph.append("esmfold2_scoring" if request.protocol in _POLYMER else "affinity")
    return (*graph, "analysis", "filtering")


def _check_request(request: PipelineRequest, graph: tuple[str, ...]) -> None:
    _device(request)
    for name in (
        "design_spec",
        "output_dir",
        "moldir",
        "folding_checkpoint",
        "solublempnn_checkpoint",
        "boltzif_checkpoint",
        "affinity_checkpoint",
        "esmfold2_python",
    ):
        path = getattr(request, name)
        if path is not None and (
            "${" in str(path) or any(char in str(path) for char in "\n\r\0")
        ):
            raise PipelineValidationError(
                f"{name}: Hydra path interpolation is forbidden"
            )
    if any(
        "${" in str(path) or any(char in str(path) for char in "\n\r\0")
        for path in request.design_checkpoints
    ):
        raise PipelineValidationError(
            "design_checkpoints: Hydra path interpolation is forbidden"
        )
    for name in ("skip_inverse_folding", "only_inverse_fold", "filter_biased"):
        if type(getattr(request, name)) is not bool:
            raise PipelineValidationError(f"{name} must be a boolean")
    for name in (
        "num_designs",
        "budget",
        "inverse_fold_num_sequences",
        "num_workers",
        "analysis_processes",
    ):
        value = getattr(request, name)
        if type(value) is not int or value < (0 if name == "num_workers" else 1):
            raise PipelineValidationError(
                f"{name} must be a positive integer"
                if name != "num_workers"
                else "num_workers must be nonnegative"
            )
    if request.diffusion_batch_size is not None and (
        type(request.diffusion_batch_size) is not int
        or request.diffusion_batch_size < 1
    ):
        raise PipelineValidationError("diffusion_batch_size must be positive")
    for name in ("step_scale", "noise_scale"):
        value = getattr(request, name)
        if value is not None and (
            type(value) not in (float, int) or not math.isfinite(value) or value <= 0
        ):
            raise PipelineValidationError(f"{name} must be finite and positive")
    if request.seed is not None and (
        type(request.seed) is not int or not 0 <= request.seed < 2**32
    ):
        raise PipelineValidationError("seed must be an unsigned 32-bit integer")
    if request.precision not in (None, "32", "32-true", "bf16-mixed", "16-mixed"):
        raise PipelineValidationError(
            "precision must be 32, 32-true, bf16-mixed or 16-mixed"
        )
    if request.precision == "16-mixed" and _device(request)[0] == "cpu":
        raise PipelineValidationError(
            "CPU cannot run 16-mixed precision without a silent bf16 fallback"
        )
    if request.use_kernels not in ("auto", "true", "false"):
        raise PipelineValidationError("use_kernels must be auto, true or false")
    if request.use_kernels == "true" and _device(request)[0] != "cuda":
        raise PipelineValidationError(
            "cuEquivariance kernels require an explicit CUDA device"
        )
    if request.esmfold2_acceleration not in ("auto", "fused", "off"):
        raise PipelineValidationError(
            "esmfold2_acceleration must be auto, fused or off"
        )
    if request.esmfold2_acceleration == "fused" and _device(request)[0] != "cuda":
        raise PipelineValidationError("Fused ESMFold2 acceleration requires CUDA")
    if request.inverse_fold_model not in (None, "boltzif", "solublempnn"):
        raise PipelineValidationError(
            "inverse_fold_model must be boltzif or solublempnn"
        )
    if request.alpha is not None and (
        type(request.alpha) not in (float, int)
        or not math.isfinite(request.alpha)
        or not 0 <= request.alpha <= 1
    ):
        raise PipelineValidationError("alpha must be finite and between 0 and 1")
    if request.refolding_rmsd_threshold is not None and (
        type(request.refolding_rmsd_threshold) not in (float, int)
        or not math.isfinite(request.refolding_rmsd_threshold)
        or request.refolding_rmsd_threshold < 0
    ):
        raise PipelineValidationError(
            "refolding_rmsd_threshold must be nonnegative and finite"
        )
    if (
        type(request.solublempnn_sampling_temperature) not in (float, int)
        or not math.isfinite(request.solublempnn_sampling_temperature)
        or request.solublempnn_sampling_temperature <= 0
    ):
        raise PipelineValidationError(
            "solublempnn_sampling_temperature must be finite and positive"
        )
    if request.inverse_fold_avoid is not None and (
        not isinstance(request.inverse_fold_avoid, str)
        or not set(request.inverse_fold_avoid) < set(const.prot_letter_to_token)
    ):
        raise PipelineValidationError(
            "inverse_fold_avoid must contain canonical amino-acid letters and leave one allowed"
        )
    if request.moldir is None or not (
        request.moldir.is_dir() or request.moldir.is_file()
    ):
        raise PipelineValidationError(
            "moldir must name an existing local molecule directory or zip"
        )
    for name, weight in request.metrics_override.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
            or (
                weight is not None
                and (
                    type(weight) not in (float, int)
                    or not math.isfinite(weight)
                    or weight <= 0
                )
            )
        ):
            raise PipelineValidationError(f"Invalid filtering metric/weight: {name}")
    if any(not isinstance(item, str) for item in request.additional_filters):
        raise PipelineValidationError(
            "additional_filters must be threshold expressions"
        )
    for item in parse_additional_filters(request.additional_filters) or ():
        if not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]*", item["feature"]
        ) or not math.isfinite(item["threshold"]):
            raise PipelineValidationError(
                "additional_filters require finite named metrics"
            )
    if any(not isinstance(item, str) for item in request.size_buckets):
        raise PipelineValidationError("size_buckets must be min-max:count expressions")
    for item in parse_size_buckets(request.size_buckets) or ():
        if item["min"] < 0 or item["min"] > item["max"] or item["num_designs"] < 1:
            raise PipelineValidationError(
                "size_buckets require valid ranges and positive counts"
            )
    inverse_fold_model = request.inverse_fold_model or (
        "boltzif" if request.protocol == "protein-small_molecule" else "solublempnn"
    )
    for stage, options in request.step_options.items():
        if stage not in graph:
            raise PipelineValidationError(
                f"{stage}: stage is not in the {request.protocol} graph"
            )
        for key, value in options.items():
            if not isinstance(key, str):
                raise PipelineValidationError(f"{stage}: option names must be strings")
            if key in _INT_OPTIONS.get(stage, ()):
                if type(value) is not int or value < (
                    0 if key == "recycling_steps" or key == "num_liability_plots" else 1
                ):
                    raise PipelineValidationError(
                        f"{stage}.{key} must be a nonnegative integer"
                    )
                if stage == "inverse_folding" and inverse_fold_model == "solublempnn":
                    raise PipelineValidationError(
                        "SolubleMPNN does not accept recycling/sampling step options"
                    )
            elif key in _BOOL_OPTIONS.get(stage, ()):
                if type(value) is not bool:
                    raise PipelineValidationError(f"{stage}.{key} must be a boolean")
            elif key in _CHOICE_OPTIONS.get(stage, {}):
                if (
                    not isinstance(value, str)
                    or value not in _CHOICE_OPTIONS[stage][key]
                ):
                    raise PipelineValidationError(f"Invalid {stage}.{key}: {value!r}")
            elif stage == "esmfold2_scoring" and key == "lm_dropout":
                if (
                    type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise PipelineValidationError(
                        "esmfold2_scoring.lm_dropout must be in [0, 1]"
                    )
            elif stage == "esmfold2_scoring" and key == "scoring_target_chains":
                if (
                    not isinstance(value, tuple)
                    or not value
                    or any(
                        not isinstance(chain, str)
                        or not re.fullmatch(r"[A-Za-z0-9]+", chain)
                        for chain in value
                    )
                    or len(set(value)) != len(value)
                    or request.protocol == "protein-redesign"
                ):
                    raise PipelineValidationError(
                        "scoring_target_chains requires distinct target polymer chains and binder scoring"
                    )
            else:
                raise PipelineValidationError(
                    f"Unreviewed step option {stage}.{key}; raw Hydra overrides are forbidden"
                )


def _spec_inputs(path: Path) -> dict[str, str]:
    files: dict[str, str] = {}

    def record(file: Path) -> None:
        file = file.expanduser().resolve()
        if not file.is_file():
            raise PipelineValidationError(f"Missing local design input: {file}")
        if str(file) in files:
            return
        files[str(file)] = _sha256(file)
        if file.suffix != ".yaml":
            return
        data = yaml.safe_load(file.read_text())

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if "file" in value and isinstance(value["file"], dict):
                    nested = value["file"]
                    for name in (
                        nested["path"]
                        if isinstance(nested.get("path"), list)
                        else [nested.get("path")]
                    ):
                        if not isinstance(name, str):
                            raise PipelineValidationError(
                                f"Invalid file.path in {file}"
                            )
                        record(file.parent / name)
                elif "path" in value and "entities" not in value:
                    names = (
                        value["path"]
                        if isinstance(value["path"], list)
                        else [value["path"]]
                    )
                    for name in names:
                        if not isinstance(name, str):
                            raise PipelineValidationError(f"Invalid path in {file}")
                        record(file.parent / name)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(data)

    record(path)
    return dict(sorted(files.items()))


def _settings(request: PipelineRequest, stage: str) -> str:
    common = {
        "stage": stage,
        "options": dict(request.step_options.get(stage, {})),
        "num_workers": request.num_workers,
    }
    if stage == "design":
        common.update(
            num_designs=request.num_designs,
            diffusion_batch_size=request.diffusion_batch_size,
            step_scale=request.step_scale,
            noise_scale=request.noise_scale,
            checkpoints=tuple(map(str, request.design_checkpoints)),
            use_kernels=request.use_kernels,
        )
    elif stage == "inverse_folding":
        common.update(
            model=request.inverse_fold_model
            or (
                "boltzif"
                if request.protocol == "protein-small_molecule"
                else "solublempnn"
            ),
            sequences=request.inverse_fold_num_sequences,
            avoid=request.inverse_fold_avoid,
            temperature=request.solublempnn_sampling_temperature,
            checkpoint=str(
                request.boltzif_checkpoint
                if (
                    request.inverse_fold_model == "boltzif"
                    or request.protocol == "protein-small_molecule"
                )
                else request.solublempnn_checkpoint
            ),
        )
    elif stage in ("folding", "design_folding"):
        common.update(
            checkpoint=str(request.folding_checkpoint), use_kernels=request.use_kernels
        )
    elif stage == "affinity":
        common.update(
            checkpoint=str(request.affinity_checkpoint), use_kernels=request.use_kernels
        )
    elif stage == "esmfold2_scoring":
        common.update(
            python=str(request.esmfold2_python),
            acceleration=request.esmfold2_acceleration,
            model_revision=MODEL_REVISION,
            esmc_revision=ESMC_REVISION,
            esm_version=ESM_VERSION,
        )
    elif stage == "analysis":
        common["analysis_processes"] = request.analysis_processes
    elif stage == "filtering":
        common.update(
            budget=request.budget,
            alpha=request.alpha,
            filter_biased=request.filter_biased,
            refolding_rmsd_threshold=request.refolding_rmsd_threshold,
            metrics_override=dict(request.metrics_override),
            additional_filters=request.additional_filters,
            size_buckets=request.size_buckets,
        )
    return _digest(common)


def _dependencies(graph: tuple[str, ...], stage: str) -> tuple[str, ...]:
    if stage == "design" or stage == graph[0]:
        return ()
    if stage == "inverse_folding":
        return ("design",)
    if stage == "folding":
        return ("inverse_folding",) if "inverse_folding" in graph else ("design",)
    if stage == "design_folding":
        return ("folding",)
    if stage in ("affinity", "esmfold2_scoring"):
        return ("folding",) + (("design_folding",) if stage == "affinity" else ())
    if stage == "analysis":
        return tuple(
            name
            for name in graph
            if name in ("folding", "design_folding", "affinity", "esmfold2_scoring")
        )
    return ("analysis",)


def _local_path(
    value: Path | str | None, label: str, *, preserve_symlink: bool = False
) -> Path:
    if value is None or str(value).startswith(("huggingface:", "http:", "https:")):
        raise PipelineValidationError(
            f"{label}: supply an explicit local file; implicit downloads are disabled"
        )
    path = Path(value).expanduser().absolute()
    if not preserve_symlink:
        path = path.resolve()
    if not path.is_file():
        raise PipelineValidationError(
            f"{label}: missing local file {path}; no download or fallback is attempted"
        )
    return path


def _esm_assets(request: PipelineRequest) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    from boltzgen.task.esmfold2.contract import ESMC_REPO, MODEL_REPO
    from boltzgen.task.esmfold2.runtime import worker_probe_command

    python = _local_path(
        request.esmfold2_python,
        "esmfold2_scoring.esmfold2_python",
        preserve_symlink=True,
    )
    backend, index = _device(request, available=True)
    worker_device = "cpu" if backend == "cpu" else f"{backend}:{index}"
    try:
        ccd = Path(
            hf_hub_download(
                MODEL_REPO, "ccd.pkl", revision=MODEL_REVISION, local_files_only=True
            )
        )
        model = Path(
            snapshot_download(
                MODEL_REPO,
                revision=MODEL_REVISION,
                allow_patterns=["*.json", "*.safetensors"],
                local_files_only=True,
            )
        )
        esmc = Path(
            snapshot_download(
                ESMC_REPO,
                revision=ESMC_REVISION,
                allow_patterns=["*.json", "*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise PipelineValidationError(
            "esmfold2_scoring: provision the pinned ESMFold2/ESMC checkpoints and CCD "
            "in the local HF cache before running offline"
        ) from exc
    if not all(path.is_file() for path in (ccd,)) or not all(
        any(folder.rglob("*.safetensors")) for folder in (model, esmc)
    ):
        raise PipelineValidationError(
            "esmfold2_scoring: incomplete pinned local ESM model assets"
        )
    try:
        subprocess.run(
            worker_probe_command(str(python), worker_device),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            (exc.stderr or "")
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise PipelineValidationError(
            "esmfold2_scoring: the isolated ESMFold2 worker needs Python 3.12, "
            f"esm=={ESM_VERSION}, and the selected device: {detail[-800:]}"
        ) from exc
    return {
        "esmfold2_python": python,
        "esmfold2_ccd": ccd,
        "esmfold2_model": model,
        "esmc_model": esmc,
    }


def _stage_assets(request: PipelineRequest, stage: str) -> dict[str, Path]:
    if stage == "design":
        if not request.design_checkpoints:
            raise PipelineValidationError(
                "design: at least one local design checkpoint is required"
            )
        return {
            f"design_checkpoints.{i}": _local_path(path, f"design checkpoint {i}")
            for i, path in enumerate(request.design_checkpoints)
        }
    if stage == "inverse_folding":
        model = request.inverse_fold_model or (
            "boltzif" if request.protocol == "protein-small_molecule" else "solublempnn"
        )
        key = "boltzif_checkpoint" if model == "boltzif" else "solublempnn_checkpoint"
        return {key: _local_path(getattr(request, key), f"inverse_folding.{key}")}
    if stage in ("folding", "design_folding"):
        return {
            "folding_checkpoint": _local_path(
                request.folding_checkpoint, f"{stage}.folding_checkpoint"
            )
        }
    if stage == "affinity":
        return {
            "affinity_checkpoint": _local_path(
                request.affinity_checkpoint, "affinity.affinity_checkpoint"
            )
        }
    if stage == "esmfold2_scoring":
        return _esm_assets(request)
    return {}


def _option_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, tuple)):
        return json.dumps(value if isinstance(value, str) else list(value))
    return str(value)


def _recipe(
    request: PipelineRequest, selected: Sequence[str]
) -> dict[str, omegaconf.DictConfig]:
    needs_accelerator = any(
        stage not in ("analysis", "filtering") for stage in selected
    )
    backend, index = _device(request, available=needs_accelerator)
    overrides: dict[str, list[str]] = {stage: [] for stage in selected}
    for stage in selected:
        for key, value in request.step_options.get(stage, {}).items():
            overrides[stage].append(f"{key}={_option_value(value)}")
        if stage in (
            "design",
            "inverse_folding",
            "folding",
            "design_folding",
            "affinity",
        ):
            overrides[stage].append(f"trainer.accelerator={backend}")
            if index:
                overrides[stage].append(f"trainer.devices=[{index}]")
            if request.precision is not None:
                overrides[stage].append(f"trainer.precision={request.precision}")
        if stage == "analysis":
            overrides[stage].append(f"num_processes={request.analysis_processes}")
        if stage == "esmfold2_scoring":
            overrides[stage].extend((f"device_type={backend}", f"device_index={index}"))
    args = SimpleNamespace(
        protocol=request.protocol,
        only_inverse_fold=request.only_inverse_fold,
        skip_inverse_folding=request.skip_inverse_folding,
        inverse_fold_model=request.inverse_fold_model,
        solublempnn_sampling_temperature=request.solublempnn_sampling_temperature,
        inverse_fold_avoid=request.inverse_fold_avoid,
        config=[[stage, *opts] for stage, opts in overrides.items() if opts],
        seed=request.seed,
        devices=1,
        device=backend,
        device_index=index,
        use_kernels="false"
        if backend != "cuda" and request.use_kernels == "auto"
        else request.use_kernels,
        output=request.output_dir,
        diffusion_batch_size=request.diffusion_batch_size,
        num_designs=request.num_designs,
        design_checkpoints=request.design_checkpoints,
        step_scale=request.step_scale,
        noise_scale=request.noise_scale,
        inverse_fold_num_sequences=request.inverse_fold_num_sequences,
        solublempnn_checkpoint=request.solublempnn_checkpoint,
        inverse_fold_checkpoint=request.boltzif_checkpoint,
        folding_checkpoint=request.folding_checkpoint,
        affinity_checkpoint=request.affinity_checkpoint,
        design_spec=[request.design_spec],
        config_dir=config_dir,
        num_workers=request.num_workers,
        reuse=False,
        esmfold2_python=str(request.esmfold2_python)
        if request.esmfold2_python
        else None,
        esmfold2_acceleration=request.esmfold2_acceleration,
        budget=request.budget,
        alpha=request.alpha,
        filter_biased=str(request.filter_biased).lower(),
        metrics_override=[
            f"{key}={value if value is not None else 'none'}"
            for key, value in request.metrics_override.items()
        ]
        or None,
        additional_filters=list(request.additional_filters) or None,
        size_buckets=list(request.size_buckets) or None,
        refolding_rmsd_threshold=request.refolding_rmsd_threshold,
    )
    pipeline = BinderDesignPipeline(
        args,
        request.moldir,
        selected_steps=set(selected),
        resolve_artifact=lambda _args, path: _local_path(path, "pipeline checkpoint"),
    )
    configs = {step.name: step.get_config() for step in pipeline.steps}
    if set(configs) != set(selected):
        raise PipelineValidationError(
            f"Protocol {request.protocol}: unavailable stages {set(selected) - set(configs)}"
        )
    return configs


def _design_dir(request: PipelineRequest) -> Path:
    return request.output_dir / (
        "intermediate_designs"
        if request.only_inverse_fold or request.skip_inverse_folding
        else "intermediate_designs_inverse_folded"
    )


def _candidate_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.glob("*.cif"))
        if path.is_file() and not path.stem.endswith("_native")
    }


def _filter_run_dir(root: Path, revision: str | None) -> Path:
    if not isinstance(revision, str) or not re.fullmatch(r"v\d{4,}", revision):
        raise PipelineResumeError(f"Invalid filtering revision: {revision!r}")
    return root / _FILTER_RUNS / revision


def _config_path(root: Path, stage: str, revision: str | None = None) -> Path:
    if stage == "filtering":
        return _filter_run_dir(root, revision) / "filtering.yaml"
    return root / "config" / f"{stage}.yaml"


def _stage_paths(
    request: PipelineRequest,
    stage: str,
    *,
    merged: bool = False,
    revision: str | None = None,
) -> tuple[Path, ...]:
    root = request.output_dir
    base = _design_dir(request)
    if stage == "design":
        base = root / "intermediate_designs"
    if stage in ("design", "inverse_folding"):
        if not base.is_dir():
            return ()
        return tuple(
            sorted(
                (
                    path
                    for path in base.iterdir()
                    if path.is_file() and path.suffix in (".cif", ".npz")
                ),
            )
        ) + tuple(sorted((base / const.molecules_dirname).glob("*.pkl")))
    if stage in ("folding", "design_folding"):
        pair = (
            (const.folding_dirname, const.refold_cif_dirname)
            if stage == "folding"
            else (const.folding_design_dirname, const.refold_design_cif_dirname)
        )
        return tuple(
            sorted(
                path
                for folder in pair
                for path in (base / folder).glob("*")
                if path.is_file()
            )
        )
    if stage == "affinity":
        return tuple(sorted((base / const.affinity_dirname).glob("*.npz")))
    if stage == "esmfold2_scoring":
        return tuple(
            sorted(path for path in (base / SCORE_DIR).glob("*") if path.is_file())
        )
    if stage == "analysis":
        output = [
            base / "aggregate_metrics_analyze.csv",
            base / "ca_coords_sequences.pkl.gz",
        ]
        output += list((base / const.metrics_dirname).glob("*.npz"))
        output += list(base.glob("per_target_metrics_analyze.csv"))
        if merged:
            output += list(base.glob("*.cif")) + list(base.glob("*.npz"))
            output += list((base / const.molecules_dirname).glob("*.pkl"))
            output += list((base / const.refold_cif_dirname).glob("*.cif"))
            output += list((base / const.affinity_dirname).glob("*.npz"))
            output += list((base / SCORE_DIR).glob("*"))
        return tuple(sorted(path for path in output if path.is_file()))
    return tuple(
        sorted(
            path
            for path in (
                _filter_run_dir(root, revision) / "final_ranked_designs"
            ).rglob("*")
            if path.is_file()
        )
    )


def _safe_relative(root: Path, path: Path) -> str:
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise PipelineResumeError(f"Artifact escapes the run directory: {path}")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise PipelineResumeError(
                f"Symlinked run artifact is not safe to reuse: {path}"
            )
    return str(path.relative_to(root))


def _artifact_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise PipelineResumeError(f"Unsafe artifact reference: {relative!r}")
    path = root / relative
    _safe_relative(root, path)
    return path


def _file_ref(root: Path, path: Path, sources: Mapping[str, Any]) -> ArtifactRef:
    relative = _safe_relative(root, path)
    stem = path.name.removesuffix(".input.json")
    if stem == path.name:
        stem = path.stem
    if path.parent.name.startswith(("final_", "intermediate_ranked_")):
        stem = re.sub(r"^rank\d+_", "", stem)
    if path.suffix not in (".cif", ".npz", ".json", ".pkl"):
        stem = ""
    lineage = sources.get(stem)
    return ArtifactRef(
        name=relative,
        path=path,
        sha256=_sha256(path),
        source_id=lineage["source_id"] if lineage else stem or None,
        source_run=Path(lineage["source_run"]) if lineage else root if stem else None,
    )


def _stage_results(
    manifest: dict, root: Path, graph: tuple[str, ...]
) -> tuple[StageResult, ...]:
    results = []
    for name in graph:
        receipt = manifest["stage_receipts"].get(name)
        if receipt is None:
            continue
        files = tuple(
            ArtifactRef(
                name=file["name"],
                path=_artifact_path(root, file["name"]),
                sha256=file["sha256"],
                source_id=file.get("source_id"),
                source_run=Path(file["source_run"]) if file.get("source_run") else None,
            )
            for file in receipt["files"]
        )
        results.append(
            StageResult(
                name=name,
                status="merged" if receipt.get("merged") else "completed",
                files=files,
                elapsed_seconds=receipt["elapsed_seconds"],
                device=receipt["device"],
                design_ids=tuple(receipt["design_ids"]),
            )
        )
    return tuple(results)


def _validate_outputs(
    request: PipelineRequest,
    stage: str,
    *,
    merged: bool = False,
    revision: str | None = None,
) -> tuple[str, ...]:
    base = _design_dir(request)
    candidates = _candidate_files(base)
    ids = set(candidates)
    if stage == "design":
        raw = _candidate_files(request.output_dir / "intermediate_designs")
        pairs = {
            p.stem for p in (request.output_dir / "intermediate_designs").glob("*.npz")
        }
        if len(raw) < request.num_designs or set(raw) != pairs:
            raise PipelineValidationError(
                "design: incomplete CIF/NPZ pairs; retry in a fresh output directory"
            )
        return tuple(sorted(raw))
    if stage == "inverse_folding":
        pairs = {p.stem for p in base.glob("*.npz")}
        if not ids or ids != pairs:
            raise PipelineValidationError("inverse_folding: incomplete CIF/NPZ pairs")
        if not request.only_inverse_fold:
            raw = _candidate_files(request.output_dir / "intermediate_designs")
            if len(ids) < len(raw) * request.inverse_fold_num_sequences:
                raise PipelineValidationError(
                    "inverse_folding: some sequences are missing"
                )
        return tuple(sorted(ids))
    if not ids:
        raise PipelineValidationError(f"{stage}: no candidate CIFs were produced")
    if stage in ("folding", "design_folding"):
        folder = (
            const.folding_dirname
            if stage == "folding"
            else const.folding_design_dirname
        )
        cif_folder = (
            const.refold_cif_dirname
            if stage == "folding"
            else const.refold_design_cif_dirname
        )
        if {p.stem for p in (base / folder).glob("*.npz")} != ids or {
            p.stem for p in (base / cif_folder).glob("*.cif")
        } != ids:
            raise PipelineValidationError(
                f"{stage}: missing or extra refold CIF/NPZ artifacts"
            )
    elif stage == "affinity":
        if {p.stem for p in (base / const.affinity_dirname).glob("*.npz")} != ids:
            raise PipelineValidationError(
                "affinity: missing or extra affinity archives"
            )
    elif stage == "esmfold2_scoring":
        score_dir = base / SCORE_DIR
        if {
            p.stem
            for p in score_dir.glob("*.json")
            if not p.name.endswith(".input.json")
        } != ids:
            raise PipelineValidationError(
                "esmfold2_scoring: incomplete per-design scores"
            )
        for design_id in ids:
            input_path = score_dir / f"{design_id}.input.json"
            if not input_path.is_file():
                raise PipelineValidationError(f"esmfold2_scoring: missing {input_path}")
            source = json.loads(input_path.read_text())
            result = load_result(score_dir / f"{design_id}.json", fingerprint(source))
            if source.get("design_id") != design_id or source.get(
                "design_sha256"
            ) != _sha256(candidates[design_id]):
                raise PipelineValidationError(
                    f"esmfold2_scoring: stale design identity {design_id}"
                )
            mode = "redesign" if request.protocol == "protein-redesign" else "binder"
            if (
                source.get("scoring_mode", "binder") != mode
                or result.get("scoring_mode", "binder") != mode
            ):
                raise PipelineValidationError(
                    f"esmfold2_scoring: wrong scoring mode for {design_id}"
                )
    elif stage == "analysis":
        metrics = base / "aggregate_metrics_analyze.csv"
        sequences = base / "ca_coords_sequences.pkl.gz"
        if not metrics.is_file() or not sequences.is_file():
            raise PipelineValidationError(
                "analysis: missing metrics CSV or sequence archive"
            )
        analyzed = pd.read_csv(
            metrics, dtype={"id": str, "file_name": str}, keep_default_na=False
        )
        if (
            analyzed.empty
            or analyzed["id"].duplicated().any()
            or set(analyzed["id"]) != ids
            or any(
                name != f"{key}.cif"
                for key, name in zip(analyzed["id"], analyzed["file_name"])
            )
        ):
            raise PipelineValidationError(
                "analysis: metrics IDs and candidate CIFs do not match"
            )
        if set(pd.read_pickle(sequences)["id"].astype(str)) != ids:  # noqa: S301 -- own trusted analysis output
            raise PipelineValidationError(
                "analysis: sequence archive does not match metrics"
            )
        if (
            merged
            and {p.stem for p in (base / const.refold_cif_dirname).glob("*.cif")} != ids
        ):
            raise PipelineValidationError(
                "merge: missing refold CIF for an analyzed design"
            )
        if request.protocol == "protein-small_molecule":
            score_key = "affinity_probability_binary1"
            if (
                score_key not in analyzed
                or not np.isfinite(
                    pd.to_numeric(analyzed[score_key], errors="coerce")
                ).all()
            ):
                raise PipelineValidationError(
                    "analysis: missing or invalid ligand affinity probabilities"
                )
            for row in analyzed.to_dict("records"):
                archive = base / const.affinity_dirname / f"{row['id']}.npz"
                try:
                    with np.load(archive, allow_pickle=False) as prediction:
                        score = float(prediction[score_key].item())
                except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                    raise PipelineValidationError(
                        f"analysis: missing or invalid affinity evidence for {row['id']}"
                    ) from exc
                if (
                    not math.isfinite(score)
                    or not 0 <= score <= 1
                    or not math.isclose(
                        float(row[score_key]), score, rel_tol=0, abs_tol=1e-12
                    )
                ):
                    raise PipelineValidationError(
                        f"analysis: stale {score_key} for {row['id']}"
                    )
        else:
            score_key = (
                REDESIGN_SCORE_KEY
                if request.protocol == "protein-redesign"
                else SCORE_KEY
            )
            if score_key not in analyzed or "esmfold2_input_hash" not in analyzed:
                raise PipelineValidationError(
                    f"analysis: missing {score_key} or ESM score provenance"
                )
            score_metrics = set()
            for row in analyzed.to_dict("records"):
                score_dir = base / SCORE_DIR
                source = json.loads((score_dir / f"{row['id']}.input.json").read_text())
                result = load_result(
                    score_dir / f"{row['id']}.json", fingerprint(source)
                )
                if (
                    source.get("design_sha256") != _sha256(candidates[row["id"]])
                    or row["esmfold2_input_hash"] != result["input_hash"]
                    or not math.isclose(
                        float(row[score_key]),
                        result["metrics"][score_key],
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    or (
                        request.protocol == "protein-redesign"
                        and row.get("esmfold2_score_metric") != result["score_metric"]
                    )
                ):
                    raise PipelineValidationError(
                        f"analysis: stale {score_key} for {row['id']}"
                    )
                if request.protocol == "protein-redesign":
                    score_metrics.add(result["score_metric"])
            if len(score_metrics) > 1:
                raise PipelineValidationError(
                    "analysis: cannot rank mixed monomer and multichain redesign scores"
                )
    elif stage == "filtering":
        _selected(request, {}, revision=revision)
    return tuple(sorted(ids))


def _selected(
    request: PipelineRequest, sources: Mapping[str, Any], *, revision: str
) -> tuple[Path, tuple[SelectedDesign, ...]]:
    root = request.output_dir
    directory = _filter_run_dir(root, revision) / "final_ranked_designs"
    csv = directory / f"final_designs_metrics_{request.budget}.csv"
    if not csv.is_file():
        raise PipelineValidationError(f"filtering: missing final selected CSV {csv}")
    selected = pd.read_csv(
        csv, dtype={"id": str, "file_name": str}, keep_default_na=False
    )
    all_metrics = directory / "all_designs_metrics.csv"
    if (
        not all_metrics.is_file()
        or selected.empty
        or "id" not in selected
        or "file_name" not in selected
        or selected["id"].duplicated().any()
    ):
        raise PipelineValidationError("filtering: no unambiguous selected designs")
    pool = pd.read_csv(
        all_metrics, dtype={"id": str, "file_name": str}, keep_default_na=False
    )
    candidates = _candidate_files(_design_dir(request))
    if (
        pool.empty
        or "id" not in pool
        or "file_name" not in pool
        or "final_rank" not in pool
        or pool["id"].duplicated().any()
        or not set(pool["id"]) <= set(candidates)
        or any(
            filename != f"{design_id}.cif"
            for design_id, filename in zip(pool["id"], pool["file_name"])
        )
        or len(selected) > request.budget
    ):
        raise PipelineValidationError("filtering: invalid ranked design pool")
    rank_width = len(str(len(pool)))
    ranked = pool.set_index("id")
    analyzed = pd.read_csv(
        _design_dir(request) / "aggregate_metrics_analyze.csv",
        dtype={"id": str, "file_name": str},
        keep_default_na=False,
    ).set_index("id")
    seen_ranks: set[int] = set()
    designs = []
    for row in selected.to_dict("records"):
        design_id = row["id"]
        filename = row["file_name"]
        if (
            design_id not in ranked.index
            or filename != f"{design_id}.cif"
            or "final_rank" not in row
            or "quality_score" not in row
        ):
            raise PipelineValidationError(
                f"filtering: selected candidate {design_id} is not from the ranked pool"
            )
        try:
            rank = int(row["final_rank"])
            quality = float(row["quality_score"])
            original_rank = int(ranked.loc[design_id, "final_rank"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise PipelineValidationError(
                f"filtering: invalid rank or quality for {design_id}"
            ) from exc
        if (
            rank < 1
            or rank > len(pool)
            or rank in seen_ranks
            or rank != original_rank
            or float(row["final_rank"]) != rank
            or float(ranked.loc[design_id, "final_rank"]) != original_rank
            or not math.isfinite(quality)
            or not 0 <= quality <= 1
        ):
            raise PipelineValidationError(
                f"filtering: selected rank or quality is stale for {design_id}"
            )
        seen_ranks.add(rank)
        exported = (
            directory
            / f"final_{request.budget}_designs"
            / f"rank{rank:0{rank_width}d}_{filename}"
        )
        before = (
            directory
            / f"final_{request.budget}_designs"
            / "before_refolding"
            / exported.name
        )
        if not exported.is_file() or not before.is_file():
            raise PipelineValidationError(
                f"filtering: missing paired exported CIF for {design_id}"
            )
        _safe_relative(root, exported)
        _safe_relative(root, before)
        if _sha256(exported) != _sha256(
            _design_dir(request) / const.refold_cif_dirname / filename
        ) or _sha256(before) != _sha256(candidates[design_id]):
            raise PipelineValidationError(
                f"filtering: exported CIF does not match source {design_id}"
            )
        score_key = (
            "affinity_probability_binary1"
            if request.protocol == "protein-small_molecule"
            else REDESIGN_SCORE_KEY
            if request.protocol == "protein-redesign"
            else SCORE_KEY
        )
        try:
            score = float(row[score_key])
            analyzed_score = float(analyzed.loc[design_id, score_key])
            ranked_score = float(ranked.loc[design_id, score_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineValidationError(
                f"filtering: {design_id} has no valid {score_key}"
            ) from exc
        if not math.isfinite(score) or not all(
            math.isclose(score, value, rel_tol=0, abs_tol=1e-12)
            for value in (analyzed_score, ranked_score)
        ):
            raise PipelineValidationError(
                f"filtering: {design_id} has a stale {score_key}"
            )
        if request.protocol in _POLYMER and (
            row.get("esmfold2_input_hash")
            != analyzed.loc[design_id, "esmfold2_input_hash"]
            or (
                request.protocol == "protein-redesign"
                and row.get("esmfold2_score_metric")
                != analyzed.loc[design_id, "esmfold2_score_metric"]
            )
        ):
            raise PipelineValidationError(
                f"filtering: stale ESM score provenance for {design_id}"
            )
        lineage = sources.get(
            design_id, {"source_id": design_id, "source_run": str(root)}
        )
        esm_dir = _design_dir(request) / SCORE_DIR
        esm_cif = esm_dir / f"{design_id}.cif"
        esm_json = esm_dir / f"{design_id}.json"
        if request.protocol in _POLYMER and not (
            esm_cif.is_file() and esm_json.is_file()
        ):
            raise PipelineValidationError(
                f"filtering: missing paired ESMFold2 structure/score for {design_id}"
            )
        if request.protocol in _POLYMER:
            _safe_relative(root, esm_cif)
            _safe_relative(root, esm_json)
        affinity_npz = (
            _design_dir(request) / const.affinity_dirname / f"{design_id}.npz"
        )
        if request.protocol == "protein-small_molecule":
            if not affinity_npz.is_file():
                raise PipelineValidationError(
                    f"filtering: missing paired affinity evidence for {design_id}"
                )
            _safe_relative(root, affinity_npz)
        verdicts = [
            value
            for key, value in row.items()
            if key.startswith("pass_") and key.endswith("_filter")
        ]
        passed = bool(verdicts) and all(
            str(v).lower() in ("true", "1", "1.0") for v in verdicts
        )
        if not passed:
            logging.getLogger(__name__).warning(
                "Selected %s did not pass every filter", design_id
            )
        designs.append(
            SelectedDesign(
                id=design_id,
                source_id=lineage["source_id"],
                source_run=Path(lineage["source_run"]),
                cif=exported,
                rank=rank,
                before_refolding_cif=before,
                esmfold2_cif=esm_cif if request.protocol in _POLYMER else None,
                esmfold2_json=esm_json if request.protocol in _POLYMER else None,
                passes_filters=passed,
                affinity_npz=(
                    affinity_npz
                    if request.protocol == "protein-small_molecule"
                    else None
                ),
            )
        )
    final_dir = directory / f"final_{request.budget}_designs"
    names = {design.cif.name for design in designs}
    if {path.name for path in final_dir.glob("*.cif")} != names or {
        path.name for path in (final_dir / "before_refolding").glob("*.cif")
    } != names:
        raise PipelineValidationError(
            "filtering: selected CSV and exported CIF pairs disagree"
        )
    return csv, tuple(designs)


def _serialize_request(request: PipelineRequest) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return [plain(child) for child in value]
        return value

    return {
        attribute.name: plain(getattr(request, attribute.name))
        for attribute in fields(request)
    }


def _restore_request(value: dict[str, Any]) -> PipelineRequest:
    options = value.get("step_options", {})
    if "scoring_target_chains" in options.get("esmfold2_scoring", {}):
        options["esmfold2_scoring"]["scoring_target_chains"] = tuple(
            options["esmfold2_scoring"]["scoring_target_chains"]
        )
    return PipelineRequest(**value)


def _write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    destination = root / _MANIFEST
    pending = root / f".{_MANIFEST}.next"
    pending.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    pending.replace(destination)
    return destination


def _read_manifest(plan: PipelinePlan) -> dict[str, Any]:
    path = plan.request.output_dir / _MANIFEST
    pending = path.with_name(f".{_MANIFEST}.next")
    if pending.exists() or pending.is_symlink():
        raise PipelineResumeError(
            f"Incomplete manifest update in {pending}; inspect the interrupted run"
        )
    if path.is_symlink():
        raise PipelineResumeError(f"Refusing a symlinked resume manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PipelineResumeError(f"Missing or invalid run manifest {path}") from exc
    expected_inputs = [
        {"path": str(ref.path), "sha256": ref.sha256} for ref in plan.input_files
    ]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _VERSION
        or manifest.get("fingerprint") != plan.fingerprint
        or manifest.get("protocol") != plan.request.protocol
        or manifest.get("graph") != list(plan.stages)
        or manifest.get("inputs") != expected_inputs
        or manifest.get("fork_sha") != _source_revision()
        or manifest.get("implementation_hash") != _implementation_hash()
    ):
        raise PipelineResumeError(
            "Resume protocol, pinned inputs, device, fork revision or stage graph differs"
        )
    if not isinstance(manifest.get("stage_receipts"), dict):
        raise PipelineResumeError("Resume manifest has no stage receipts")
    return manifest


def _receipt_hash(receipt: dict[str, Any]) -> str:
    return _digest(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )


def _check_stage_files(
    request: PipelineRequest, name: str, receipt: dict[str, Any]
) -> None:
    root = request.output_dir
    if not isinstance(receipt.get("files"), list) or any(
        not isinstance(file, dict)
        or not isinstance(file.get("name"), str)
        or not isinstance(file.get("sha256"), str)
        for file in receipt["files"]
    ):
        raise PipelineResumeError(f"{name}: invalid stage file references")
    expected = {file["name"]: file for file in receipt["files"]}
    if len(expected) != len(receipt["files"]):
        raise PipelineResumeError(f"{name}: duplicate stage file references")
    observed = {
        _safe_relative(root, path)
        for path in (
            *_stage_paths(
                request,
                name,
                merged=bool(receipt.get("merged")),
                revision=receipt.get("revision"),
            ),
            _config_path(root, name, receipt.get("revision")),
        )
        if path.is_file()
    }
    if set(expected) != observed:
        raise PipelineResumeError(
            f"{name}: missing, partial or unexpected stage artifacts"
        )
    for relative, file in expected.items():
        path = _artifact_path(root, relative)
        if not path.is_file() or _sha256(path) != file["sha256"]:
            raise PipelineResumeError(
                f"{name}: artifact changed or was swapped: {path}"
            )


def _checked_filter_history(plan: PipelinePlan, manifest: dict[str, Any]) -> None:
    history = manifest.get("filter_history")
    if not isinstance(history, dict) or set(history) != {
        f"v{index:04d}" for index in range(1, len(history) + 1)
    }:
        raise PipelineResumeError("Filtering revision history is invalid")
    pending = manifest.get("pending_filter_revision")
    if pending is not None and pending != f"v{len(history) + 1:04d}":
        raise PipelineResumeError("The pending filtering revision is invalid")
    receipts = manifest["stage_receipts"]
    current = receipts.get("filtering")
    if current is not None and (
        pending is not None or current.get("revision") != f"v{len(history):04d}"
    ):
        raise PipelineResumeError(
            "Filtering receipt does not match its revision history"
        )
    upstream = {
        name: receipt["receipt_hash"]
        for name, receipt in receipts.items()
        if name != "filtering"
    }
    prior: dict[str, str] = {}
    for revision in sorted(history):
        path = _filter_run_dir(plan.request.output_dir, revision) / _MANIFEST
        _safe_relative(plan.request.output_dir, path)
        if not path.is_file() or not isinstance(history[revision], str):
            raise PipelineResumeError(f"{revision}: filtering snapshot is missing")
        if _sha256(path) != history[revision]:
            raise PipelineResumeError(f"{revision}: filtering snapshot changed")
        try:
            snapshot = json.loads(path.read_text())
            saved_receipts = snapshot["stage_receipts"]
            saved = saved_receipts["filtering"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineResumeError(
                f"{revision}: invalid filtering snapshot"
            ) from exc
        if (
            snapshot.get("fingerprint") != plan.fingerprint
            or snapshot.get("schema_version") != _VERSION
            or snapshot.get("protocol") != plan.request.protocol
            or snapshot.get("graph") != list(plan.stages)
            or snapshot.get("filter_history") != prior
            or snapshot.get("pending_filter_revision") is not None
            or snapshot.get("source_runs") != manifest["source_runs"]
            or not isinstance(saved, dict)
            or saved.get("revision") != revision
            or saved.get("receipt_hash") != _receipt_hash(saved)
            or {
                name: receipt["receipt_hash"]
                for name, receipt in saved_receipts.items()
                if name != "filtering"
            }
            != upstream
            or (
                current is not None
                and revision == current.get("revision")
                and saved["receipt_hash"] != current["receipt_hash"]
            )
        ):
            raise PipelineResumeError(
                f"{revision}: filtering snapshot has incompatible provenance"
            )
        _check_stage_files(plan.request, "filtering", saved)
        prior[revision] = history[revision]


def _checked_receipts(
    plan: PipelinePlan, manifest: dict[str, Any], *, rerun_from: str | None = None
) -> None:
    request = plan.request
    receipts = manifest["stage_receipts"]
    if set(receipts) - set(plan.stages):
        raise PipelineResumeError("Resume manifest names a stage outside this protocol")
    if not isinstance(manifest.get("source_runs"), dict):
        raise PipelineResumeError("Resume manifest has no source lineage")
    asset_hashes: dict[Path, str] = {}
    for name in plan.stages:
        if name not in receipts:
            continue
        receipt = receipts[name]
        if not isinstance(receipt, dict) or receipt.get(
            "receipt_hash"
        ) != _receipt_hash(receipt):
            raise PipelineResumeError(f"{name}: stage receipt is corrupted")
        if receipt.get("merged") and receipt.get("source_runs_hash") != _digest(
            manifest["source_runs"]
        ):
            raise PipelineResumeError(f"{name}: merged source lineage changed")
        if not receipt.get("merged"):
            for dependency in _dependencies(plan.stages, name):
                parent = receipts.get(dependency)
                if parent is None or receipt["upstream"].get(dependency) != parent.get(
                    "receipt_hash"
                ):
                    raise PipelineResumeError(
                        f"{name}: upstream {dependency} receipt changed"
                    )
        _check_stage_files(request, name, receipt)
        if rerun_from is not None and plan.stages.index(name) >= plan.stages.index(
            rerun_from
        ):
            continue
        if not isinstance(receipt.get("assets"), dict):
            raise PipelineResumeError(f"{name}: missing pinned asset references")
        for label, asset in receipt["assets"].items():
            if (
                not isinstance(asset, dict)
                or not isinstance(asset.get("path"), str)
                or not isinstance(asset.get("sha256"), str)
            ):
                raise PipelineResumeError(f"{name}: invalid pinned asset {label}")
            path = Path(asset["path"])
            if not path.is_absolute() or not path.exists():
                raise PipelineResumeError(
                    f"{name}: pinned asset {label} is missing; explicitly rerun this stage"
                )
            if path not in asset_hashes:
                asset_hashes[path] = _sha256(path)
            if asset_hashes[path] != asset["sha256"]:
                raise PipelineResumeError(
                    f"{name}: pinned asset {label} changed; explicitly rerun this stage"
                )
        if receipt.get("merged"):
            _validate_outputs(request, "analysis", merged=True)
    _checked_filter_history(plan, manifest)


def _invalidate_after(plan: PipelinePlan, manifest: dict[str, Any], name: str) -> None:
    root = plan.request.output_dir
    for stage in plan.stages[plan.stages.index(name) :]:
        receipt = manifest["stage_receipts"].pop(stage, None)
        if receipt is None:
            continue
        if stage == "filtering":
            continue
        for file in receipt["files"]:
            _artifact_path(root, file["name"]).unlink(missing_ok=True)
    manifest["failed_stage"] = None
    _write_manifest(root, manifest)


def _clean_failed_step(
    request: PipelineRequest, stage: str, manifest: dict[str, Any]
) -> None:
    root = request.output_dir
    if stage == "filtering":
        directory = _filter_run_dir(root, manifest.get("pending_filter_revision"))
        _safe_relative(root, directory)
        if (directory / _MANIFEST).exists():
            raise PipelineResumeError(
                "An interrupted filtering snapshot exists; inspect it before retrying"
            )
        if directory.exists():
            import shutil

            shutil.rmtree(directory)
        return
    for path in _stage_paths(request, stage):
        _safe_relative(root, path)
        path.unlink()
    _config_path(root, stage).unlink(missing_ok=True)


def _record_stage(
    plan: PipelinePlan,
    manifest: dict[str, Any],
    stage: str,
    config: omegaconf.DictConfig | None,
    assets: Mapping[str, dict[str, str]],
    elapsed: float,
    *,
    merged: bool = False,
    revision: str | None = None,
    persist: bool = True,
) -> StageResult:
    root = plan.request.output_dir
    for label, asset in assets.items():
        path = Path(asset["path"])
        if not path.exists() or _sha256(path) != asset["sha256"]:
            raise PipelineValidationError(
                f"{stage}: pinned {label} changed during execution"
            )
    ids = _validate_outputs(plan.request, stage, merged=merged, revision=revision)
    files = tuple(
        _file_ref(root, path, manifest["source_runs"])
        for path in (
            *_stage_paths(plan.request, stage, merged=merged, revision=revision),
            *((_config_path(root, stage, revision),) if config is not None else ()),
        )
    )
    if not files:
        raise PipelineValidationError(
            f"{stage}: task returned without producing any durable artifacts"
        )
    backend = "cpu" if stage in ("analysis", "filtering") else plan.request.device
    receipt: dict[str, Any] = {
        "name": stage,
        "elapsed_seconds": elapsed,
        "device": backend,
        "settings_hash": _settings(plan.request, stage),
        "config_hash": _digest(omegaconf.OmegaConf.to_container(config, resolve=True))
        if config is not None
        else None,
        "assets": dict(assets),
        "source_runs_hash": _digest(manifest["source_runs"]) if merged else None,
        "upstream": {}
        if merged
        else {
            parent: manifest["stage_receipts"][parent]["receipt_hash"]
            for parent in _dependencies(plan.stages, stage)
        },
        "files": [
            {
                "name": file.name,
                "sha256": file.sha256,
                "source_id": file.source_id,
                "source_run": str(file.source_run) if file.source_run else None,
            }
            for file in files
        ],
        "design_ids": list(ids),
        "merged": merged,
        "revision": revision,
    }
    receipt["receipt_hash"] = _receipt_hash(receipt)
    manifest["stage_receipts"][stage] = receipt
    manifest["failed_stage"] = None
    manifest["request"] = _serialize_request(plan.request)
    if persist:
        _write_manifest(root, manifest)
    return StageResult(
        name=stage,
        status="merged" if merged else "completed",
        files=files,
        elapsed_seconds=elapsed,
        device=backend,
        design_ids=ids,
    )


def _publish_filter_snapshot(
    root: Path, manifest: dict[str, Any], revision: str
) -> Path:
    directory = _filter_run_dir(root, revision)
    path = directory / _MANIFEST
    pending = directory / f".{_MANIFEST}.next"
    if path.exists() or path.is_symlink() or pending.exists() or pending.is_symlink():
        raise PipelineResumeError(f"{revision}: filtering snapshot already exists")
    manifest["pending_filter_revision"] = None
    pending.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    pending.replace(path)
    manifest["filter_history"][revision] = _sha256(path)
    _write_manifest(root, manifest)
    return path


def _validate_native_spec(request: PipelineRequest) -> None:
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.parse.schema import YamlDesignParser

    mols = load_canonicals(str(request.moldir))
    YamlDesignParser(request.moldir).parse_yaml(
        request.design_spec, mols, request.moldir
    )


@contextmanager
def _run_state(request: PipelineRequest):
    if not _ACTIVE_RUN.acquire(blocking=False):
        raise RuntimeError(
            "Concurrent BoltzGen API runs cannot share process-global model state"
        )
    try:
        environment = dict(os.environ)
        grad = torch.is_grad_enabled()
        matmul = torch.get_float32_matmul_precision()
        rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        cuda_device = torch.cuda.current_device() if cuda is not None else None
        xpu = (
            torch.xpu.get_rng_state_all()
            if hasattr(torch, "xpu") and torch.xpu.is_initialized()
            else None
        )
        xpu_device = torch.xpu.current_device() if xpu is not None else None
        filters = warnings.filters[:]
        lightning = logging.getLogger("pytorch_lightning")
        log_level = lightning.level
        from rdkit import Chem

        pickle_properties = Chem.GetDefaultPickleProperties()
        from boltzgen.data.mol import MOLDIR_ZIP_CACHE

        cached = set(MOLDIR_ZIP_CACHE)
        try:
            os.environ.update(
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", UV_OFFLINE="1"
            )
            set_timing_file(request.output_dir / "timings.jsonl")
            if request.seed is not None:
                random.seed(request.seed)
                np.random.seed(request.seed)
                if request.device == "cpu":
                    # Global seeding schedules accelerator seeds we cannot restore.
                    torch.default_generator.manual_seed(request.seed)
                else:
                    torch.manual_seed(request.seed)
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            failures: list[Exception] = []

            def restore(action: Callable[[], Any]) -> None:
                try:
                    action()
                except Exception as exc:
                    failures.append(exc)

            restore(flush_rollup)
            restore(lambda: torch.set_grad_enabled(grad))
            restore(lambda: torch.set_float32_matmul_precision(matmul))
            restore(lambda: random.setstate(rng[0]))
            restore(lambda: np.random.set_state(rng[1]))
            restore(lambda: torch.random.set_rng_state(rng[2]))
            if cuda is not None:
                restore(lambda: torch.cuda.set_rng_state_all(cuda))
            if xpu is not None:
                restore(lambda: torch.xpu.set_rng_state_all(xpu))
            restore(lambda: Chem.SetDefaultPickleProperties(pickle_properties))
            for key in set(os.environ) | set(environment):
                if key in environment:
                    os.environ[key] = environment[key]
                else:
                    os.environ.pop(key, None)
            warnings.filters[:] = filters
            lightning.setLevel(log_level)
            for key in set(MOLDIR_ZIP_CACHE) - cached:
                restore(lambda key=key: MOLDIR_ZIP_CACHE.pop(key).close())
            restore(gc.collect)
            backend, index = _device(request)
            if backend == "cuda" and torch.cuda.is_initialized():

                def clear_cuda() -> None:
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()

                restore(clear_cuda)
            if backend == "xpu" and torch.xpu.is_initialized():

                def clear_xpu() -> None:
                    with torch.xpu.device(index):
                        torch.xpu.empty_cache()

                restore(clear_xpu)
            if cuda_device is not None:
                restore(lambda: torch.cuda.set_device(cuda_device))
            if xpu_device is not None:
                restore(lambda: torch.xpu.set_device(xpu_device))
            if failures:
                if not active_error:
                    raise RuntimeError(
                        "Could not fully restore BoltzGen run state"
                    ) from failures[0]
                logging.getLogger(__name__).warning(
                    "Could not fully restore BoltzGen run state: %s", failures[0]
                )
    finally:
        _ACTIVE_RUN.release()


class BoltzGenEngine:
    """Own sequential, single-device pipeline runs and their durable receipts."""

    def __init__(self) -> None:
        self._closed = False

    def __enter__(self) -> BoltzGenEngine:
        if self._closed:
            raise RuntimeError("BoltzGenEngine is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True

    def plan(self, request: PipelineRequest) -> PipelinePlan:
        if self._closed:
            raise RuntimeError("BoltzGenEngine is closed")
        if not isinstance(request, PipelineRequest):
            raise TypeError("plan() requires a PipelineRequest")
        graph = _graph(request)
        _check_request(request, graph)
        if request.design_spec is None:
            raise PipelineValidationError(
                "design_spec is required; use the original native spec to merge or refilter"
            )
        if request.design_spec.suffix != ".yaml":
            raise PipelineValidationError("design_spec must be a native .yaml file")
        inputs = _spec_inputs(request.design_spec)
        inputs[str(request.moldir)] = _sha256(request.moldir)
        refs = tuple(
            ArtifactRef(name=path, path=Path(path), sha256=digest)
            for path, digest in sorted(inputs.items())
        )
        identity = {
            "protocol": request.protocol,
            "output_dir": str(request.output_dir),
            "inputs": inputs,
            "device": request.device,
            "precision": request.precision,
            "seed": request.seed,
            "only_inverse_fold": request.only_inverse_fold,
            "skip_inverse_folding": request.skip_inverse_folding,
            "inverse_fold_model": request.inverse_fold_model,
            "fork_sha": _source_revision(),
            "implementation_hash": _implementation_hash(),
            "graph": graph,
        }
        return PipelinePlan(
            request=request,
            stages=graph,
            input_files=refs,
            fingerprint=_digest(identity),
        )

    def run(
        self,
        plan: PipelinePlan,
        *,
        through: str | None = None,
        stages: Sequence[str] | None = None,
        resume_from: Path | PipelineRun | None = None,
        on_stage: Callable[[StageResult], None] | None = None,
    ) -> PipelineRun:
        if self._closed:
            raise RuntimeError("BoltzGenEngine is closed")
        if not isinstance(plan, PipelinePlan) or self.plan(plan.request) != plan:
            raise PipelineValidationError(
                "Plan or pinned inputs changed; call plan() again"
            )
        graph = plan.stages
        if stages is not None and through is not None:
            raise PipelineValidationError("through and stages are mutually exclusive")
        if through is not None and through not in graph:
            raise PipelineValidationError(
                f"{plan.request.protocol}: unavailable through stage {through}"
            )
        if stages is not None:
            chosen = tuple(stages)
            if (
                not chosen
                or len(chosen) != len(set(chosen))
                or tuple(name for name in graph if name in chosen) != chosen
            ):
                raise PipelineValidationError(
                    f"Stages must be distinct and in protocol order: {graph}"
                )
        else:
            chosen = graph[: graph.index(through) + 1] if through is not None else graph
        request = plan.request
        root = request.output_dir
        handle = root / _MANIFEST
        snapshot_source: Path | None = None
        if resume_from is not None:
            source = (
                resume_from.resume_handle
                if isinstance(resume_from, PipelineRun)
                else Path(resume_from)
            )
            source = source.expanduser().resolve()
            if source not in (root, handle):
                if (
                    source.name != _MANIFEST
                    or source.parent.parent != root / _FILTER_RUNS
                    or source != _filter_run_dir(root, source.parent.name) / _MANIFEST
                ):
                    raise PipelineResumeError(
                        "resume_from must point to this run directory or one of its manifests"
                    )
                snapshot_source = source
        with _run_state(request):
            if resume_from is None:
                if root.exists() and (not root.is_dir() or any(root.iterdir())):
                    raise PipelineResumeError(
                        f"Output directory is not empty; use resume_from or a new directory: {root}"
                    )
                root.mkdir(parents=True, exist_ok=True)
                manifest: dict[str, Any] = {
                    "schema_version": _VERSION,
                    "fingerprint": plan.fingerprint,
                    "protocol": request.protocol,
                    "graph": list(graph),
                    "fork_sha": _source_revision(),
                    "implementation_hash": _implementation_hash(),
                    "inputs": [
                        {"path": str(ref.path), "sha256": ref.sha256}
                        for ref in plan.input_files
                    ],
                    "request": _serialize_request(request),
                    "stage_receipts": {},
                    "source_runs": {},
                    "failed_stage": None,
                    "merged_from": [],
                    "filter_history": {},
                    "pending_filter_revision": None,
                }
                _write_manifest(root, manifest)
            else:
                manifest = _read_manifest(plan)
                _checked_receipts(
                    plan, manifest, rerun_from=chosen[0] if stages is not None else None
                )
                if snapshot_source is not None and snapshot_source != (
                    _filter_run_dir(root, f"v{len(manifest['filter_history']):04d}")
                    / _MANIFEST
                ):
                    raise PipelineResumeError(
                        "resume_from names an older filtering snapshot; use the latest run manifest"
                    )
            receipts = manifest["stage_receipts"]
            changed = [
                stage
                for stage, receipt in receipts.items()
                if receipt["settings_hash"] != _settings(request, stage)
            ]
            for stage in changed:
                if stage != "filtering" and (stages is None or stage not in chosen):
                    raise PipelineResumeError(
                        f"{stage}: settings changed; explicitly rerun this stage and its descendants"
                    )
            invalid_filter = "filtering" in changed
            available = set(receipts) - ({"filtering"} if invalid_filter else set())
            execute: list[str] = []
            for stage in chosen:
                if stages is not None or stage not in available:
                    missing = set(_dependencies(graph, stage)) - available
                    if missing:
                        raise PipelineResumeError(
                            f"{stage}: missing completed dependencies {sorted(missing)}"
                        )
                    available.difference_update(graph[graph.index(stage) :])
                    available.add(stage)
                    execute.append(stage)
            if manifest["filter_history"] and any(
                stage != "filtering" for stage in execute
            ):
                raise PipelineResumeError(
                    "Immutable filtering results exist; rerun upstream stages in a new output directory"
                )
            assets = {stage: _stage_assets(request, stage) for stage in execute}
            digests: dict[Path, str] = {}
            pinned_assets: dict[str, dict[str, dict[str, str]]] = {}
            for stage, entries in assets.items():
                pinned_assets[stage] = {}
                for label, path in entries.items():
                    if path not in digests:
                        digests[path] = _sha256(path)
                    pinned_assets[stage][label] = {
                        "path": str(path),
                        "sha256": digests[path],
                    }
            if any(
                stage in ("design", "inverse_folding") and stage == graph[0]
                for stage in execute
            ):
                _validate_native_spec(request)
            configs = _recipe(request, execute) if execute else {}
            if invalid_filter and "filtering" in receipts:
                _invalidate_after(plan, manifest, "filtering")
            started = set(receipts)
            for index, stage in enumerate(chosen):
                if stage not in execute:
                    continue
                if stage in receipts:
                    _invalidate_after(plan, manifest, stage)
                revision = (
                    manifest["pending_filter_revision"]
                    or f"v{len(manifest['filter_history']) + 1:04d}"
                    if stage == "filtering"
                    else None
                )
                if stage == "filtering":
                    configs[stage].outdir = str(_filter_run_dir(root, revision))
                if manifest.get("failed_stage"):
                    failed = manifest["failed_stage"]["name"]
                    if failed != stage:
                        raise PipelineResumeError(
                            f"Unfinished {failed} must be rerun before {stage}"
                        )
                    _clean_failed_step(request, stage, manifest)
                elif (
                    _stage_paths(request, stage, revision=revision)
                    or _config_path(root, stage, revision).exists()
                    or (
                        stage == "filtering"
                        and _filter_run_dir(root, revision).exists()
                    )
                ):
                    raise PipelineResumeError(
                        f"{stage}: unreceipted partial artifacts; use a fresh directory"
                    )
                if stage == "filtering":
                    manifest["pending_filter_revision"] = revision
                    _write_manifest(root, manifest)
                config = configs[stage]
                config_path = _config_path(root, stage, revision)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                omegaconf.OmegaConf.save(config, config_path)
                os.environ["BOLTZGEN_PIPELINE_PROGRESS"] = (
                    f"Step {index + 1}/{len(chosen)}"
                )
                os.environ["BOLTZGEN_PIPELINE_STEP"] = stage
                started_at = time.perf_counter()
                try:
                    run_task(config)
                    result = _record_stage(
                        plan,
                        manifest,
                        stage,
                        config,
                        pinned_assets[stage],
                        time.perf_counter() - started_at,
                        revision=revision,
                        persist=stage != "filtering",
                    )
                    if stage == "filtering":
                        handle = _publish_filter_snapshot(root, manifest, revision)
                except Exception as exc:
                    if stage == "filtering":
                        manifest["stage_receipts"].pop(stage, None)
                        manifest["pending_filter_revision"] = revision
                    failed = StageResult(
                        name=stage,
                        status="failed",
                        files=(),
                        elapsed_seconds=time.perf_counter() - started_at,
                        device=request.device,
                    )
                    manifest["failed_stage"] = {
                        "name": stage,
                        "cause": str(exc),
                        "type": type(exc).__name__,
                    }
                    _write_manifest(root, manifest)
                    raise PipelineStageError(
                        request.protocol, stage, exc, failed
                    ) from exc
                finally:
                    os.environ.pop("BOLTZGEN_PIPELINE_STEP", None)
                    os.environ.pop("BOLTZGEN_PIPELINE_PROGRESS", None)
                    gc.collect()
                if on_stage is not None:
                    on_stage(result)
            results = _stage_results(manifest, root, graph)
            results = tuple(
                replace(result, status="reused")
                if result.name in started and result.name not in execute
                else result
                for result in results
            )
            complete = "filtering" in receipts and (
                len(receipts) == len(graph)
                or receipts.get("analysis", {}).get("merged", False)
            )
            final_csv, selected = (
                _selected(
                    request,
                    manifest["source_runs"],
                    revision=receipts["filtering"]["revision"],
                )
                if "filtering" in receipts
                else (None, ())
            )
            if "filtering" in receipts:
                handle = (
                    _filter_run_dir(root, receipts["filtering"]["revision"]) / _MANIFEST
                )
            return PipelineRun(
                stages=results,
                resume_handle=handle,
                output_dir=root,
                completed=complete,
                final_csv=final_csv,
                selected=selected,
            )

    def merge(
        self,
        request: PipelineRequest,
        *,
        runs: Sequence[PipelineRun | Path],
    ) -> PipelineRun:
        if self._closed:
            raise RuntimeError("BoltzGenEngine is closed")
        if not runs:
            raise PipelineValidationError("merge requires analyzed source runs")
        plan = self.plan(request)
        sources: list[Path] = []
        manifests: list[dict[str, Any]] = []
        source_requests: list[PipelineRequest] = []
        for run in runs:
            handle = run.resume_handle if isinstance(run, PipelineRun) else Path(run)
            root = (
                handle
                if handle.is_dir()
                else handle.parent.parent.parent
                if handle.name == _MANIFEST
                and handle.parent.parent.name == _FILTER_RUNS
                else handle.parent
            )
            root = root.expanduser().resolve()
            try:
                source = json.loads((root / _MANIFEST).read_text())
                source_request = _restore_request(source["request"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise PipelineResumeError(
                    f"merge: invalid source manifest in {root}"
                ) from exc
            if source_request.output_dir != root:
                raise PipelineResumeError(
                    f"merge: source manifest belongs to a different directory: {root}"
                )
            source_plan = self.plan(source_request)
            source = _read_manifest(source_plan)
            _checked_receipts(source_plan, source)
            if "analysis" not in source["stage_receipts"] or source["failed_stage"]:
                raise PipelineResumeError(
                    f"merge: source {root} has no completed analysis"
                )
            if source["stage_receipts"]["analysis"].get("merged"):
                raise PipelineResumeError(
                    "merge: use original analyzed runs; nested merges cannot preserve scoring compatibility"
                )
            if root not in sources:
                sources.append(root)
                manifests.append(source)
                source_requests.append(source_request)
        baseline = manifests[0]
        reference = source_requests[0]
        for source, source_request in zip(manifests[1:], source_requests[1:]):
            if (
                source["graph"] != baseline["graph"]
                or source["implementation_hash"] != baseline["implementation_hash"]
                or source["protocol"] != baseline["protocol"]
                or [(item["path"], item["sha256"]) for item in source["inputs"]]
                != [(item["path"], item["sha256"]) for item in baseline["inputs"]]
                or source_request.device != reference.device
                or source_request.precision != reference.precision
                or source_request.only_inverse_fold != reference.only_inverse_fold
                or source_request.skip_inverse_folding != reference.skip_inverse_folding
            ):
                raise PipelineResumeError(
                    "merge: incompatible protocol, inputs, implementation or device"
                )
            for stage in baseline["graph"]:
                if stage in ("design", "inverse_folding", "filtering"):
                    continue
                first = baseline["stage_receipts"].get(stage)
                other = source["stage_receipts"].get(stage)
                if (
                    first is None
                    or other is None
                    or first["settings_hash"] != other["settings_hash"]
                    or {key: value["sha256"] for key, value in first["assets"].items()}
                    != {key: value["sha256"] for key, value in other["assets"].items()}
                ):
                    raise PipelineResumeError(
                        f"merge: incompatible {stage} scoring or analysis"
                    )
        output = request.output_dir
        if output in sources or (
            output.exists() and (not output.is_dir() or any(output.iterdir()))
        ):
            raise PipelineResumeError("merge: destination must be distinct and empty")
        if (
            request.protocol != reference.protocol
            or request.device != reference.device
            or request.precision != reference.precision
            or request.only_inverse_fold != reference.only_inverse_fold
            or request.skip_inverse_folding != reference.skip_inverse_folding
            or [(ref.path, ref.sha256) for ref in plan.input_files]
            != [(Path(item["path"]), item["sha256"]) for item in baseline["inputs"]]
        ):
            raise PipelineResumeError(
                "merge: destination request must match the sources' protocol and inputs"
            )
        if any(
            _settings(request, stage)
            != baseline["stage_receipts"][stage]["settings_hash"]
            for stage in baseline["graph"]
            if stage not in ("design", "inverse_folding", "filtering")
            and stage in baseline["stage_receipts"]
        ):
            raise PipelineResumeError(
                "merge: destination scoring/analysis options differ"
            )
        output.mkdir(parents=True, exist_ok=True)
        with _run_state(request):
            mapping = merge_directories(sources, output)
            manifest = {
                "schema_version": _VERSION,
                "fingerprint": plan.fingerprint,
                "protocol": request.protocol,
                "graph": list(plan.stages),
                "fork_sha": _source_revision(),
                "implementation_hash": _implementation_hash(),
                "inputs": [
                    {"path": str(ref.path), "sha256": ref.sha256}
                    for ref in plan.input_files
                ],
                "request": _serialize_request(request),
                "stage_receipts": {},
                "source_runs": mapping,
                "failed_stage": None,
                "merged_from": [
                    {"path": str(path), "manifest_sha256": _sha256(path / _MANIFEST)}
                    for path in sources
                ],
                "filter_history": {},
                "pending_filter_revision": None,
            }
            _record_stage(plan, manifest, "analysis", None, {}, 0.0, merged=True)
            results = _stage_results(manifest, output, plan.stages)
            return PipelineRun(
                stages=results,
                resume_handle=output / _MANIFEST,
                output_dir=output,
                completed=False,
                final_csv=None,
            )
