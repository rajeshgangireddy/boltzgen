#!/usr/bin/env python3
"""
This script orchestrates work. It sets up an output directory with yaml files of pipeline steps that need to be run, and launches processes that run the pipeline steps.

Mainly it:
1) **Writes to yaml files** when `configure_command(...)` is executed
   - For each `PipelineStep`, the resolved Hydra config is written to
     `OUTPUT/config/<step>.yaml`.
   - A manifest `OUTPUT/steps.yaml` is also written, listing the enabled steps
     and their config files in execution order.

2) **Executes from YAML** when `execute_command(...)` is executed
   - Each step is launched **as a subprocess** (`python main.py <config.yaml>`)
     unless `--no_subprocess` is set (not the default).
   - If `--no_subprocess` is specified, the config is instantiated in-process
     and the `Task.run(...)` method is called directly.

The actual code that is exectued in each pipeline step is found in `main.py` which a wrapper for running the .run() function of our `Task` class.
If you run the pipeline (for example via `boltzgen run design_spec.yaml ...`) then this function reads the yaml files of the individual pipeline steps and executes the pipeline steps.

The possible tasks (and code files you want to inspect to understand what they are running):
    - Predict src/boltzgen/task/predict/predict.py (GPU: Running BoltzGen diffusion, inverse folding, refolding, designfolding, or affinity prediction)
    - Analyze src/boltzgen/task/analyze/analyze.py (CPU: Compute CPU Metrics and aggregate metrics from GPU steps)
    - Filter src/boltzgen/task/filter/filter.py (CPU: Very fast (20s) computes ranking and writes final output files)
"""

from boltzgen.utils.quiet import quiet_startup

quiet_startup()

import collections
import huggingface_hub
import huggingface_hub.constants
import argparse
from dataclasses import dataclass
import shlex
import subprocess
import os
import time
import math
import pickle  # noqa: E402
import re
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import hydra
import omegaconf
import torch
from rdkit import Chem  # noqa: E402

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.task import Task
from boltzgen.utils.timing import (
    Timer,
    record_timing,
    set_timing_file,
    flush_rollup,
    print_timing_summary,
    DEFAULT_TIMING_FILENAME,
)
from boltzgen.utils.device import (
    accelerator_type,
    device_capability_safe,
    device_count,
)
from importlib.metadata import PackageNotFoundError, version as pkg_version

from boltzgen._pipeline import (
    ARTIFACTS,
    SOLUBLEMPNN_SHA256,
    BinderDesignPipeline as _BinderDesignPipeline,
    PipelineStep,
    check_design_spec,
    check_design_specs,
    config_dir,
    configure_pipeline,
    get_artifact_path,
    execute_pipeline,
    main_script,
    merge_directories,
    parse_additional_filters,
    parse_config_args,
    parse_metrics_override,
    parse_size_buckets,
    protocol_configs,
    resolve_inverse_fold_model as _resolve_inverse_fold_model,
    step_names,
)


### CLI arguments ###
def add_configure_arguments(
    parser: argparse.ArgumentParser, *, output_required: bool = False
) -> None:
    # General configuration options
    p = parser.add_argument_group("general configuration")
    p.add_argument(
        "--protocol",
        type=str,
        choices=list(protocol_configs.keys()),
        default="protein-anything",
        help="Protocol to use for the design. This determines default settings and in some cases what steps "
        "are run. Default: %(default)s",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=output_required,
        help="Output directory for pipeline results",
    )
    p.add_argument(
        "--config",
        nargs="+",
        action="append",
        help="Override pipeline step configuration, in format <step_name> <arg1>=<value1> <arg2>=<value2> ..."
        "(example: '--config folding num_workers=4 trainer.devices=4'). Can be used multiple times.",
    )
    p.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use. Default is all devices available.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        help="Number of DataLoader worker processes.",
        default=1,
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible prediction and data-feature sampling.",
        default=None,
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        help=f"Path to the directory of default config files. Default: %(default)s",
        default=config_dir,
    )
    p.add_argument(
        "--use_kernels",
        help="Whether to use kernels. One of 'auto', 'true', or 'false'. Default: %(default)s. "
        "If 'auto', will use kernels if the device capability is >= 8.",
        choices=["auto", "true", "false"],
        default="auto",
    )
    p.add_argument(
        "--timing_file",
        type=str,
        help="Filename (relative to --output, or an absolute path) for the JSONL file "
        "that per-phase timing measurements are appended to. Change this between runs "
        "to keep timings from different experiments (e.g. different hardware backends) "
        "separate while using the same tracking mechanism. Default: %(default)s",
        default=DEFAULT_TIMING_FILENAME,
    )
    p.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing results across all steps. Generate only as many new designs are "
        "needed to achieve the specified total number of designs.",
    )

    # Design configuration options
    p = parser.add_argument_group("design")
    p.add_argument(
        "--num_designs",
        type=int,
        help="Number of total designs to generate. This commonly would be something like 10,000"
        "After generating 10,000 designs we then filter down to --budget many designs in the filter step",
        default=10000,
    )
    p.add_argument(
        "--diffusion_batch_size",
        type=int,
        default=None,
        help="Number of diffusion samples to generate per trunk run. If not specified, "
        "this defaults to 1 if --num-designs is less than 100, and 10 otherwise. Note that "
        "for design tasks that randomly sample the binder length (or use randomness in other "
        "ways), all designs generated in the same batch will share the same length. "
        "Having a large diffusion batch size compared to the total number of designs to "
        "generate will therefore not evenly sample the possible lengths.",
    )
    p.add_argument(
        "--design_checkpoints",
        type=str,
        nargs="+",
        help="Path to the boltzgen checkpoint(s). One or more checkpoints are supported. Just specifying an individual path here will work."
        "Each will be used for an equal fraction of the designs. By default, two checkpoints are used. "
        "Default: %(default)s",
        default=[
            ARTIFACTS["design-diverse"][0],
            ARTIFACTS["design-adherence"][0],
        ],
    )
    p.add_argument(
        "--step_scale",
        type=str,
        help="Fixed step scale to use (e.g. 1.8). Default is to use a schedule",
        default=None,
    )
    p.add_argument(
        "--noise_scale",
        type=str,
        help="Fixed noise scale to use (e.g. 0.98). Default is to use a schedule",
        default=None,
    )

    # Inverse folding configuration options
    p = parser.add_argument_group("inverse folding")
    p.add_argument(
        "--skip_inverse_folding",
        action="store_true",
        help="Skip inverse folding step",
    )
    p.add_argument(
        "--inverse_fold_num_sequences",
        type=int,
        help="Number of sequences per backbone to generate in the inverse fold step. Default: %(default)s",
        default=1,
    )
    p.add_argument(
        "--inverse_fold_checkpoint",
        type=str,
        help="BoltzIF checkpoint, used for protein-small_molecule or with --inverse_fold_model boltzif. Default: %(default)s",
        default=ARTIFACTS["inverse-fold"][0],
    )
    p.add_argument(
        "--inverse_fold_model",
        choices=["boltzif", "solublempnn"],
        default=None,
        help="Inverse-folding model. Default: boltzif for protein-small_molecule, "
        "solublempnn otherwise. protein-small_molecule requires boltzif.",
    )
    p.add_argument(
        "--solublempnn_checkpoint",
        default=ARTIFACTS["solublempnn"][0],
        help="SolubleMPNN checkpoint (local path or Hugging Face artifact). By default, downloads and caches the upstream v_48_020 weights.",
    )
    p.add_argument(
        "--solublempnn_sampling_temperature",
        type=float,
        default=0.1,
        help="Positive SolubleMPNN sampling temperature. Default: %(default)s",
    )
    p.add_argument(
        "--inverse_fold_avoid",
        type=str,
        default=None,
        help="Disallowed residues as a string of one letter amino acid codes, e.g. 'KEC'. "
        "This is implemented at the inverse fold step, so it only affects results if inverse folding is "
        "enabled. Default: none for protein design, 'C' for peptide and antibody/nanobody design. Pass an empty list if you want Cysteins to be generated if you are using antibody/nanobody/peptide protocol",
    )
    p.add_argument(
        "--only_inverse_fold",
        action="store_true",
        help="Skip design step and only run inverse folding. Requires a fully specified structure.",
    )

    # Folding and affinity prediction configuration options
    p = parser.add_argument_group("folding and affinity prediction")
    p.add_argument(
        "--esmfold2_python",
        default=os.environ.get("BOLTZGEN_ESMFOLD2_PYTHON"),
        help="Optional ESMFold2 interpreter override. By default BoltzGen prepares its runtime automatically.",
    )
    p.add_argument(
        "--esmfold2_acceleration",
        choices=["auto", "fused", "off"],
        default="auto",
        help="ESMFold2 acceleration: auto preserves native numerics; fused also enables native BF16 kernels; off disables acceleration.",
    )
    p.add_argument(
        "--folding_checkpoint",
        type=str,
        help="Path to the folding checkpoint. Default: %(default)s",
        default=ARTIFACTS["folding"][0],
    )
    p.add_argument(
        "--affinity_checkpoint",
        type=str,
        help="Path to the affinity predictor checkpoint. Default: %(default)s",
        default=ARTIFACTS["affinity"][0],
    )

    # Filtering configuration options
    p = parser.add_argument_group("filtering")
    p.add_argument(
        "--budget",
        type=int,
        help="How many designs should be in the final diversity optimized set. This is used in the filtering step.",
        default=30,
    )
    p.add_argument(
        "--alpha",
        type=float,
        help="Trade-off for sequence diversity selection: 0.0=quality-only, 1.0=diversity-only. Default is "
        "0.01 (peptide-anything protocol) or 0.001 (other protocols).",
        default=None,
    )
    p.add_argument(
        "--filter_biased",
        choices=["true", "false"],
        help="Remove amino-acid composition outliers (default caps on ALA/GLY/GLU/LEU/VAL). Default: %(default)s.",
        default="true",
    )
    p.add_argument(
        "--metrics_override",
        nargs="+",
        help="Per-metric inverse-importance weights for ranking. "
        "Format: metric_name=weight (e.g., plip_hbonds_refolded=4 delta_sasa_refolded=2). "
        "A larger value down-weights that metric's rank. Use 'metric_name=none' to remove a metric.",
        default=None,
    )
    p.add_argument(
        "--additional_filters",
        nargs="+",
        help="Extra hard filters. Format: feature>threshold or feature<threshold "
        "(e.g., 'design_ALA>0.3' 'design_GLY<0.2'). Use '>' if higher is better, '<' if lower is better. "
        "Make sure to single-quote the strings so your shell doesn't get confused by < and > characters.",
        default=None,
    )
    p.add_argument(
        "--size_buckets",
        nargs="+",
        help="Optional constraint for maximum number of designs in size ranges. "
        "Format: min-max:count (e.g., 10-20:5 20-30:10 30-40:5).",
        default=None,
    )
    p.add_argument(
        "--refolding_rmsd_threshold",
        type=float,
        help="Threshold used for RMSD-based filters (lower is better).",
        default=None,
    )


def add_models_download_options(p: argparse.ArgumentParser) -> None:
    p = p.add_argument_group("model and data download options")
    p.add_argument(
        "--force_download",
        help="Force a (re)-download of models and data.",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--models_token",
        type=str,
        help="Secret token to use for our models hosting service (Hugging Face). Default: %(default)s",
        default=os.environ.get("HF_TOKEN"),
    )
    p.add_argument(
        "--cache",
        type=Path,
        help="Directory for Boltz models and data. Default: ~/.cache. "
        "ESM checkpoints use HF_HOME/HF_HUB_CACHE; the ESM runtime uses UV_CACHE_DIR.",
        default=None,
    )


def add_execute_core_arguments(p: argparse.ArgumentParser) -> None:
    p = p.add_argument_group("execution options")
    p.add_argument(
        "--no_subprocess",
        dest="subprocess",
        action="store_false",
        help="Run each step in the main process. Will cause issues when devices >1.",
        default=True,
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Run only the specified pipeline steps (default: run all steps)",
    )


def build_run_parser(subparsers) -> argparse.ArgumentParser:
    run_parser = subparsers.add_parser(
        "run",
        description="Boltzgen binder design pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run the binder design pipeline",
        epilog=__doc__,
    )
    group = run_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s), or a directory containing prepared configs",
    )
    add_configure_arguments(run_parser, output_required=False)
    add_execute_core_arguments(run_parser)
    add_models_download_options(run_parser)
    return run_parser


def build_execute_parser(subparsers) -> argparse.ArgumentParser:
    execute_parser = subparsers.add_parser(
        "execute",
        description="Execute a pre-configured pipeline from a directory of config files",
        help="Run pipeline from pre-generated configuration files",
    )
    execute_parser.add_argument(
        "output",
        type=Path,
        help="Directory containing pre-configured pipeline files (generated by 'configure' command)",
    )
    add_execute_core_arguments(execute_parser)
    return execute_parser


def build_configure_parser(subparsers) -> argparse.ArgumentParser:
    configure_parser = subparsers.add_parser(
        "configure",
        description="Generate resolved pipeline configuration files without executing steps",
        help="Create configuration files for later execution",
    )
    group = configure_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )

    group = configure_parser.add_argument_group("steps to configure")
    group.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Configure only the specified pipeline steps (default: all steps)",
    )

    add_configure_arguments(configure_parser, output_required=True)
    add_models_download_options(configure_parser)

    configure_parser.set_defaults(subprocess=False)

    return configure_parser


def build_download_parser(subparsers) -> argparse.ArgumentParser:
    download_parser = subparsers.add_parser(
        "download",
        help="Download boltzgen model weights and supporting assets",
    )
    group = download_parser.add_argument_group(
        "artifacts to download (positional argument)"
    )
    group.add_argument(
        "artifacts",
        nargs="+",
        default=[],
        choices=sorted(ARTIFACTS.keys()) + ["all"],
        help="Subset of artifacts to download, or 'all' to download all artifacts.",
    )
    add_models_download_options(download_parser)
    return download_parser


def build_check_parser(subparsers) -> argparse.ArgumentParser:
    check_parser = subparsers.add_parser(
        "check",
        description="Check design specification files for validity and optionally output mmCIF",
        help="Validate design specification files",
    )
    check_parser.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory to write mmCIF files (optional)",
    )
    check_parser.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )

    add_models_download_options(check_parser)
    return check_parser


def build_merge_parser(subparsers) -> argparse.ArgumentParser:
    merge_parser = subparsers.add_parser(
        "merge",
        description="Merge multiple BoltzGen output directories so filtering can be rerun on the combined set.",
        help="Combine finished pipeline outputs into a single directory",
    )
    merge_parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Paths to completed BoltzGen output directories (results of 'run' or 'execute')",
    )
    merge_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the merged outputs",
    )
    merge_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignored: kept temporarily for backwards compatibility. In all cases, the destination data is overwritten.",
    )
    return merge_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boltzgen",
        description="Boltzgen command line interface",
    )

    # Support: boltzgen -v / --version
    def get_package_version() -> str:
        try:
            return pkg_version("boltzgen")
        except PackageNotFoundError:
            return "unknown"

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"boltzgen {get_package_version()}",
        help="Print version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_run_parser(subparsers)
    build_configure_parser(subparsers)
    build_execute_parser(subparsers)
    build_download_parser(subparsers)
    build_check_parser(subparsers)
    build_merge_parser(subparsers)
    return parser


#### Commands ####
def run_command(args: argparse.Namespace) -> None:
    """
    Run the **complete binder design pipeline** end-to-end by running:
        1. `configure_command(args)` – generates the per-step YAML configurations.
        2. `execute_command(args)` – launches each pipeline step based on those YAMLs.

    Typical CLI usage:
        $ boltzgen run path/to/design.yaml --output out_dir --protocol protein-anything
    """
    # Validate required arguments for running from design specs
    if not args.output:
        print("No output directory specified. Exiting.")
        return

    print("\n=== Configuring pipeline ===")
    configure_command(args)

    print("\n=== Executing pipeline ===")
    execute_command(args)


def download_command(args: argparse.Namespace) -> list[Path]:
    """
    Download model checkpoints and molecule data from their configured sources.

    Parameters
    ----------
    args.artifacts : list[str]
        List of artifact keys to download, or `["all"]` to fetch all available assets.
    args.models_token : str
        Optional authentication token for private model access (default: env var `HF_TOKEN`).
    args.cache : Path
        Cache directory for storing downloaded artifacts.

    Returns
    -------
    list[Path]
        Local file paths of successfully downloaded artifacts.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen download all
        $ boltzgen download design-diverse inverse-fold
    """
    selections = sorted(set(args.artifacts))
    if "all" in selections:
        selections = list(ARTIFACTS.keys())

    download_paths: list[Path] = []
    for name in selections:
        artifact, repo_type = ARTIFACTS[name]
        resolved_path = get_artifact_path(args, artifact, repo_type=repo_type)
        print(f"Downloading {name} to {resolved_path}")
        download_paths.append(resolved_path)

    return download_paths


def configure_command(args: argparse.Namespace) -> None:
    """Write the shared pipeline recipe as CLI configuration files."""
    configure_pipeline(
        args,
        resolve_artifact=get_artifact_path,
        load_molecules=load_canonicals,
        validate_specs=check_design_specs,
        pipeline_factory=BinderDesignPipeline,
        accelerator=accelerator_type,
    )


def check_command(args: argparse.Namespace) -> None:
    """
    Validate **design specification YAML files** and write cif file for visualization.

    This command parses input design specs using `YamlDesignParser`, verifies structure
    integrity, highlights unresolved residues, and  outputs colored mmCIF visualizations.

    Parameters
    ----------
    args.design_spec : list[Path]
        Input YAML file(s) describing the binder design.
    args.output : Path, optional
        Directory to write mmCIF outputs (optional).
    args.moldir : str
        Path or Hugging Face reference to the molecule dataset.

    Typical CLI usage:
        $ boltzgen check path/to/design.yaml --output checked/

    This function does **not** execute the pipeline — it only validates inputs.
    """
    moldir = get_artifact_path(args, args.moldir, repo_type="dataset")
    mols = load_canonicals(moldir=moldir)

    if args.output:
        output_dir = args.output
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(
                    f"Output path exists and is not a directory: {output_dir}"
                )
        else:
            print(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True)

    check_design_specs(args, moldir, mols)


def execute_command(args: argparse.Namespace) -> None:
    """Execute the configured pipeline via the shared runner."""
    execute_pipeline(args)


#### Pipeline implementation ####
class BinderDesignPipeline(_BinderDesignPipeline):
    """CLI compatibility wrapper around the shared protocol recipe."""

    def __init__(self, args: argparse.Namespace, moldir: Path) -> None:
        super().__init__(args, moldir, resolve_artifact=get_artifact_path)


def merge_command(args: argparse.Namespace) -> None:
    """Delegate legacy CLI merging to the shared pipeline implementation."""
    merge_directories(args.sources, args.output)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "configure":
        configure_command(args)
    elif args.command == "execute":
        execute_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "check":
        check_command(args)
    elif args.command == "merge":
        merge_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
