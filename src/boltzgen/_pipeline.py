"""Shared protocol recipes, local assets and stage orchestration for BoltzGen."""

from __future__ import annotations

import collections
import gc
import json
import shlex
import subprocess
import sys
import time
import yaml
import hydra
from boltzgen.task.task import Task
from boltzgen.utils.timing import (
    DEFAULT_TIMING_FILENAME,
    flush_rollup,
    print_timing_summary,
    record_timing,
    set_timing_file,
)
import math
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import huggingface_hub
import huggingface_hub.constants
import numpy as np
import omegaconf
import pandas as pd
import torch
from rdkit import Chem

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.utils.device import accelerator_type, device_capability_safe, device_count

### Paths and constants ####
path_to_script = Path(__file__)
project_root = path_to_script.resolve().parent
config_dir = project_root / "resources/config"
main_script = project_root / "resources/main.py"

step_names = [
    "design",
    "inverse_folding",
    "design_folding",
    "folding",
    "affinity",
    "esmfold2_scoring",
    "analysis",
    "filtering",
]

### Protocol-specific configuration overrides (which can be overridden by user) ####
protocol_configs = {
    "protein-anything": {},  # base config corresponds to protein-anything
    "peptide-anything": {
        # Note that in inverse folding step we also avoid cysteines by default; this is implemented elsewhere.
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": [
            "filter_cysteine=true",
            "alpha=0.01",
            "refolding_rmsd_threshold=2",
        ],
    },
    "protein-small_molecule": {
        "analysis": ["affinity_metrics=true"],
        "filtering": ["use_affinity=true"],
    },
    "nanobody-anything": {
        "analysis": [
            "largest_hydrophobic=false",
            "largest_hydrophobic_refolded=false",
            "liability_modality=antibody",
        ],
        "filtering": ["filter_cysteine=true", "modality=antibody"],
    },
    "antibody-anything": {
        "analysis": [
            "largest_hydrophobic=false",
            "largest_hydrophobic_refolded=false",
            "liability_modality=antibody",
        ],
        "filtering": ["filter_cysteine=true", "modality=antibody"],
    },
    "protein-redesign": {
        "esmfold2_scoring": ["scoring_mode=redesign"],
        # For redesigning/optimizing existing proteins (e.g., symmetric dimers)
        # where all chains may have designed residues. Skips design_folding and
        # uses design_mask (not chain_design_mask) for target/template definition.
        "folding": ["data.design_mask_templates=true"],
        "analysis": ["use_design_mask_for_target=true"],
        "filtering": [
            "esmfold2_redesign=true",
            "metrics_override={design_ptm: null, plip_hbonds_refolded: null, plip_saltbridge_refolded: null, delta_sasa_refolded: null, plip_hbonds: null, plip_saltbridge: null, delta_sasa_original: null, esmfold2_score: 1, neg_filter_rmsd_design: 4}",
        ],
    },
}
assert all(
    step_name in step_names for cfg in protocol_configs.values() for step_name in cfg
)


### Model checkpoints and other artifacts ####
ARTIFACTS: dict[str, tuple[str, str]] = {
    "design-diverse": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
        "model",
    ),
    "design-adherence": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
        "model",
    ),
    "inverse-fold": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt",
        "model",
    ),
    "solublempnn": (
        "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        "model",
    ),
    "folding": (
        "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt",
        "model",
    ),
    "affinity": ("huggingface:boltzgen/boltzgen-1:boltz2_aff.ckpt", "model"),
    "moldir": ("huggingface:boltzgen/inference-data:mols.zip", "dataset"),
}


SOLUBLEMPNN_SHA256 = "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a"


def resolve_inverse_fold_model(args: Any) -> str:
    """Validate inverse-fold settings before loading any pipeline artifacts."""
    if args.only_inverse_fold and args.skip_inverse_folding:
        raise ValueError(
            "--only_inverse_fold cannot be combined with --skip_inverse_folding"
        )
    protocol = args.protocol
    if protocol not in protocol_configs:
        raise ValueError(
            f"Invalid protocol: {protocol}. Valid protocols: {list(protocol_configs.keys())}"
        )

    inverse_fold_model = args.inverse_fold_model or (
        "boltzif" if protocol == "protein-small_molecule" else "solublempnn"
    )
    if protocol == "protein-small_molecule" and inverse_fold_model != "boltzif":
        raise ValueError(
            "protein-small_molecule requires BoltzIF because SolubleMPNN does "
            "not condition on the ligand. Remove --inverse_fold_model solublempnn "
            "or select --inverse_fold_model boltzif."
        )
    use_solublempnn = inverse_fold_model == "solublempnn"
    if use_solublempnn and (args.only_inverse_fold or not args.skip_inverse_folding):
        temperature = args.solublempnn_sampling_temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "SolubleMPNN sampling temperature must be finite and positive"
            )
        canonical_letters = {
            const.prot_token_to_letter[residue] for residue in const.canonical_tokens
        }
        if not set(args.inverse_fold_avoid or "") < canonical_letters:
            raise ValueError(
                "SolubleMPNN --inverse_fold_avoid must use canonical amino-acid "
                "letters and leave at least one amino acid allowed"
            )
    return inverse_fold_model


#### Pipeline implementation ####
@dataclass
class PipelineStep:
    name: str
    config_path: str
    args: List[str]

    def check(self):
        if self.name not in step_names:
            raise ValueError(
                f"Invalid step name: {self.name}. Available steps: {step_names}"
            )
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get_config(self) -> omegaconf.DictConfig:
        config = omegaconf.OmegaConf.create({})
        config_file = omegaconf.OmegaConf.load(self.config_path)
        config = omegaconf.OmegaConf.merge(config, config_file)
        args_cfg = omegaconf.OmegaConf.from_dotlist(self.args)
        config = omegaconf.OmegaConf.merge(config, args_cfg)
        return config


class BinderDesignPipeline:
    """Build protocol-specific Hydra recipes without executing a task."""

    def __init__(
        self,
        args: Any,
        moldir: Path,
        *,
        resolve_artifact: Callable[[Any, str], Path] | None = None,
        selected_steps: set[str] | None = None,
    ) -> None:
        protocol = args.protocol
        use_solublempnn = resolve_inverse_fold_model(args) == "solublempnn"
        resolve_artifact = resolve_artifact or get_artifact_path
        if selected_steps is not None and not selected_steps <= set(step_names):
            raise ValueError(
                f"Unknown pipeline stages: {selected_steps - set(step_names)}"
            )

        def include(name: str) -> bool:
            return selected_steps is None or name in selected_steps

        # Handle use_kernels argument. cuEquivariance kernels are CUDA-only,
        # so device capability only matters (and is only queryable) on CUDA.
        selected_device = getattr(args, "device", None)
        device_capability = (
            torch.cuda.get_device_capability(getattr(args, "device_index", 0))
            if selected_device == "cuda"
            else device_capability_safe()
            if selected_device is None
            else None
        )
        use_kernels = None
        if args.use_kernels == "auto":
            use_kernels = device_capability is not None and device_capability[0] >= 8
        elif args.use_kernels == "true":
            use_kernels = True
            if device_capability is None:
                print(
                    "WARNING: --use_kernels true requested, but cuEquivariance "
                    f"kernels require CUDA (detected accelerator: {accelerator_type()}). "
                    "This will fail unless the package happens to be importable."
                )
        elif args.use_kernels == "false":
            use_kernels = False
        else:
            raise ValueError(f"Invalid use_kernels value: {args.use_kernels}")
        print(f"Using kernels: {use_kernels} [device capability: {device_capability}]")

        protocol_config = protocol_configs[protocol]
        print(f"Config overrides for protocol {protocol}: {protocol_config}")

        # Set protocol-specific and user-specified step specific args
        config_args_by_step = parse_config_args(
            protocol_config, args.config, step_names
        )
        if args.seed is not None:
            for step_name in (
                "design",
                "inverse_folding",
                "folding",
                "design_folding",
                "affinity",
                "esmfold2_scoring",
            ):
                config_args_by_step[step_name].append(f"seed={args.seed}")
            config_args_by_step["filtering"].append(f"random_state={args.seed}")

        devices = args.devices if args.devices is not None else device_count()
        print(f"Using {devices} devices")

        self.steps = []

        # Design generation
        output_dir = args.output / "intermediate_designs"
        print(f"Raw designs will be saved to: {output_dir}")
        diffusion_batch_size = args.diffusion_batch_size
        if diffusion_batch_size is None:
            diffusion_batch_size = 1 if args.num_designs < 100 else 10
        num_batches = math.ceil(args.num_designs / diffusion_batch_size)
        print(f"Using diffusion batch size: {diffusion_batch_size}")
        print(f"Number of diffusion batches: {num_batches}")

        checkpoint_args = []
        if not args.only_inverse_fold and include("design"):
            first_checkpoint_path = resolve_artifact(args, args.design_checkpoints[0])
            checkpoint_args.append(f"checkpoint={first_checkpoint_path}")
            if len(args.design_checkpoints) > 1:
                fraction_per_checkpoint = 1.0 / len(args.design_checkpoints)
                checkpoint_args.append(
                    f"override.checkpoints.first_checkpoint_num_samples={fraction_per_checkpoint}"
                )
                checkpoint_args.append(
                    "override.checkpoints.checkpoint_list=["
                    + ",".join(
                        f"{{'checkpoint': {{'num_samples': {fraction_per_checkpoint}, 'path': {json.dumps(str(resolve_artifact(args, checkpoint)))}}}}}"
                        for checkpoint in args.design_checkpoints[1:]
                    )
                    + "]"
                )
        design_step_and_noise_scale_args = []
        if args.step_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.step_scale={args.step_scale}"
            )
            # Also disable the schedule when applying a fixed step scale
            design_step_and_noise_scale_args.append("override.step_scale_schedule=null")
        if args.noise_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.noise_scale={args.noise_scale}"
            )
            design_step_and_noise_scale_args.append(
                "override.noise_scale_schedule=null"
            )

        inverse_fold_model_args = []
        exclude_residues = []
        if include("inverse_folding") and (
            args.only_inverse_fold or not args.skip_inverse_folding
        ):
            inverse_fold_avoid = (
                args.inverse_fold_avoid
                if args.inverse_fold_avoid is not None
                else (
                    "C"
                    if protocol
                    in ["peptide-anything", "nanobody-anything", "antibody-anything"]
                    else ""
                )
            )
            exclude_residues = [
                const.prot_letter_to_token[letter] for letter in inverse_fold_avoid
            ]
            restriction = f"[{', '.join(exclude_residues)}]"
            if use_solublempnn:
                inverse_fold_model_args = [
                    f"checkpoint={resolve_artifact(args, args.solublempnn_checkpoint)}",
                    f"sampling_temperature={args.solublempnn_sampling_temperature}",
                    f"inverse_fold_restriction={restriction}",
                ]
            else:
                inverse_fold_model_args = [
                    f"checkpoint={resolve_artifact(args, args.inverse_fold_checkpoint)}",
                    f"override.use_kernels={use_kernels}",
                    f"override.inverse_fold_args.inverse_fold_restriction={restriction}",
                ]

        if args.only_inverse_fold:
            if exclude_residues:
                print(
                    f"Inverse fold will avoid the following residues: {exclude_residues}"
                )
            print(f"Inverse-folded designs will be saved to: {output_dir}")
            # Designs from inverse folding
            if include("inverse_folding"):
                self.steps.append(
                    PipelineStep(
                        name="inverse_folding",
                        config_path=args.config_dir
                        / (
                            "inverse_fold_only_solublempnn.yaml"
                            if use_solublempnn
                            else "inverse_fold_only.yaml"
                        ),
                        args=[
                            f"output={output_dir}",
                            f"data.cfg.yaml_path=[{', '.join(json.dumps(str(s)) for s in args.design_spec)}]",
                            f"trainer.devices={devices}",
                            f"data.cfg.multiplicity={getattr(args, 'inverse_fold_num_sequences', 10)}",
                            f"data.cfg.moldir={moldir}",
                            f"data.num_workers={args.num_workers}",
                            f"data.cfg.skip_existing={args.reuse}",
                            f"data.cfg.output_dir={output_dir}",
                        ]
                        + inverse_fold_model_args
                        + config_args_by_step.get("inverse_folding", []),
                    )
                )
        else:
            # Designs from diffusion model
            if include("design"):
                self.steps.append(
                    PipelineStep(
                        name="design",
                        config_path=args.config_dir / "design.yaml",
                        args=[
                            f"output={output_dir}",
                            f"data.cfg.yaml_path=[{', '.join(json.dumps(str(s)) for s in args.design_spec)}]",
                            f"trainer.devices={devices}",
                            f"data.num_workers={args.num_workers}",
                            f"data.cfg.skip_existing={args.reuse}",
                            f"data.cfg.multiplicity={num_batches}",
                            f"diffusion_samples={diffusion_batch_size}",
                            f"override.use_kernels={use_kernels}",
                            f"data.cfg.moldir={moldir}",
                        ]
                        + design_step_and_noise_scale_args
                        + checkpoint_args
                        + config_args_by_step["design"],
                    )
                )

            # Inverse folding of diffusion-generated backbones.
            if not args.skip_inverse_folding:
                if len(exclude_residues) > 0:
                    print(
                        f"Inverse fold will avoid the following residues: {exclude_residues}"
                    )

                input_dir = output_dir
                output_dir = args.output / "intermediate_designs_inverse_folded"
                print(f"Inverse-folded designs will be saved to: {output_dir}")
                if include("inverse_folding"):
                    self.steps.append(
                        PipelineStep(
                            name="inverse_folding",
                            config_path=args.config_dir
                            / (
                                "inverse_fold_solublempnn.yaml"
                                if use_solublempnn
                                else "inverse_fold.yaml"
                            ),
                            args=[
                                f"output={output_dir}",
                                f"data.design_dir={input_dir}",
                                f"data.cfg.multiplicity={args.inverse_fold_num_sequences}",
                                f"data.cfg.num_workers={args.num_workers}",
                                f"data.skip_existing={args.reuse}",
                                "data.skip_existing_kind=inverse_fold",
                                f"data.cfg.moldir={moldir}",
                                f"trainer.devices={devices}",
                            ]
                            + inverse_fold_model_args
                            + config_args_by_step["inverse_folding"],
                        )
                    )

        # Folding
        input_dir = output_dir
        if include("folding"):
            self.steps.append(
                PipelineStep(
                    name="folding",
                    config_path=args.config_dir / "fold.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        "data.skip_existing_kind=folded",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={resolve_artifact(args, args.folding_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step["folding"],
                )
            )

        # Design folding
        input_dir = output_dir
        do_design_folding = protocol in ["protein-anything", "protein-small_molecule"]
        if do_design_folding and include("design_folding"):
            self.steps.append(
                PipelineStep(
                    name="design_folding",
                    config_path=args.config_dir / "fold.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        "data.skip_existing_kind=design_folded",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={resolve_artifact(args, args.folding_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                        "writer.designfolding=True",
                        "data.cfg.return_designfolding=True",
                    ]
                    + config_args_by_step["design_folding"],
                )
            )

        # Affinity
        use_affinity = protocol in ["protein-small_molecule"]
        if use_affinity and include("affinity"):
            self.steps.append(
                PipelineStep(
                    name="affinity",
                    config_path=args.config_dir / "affinity.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        "data.skip_existing_kind=affinity",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={resolve_artifact(args, args.affinity_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step["affinity"],
                )
            )

        # Analysis
        if not use_affinity and include("esmfold2_scoring"):
            self.steps.append(
                PipelineStep(
                    name="esmfold2_scoring",
                    config_path=args.config_dir / "esmfold2.yaml",
                    args=[
                        f"design_dir={input_dir}",
                        f"data.cfg.moldir={moldir}",
                        f"python={args.esmfold2_python or 'null'}",
                        f"acceleration='{args.esmfold2_acceleration}'",
                        f"devices={devices}",
                        f"reuse={args.reuse}",
                    ]
                    + config_args_by_step["esmfold2_scoring"],
                )
            )

        if include("analysis"):
            self.steps.append(
                PipelineStep(
                    name="analysis",
                    config_path=args.config_dir / "analysis.yaml",
                    args=[
                        f"design_dir={input_dir}",
                        f"data.skip_existing={args.reuse and use_affinity}",
                        "data.skip_existing_kind=analyzed",
                        f"esmfold2_metrics={not use_affinity}",
                        f"data.cfg.moldir={moldir}",
                        f"designfolding_metrics={do_design_folding}",
                        f"delta_sasa_original={args.skip_inverse_folding}",
                        f"noncovalents_original={args.skip_inverse_folding}",
                        f"allatom_fold_metrics={args.skip_inverse_folding}",
                    ]
                    + config_args_by_step["analysis"],
                )
            )

        # Filtering
        output_dir = args.output / "final_ranked_designs"
        print(f"Final ranked designs will be saved to: {output_dir}")

        # Build filter arguments
        filter_args = [
            f"design_dir={input_dir}",
            f"outdir={args.output}",
            f"from_inverse_folded={not args.skip_inverse_folding}",
            f"use_affinity={use_affinity}",
            f"filter_designfolding={do_design_folding}",
            f"budget={args.budget}",
        ]

        # Add optional filtering arguments
        if args.alpha is not None:
            filter_args.append(f"alpha={args.alpha}")
        if args.filter_biased is not None:
            filter_args.append(f"filter_biased={args.filter_biased}")
        if args.refolding_rmsd_threshold is not None:
            filter_args.append(
                f"refolding_rmsd_threshold={args.refolding_rmsd_threshold}"
            )
        if args.metrics_override is not None:
            parsed_metrics = parse_metrics_override(args.metrics_override)
            print(f"Filtering metrics override: {parsed_metrics}")
            filter_args.append(f"metrics_override={parsed_metrics}")
        if args.additional_filters is not None:
            parsed_filters = parse_additional_filters(args.additional_filters)
            print(f"Filtering additional filters: {parsed_filters}")
            filter_args.append(f"additional_filters={parsed_filters}")
        if args.size_buckets is not None:
            parsed_size_buckets = parse_size_buckets(args.size_buckets)
            print(f"Filtering size buckets: {parsed_size_buckets}")
            filter_args.append(f"size_buckets={parsed_size_buckets}")

        if include("filtering"):
            self.steps.append(
                PipelineStep(
                    name="filtering",
                    config_path=args.config_dir / "filtering.yaml",
                    args=filter_args + config_args_by_step["filtering"],
                )
            )

    def filter_steps(self, enabled_steps: List[str]):
        self.steps = [s for s in self.steps if s.name in enabled_steps]

    def pretty_print(self):
        for i, step in enumerate(self.steps):
            print(f"[{i + 1}] {step.name:25s}")


### Misc utiltiies ###
def check_design_specs(args: Any, moldir: Path, mols: Dict[str, Any]):
    stem_counts = collections.Counter(Path(path).stem for path in args.design_spec)
    duplicates = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "Design input filenames must have unique stems; repeated: "
            + ", ".join(duplicates)
        )
    last_banner = ""
    for design_spec in args.design_spec:
        banner = f"************** Checking design spec: {design_spec} **************"
        last_banner = banner
        print(banner)
        check_design_spec(args, moldir, design_spec, mols)
    if last_banner:
        print("*" * len(last_banner))


def check_design_spec(args: Any, moldir: Path, design_spec: Path, mols: Dict[str, Any]):
    """
    Validate a single design specification, color/annotate its structure, and
    write an mmCIF visualization.

    This function parses a YAML design spec with `YamlDesignParser`, applies visual
    annotations to the parsed structure (via B-factors and per-residue color features),
    reports unresolved residues/atoms, and writes a colored mmCIF file if `--output`
    was provided on the CLI.

    - Writes `<output>/<design_spec.stem>.cif` if `args.output` is set.

        * B-factor encodes design/binding (100 for designed, +80 if binding).
        * `design_color_features`: 1.0 for binding residues, 0.8 otherwise.

    Examples
    --------
    From the CLI (via the `check` subcommand):

        $ boltzgen check path/to/design.yaml --output checked/

    From Python:

        moldir = PATH TO MOLDIR
        mols = load_canonicals(moldir=moldir)
        check_design_spec(args, moldir, Path("design.yaml"), mols)
    """
    parser = YamlDesignParser(moldir)
    parsed = parser.parse_yaml(design_spec, mols, moldir)
    structure = parsed.structure
    design_info = parsed.design_info
    design_color_features = np.ones_like(design_info.res_binding_type) * 0.8
    design_color_features[design_info.res_binding_type.astype(bool)] = 1.0
    extract_mask = np.zeros(len(structure.residues), dtype=bool)
    for i, residue in enumerate(structure.residues):
        structure.atoms["bfactor"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ] = 100 * design_info.res_design_mask[i] + 80 * design_info.res_binding_type[i]

        atom_positions = structure.atoms["coords"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ]
        zero_position_count = ((atom_positions**2).sum(axis=1) < 1e-6).sum()
        if not zero_position_count == len(atom_positions):
            extract_mask[i] = True

    structure_write = structure
    if extract_mask.sum() > 0:
        structure_write = structure.extract_residues(structure, extract_mask)
        design_color_features = design_color_features[extract_mask]

    mmcif = to_mmcif(
        structure_write,
        design_coloring=True,
        color_features=design_color_features,
    )

    # Check for unresolved residues and atoms to log a warning
    unresolved_residues = (~structure.residues["is_present"]).nonzero()[0]
    unresolved_atoms = (~structure.atoms["is_present"]).nonzero()[0]

    print(f"Total designed residues: {design_info.res_design_mask.sum()}")

    if len(unresolved_residues) > 0 or len(unresolved_atoms) > 0:
        atom_to_residue = {}
        for residue_idx, residue in enumerate(structure.residues):
            start = residue["atom_idx"]
            end = start + residue["atom_num"]
            for i in range(start, end):
                atom_to_residue[i] = residue_idx

        residue_to_chain = {}
        residue_in_chain = {}
        for chain in structure.chains:
            start = chain["res_idx"]
            end = start + chain["res_num"]
            j = 1
            for i in range(start, end):
                residue_to_chain[i] = chain["name"]
                residue_in_chain[i] = j
                j += 1

        atom_to_chain = {}
        for chain in structure.chains:
            start = chain["atom_idx"]
            end = start + chain["atom_num"]
            for i in range(start, end):
                atom_to_chain[i] = chain["name"]

        msg = f"There are {len(unresolved_residues)} unresolved residues and {len(unresolved_atoms)} unresolved atoms in the target."
        print(msg)

    if args.output is not None:
        output_path = args.output / (design_spec.stem + ".cif")
    else:
        output_path = design_spec.stem + ".cif"
    with open(output_path, "w") as f:
        f.write(mmcif)
    print(f"Design specification visualization is written to {str(output_path)}")


def get_artifact_path(
    args, artifact: str, repo_type: str = "model", verbose: bool = True
) -> Path:
    """Resolve a local file, Hugging Face artifact, or the upstream SolubleMPNN weights."""
    if artifact == ARTIFACTS["solublempnn"][0]:
        cache = (
            Path(args.cache)
            if args.cache is not None
            else Path(huggingface_hub.constants.HF_HUB_CACHE)
        )
        result = cache / "boltzgen" / "solublempnn_v_48_020.pt"
        if args.force_download or not result.exists():
            result.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                artifact, str(result), hash_prefix=SOLUBLEMPNN_SHA256
            )
    elif artifact.startswith("huggingface:"):
        try:
            _, repo_id, filename = artifact.split(":")
        except ValueError:
            raise ValueError(
                f"Invalid artifact: {artifact}. Expected format: huggingface:<repo_id>:<filename>"
            )
        result = huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            repo_type=repo_type,
            library_name="boltzgen",
            force_download=args.force_download,
            token=args.models_token,
            cache_dir=args.cache,
        )
        result = Path(result)
    else:
        result = Path(artifact)
    if not result.exists():
        raise FileNotFoundError(f"Model not found: {result}")
    if verbose:
        print(f"Using {repo_type} artifact: {result}")
    return result


def parse_config_args(base_config, config_args, valid_step_names):
    config_args_by_step = collections.defaultdict(list)
    config_args_by_step.update(
        {step: list(values) for step, values in base_config.items()}
    )
    if config_args:
        for config in config_args:
            if len(config) < 2:
                raise ValueError(
                    f"Invalid config: {config}. Expected format: <step_name> <arg1>=<value1> <arg2>=<value2> ..."
                )
            step_name = config[0]
            if step_name not in valid_step_names:
                raise ValueError(
                    f"Invalid step name: {step_name}. Available steps: {valid_step_names}"
                )
            key_value_pairs = config[1:]
            config_args_by_step[step_name].extend(key_value_pairs)
    return config_args_by_step


### Filtering argument parsing functions ####
def parse_metrics_override(value_list):
    """Parse metrics_override from key=value pairs."""
    if not value_list:
        return None

    metrics_override = {}
    for item in value_list:
        if "=" in item:
            key, value = item.split("=", 1)
            if value == "" or value.lower() == "none":
                metrics_override[key] = None  # Remove metric
            else:
                try:
                    metrics_override[key] = float(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid weight value for metric '{key}': '{value}'. Must be a number."
                    )
        else:
            raise ValueError(
                f"Invalid metrics_override format: '{item}'. Use 'metric_name=weight' format."
            )
    return metrics_override


def parse_additional_filters(value_list):
    """Parse additional_filters from feature>threshold or feature<threshold format."""
    if not value_list:
        return None

    additional_filters = []
    for item in value_list:
        if ">" in item:
            feature, threshold_str = item.split(">", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": False,  # > means higher is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        elif "<" in item:
            feature, threshold_str = item.split("<", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": True,  # < means lower is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        else:
            raise ValueError(
                f"Invalid additional_filters format: '{item}'. Use 'feature>threshold' or 'feature<threshold' format."
            )
    return additional_filters


def parse_size_buckets(value_list):
    """Parse size_buckets from min-max:count format."""
    if not value_list:
        return None

    size_buckets = []
    for item in value_list:
        if ":" in item and "-" in item:
            range_part, count_str = item.split(":", 1)
            if "-" in range_part:
                min_str, max_str = range_part.split("-", 1)
                try:
                    min_size = int(min_str)
                    max_size = int(max_str)
                    count = int(count_str)
                    size_buckets.append(
                        {"num_designs": count, "min": min_size, "max": max_size}
                    )
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(
                            f"Invalid size_buckets format: '{item}'. All values must be integers. Use 'min-max:count' format."
                        )
                    else:
                        raise e
            else:
                raise ValueError(
                    f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
                )
        else:
            raise ValueError(
                f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
            )
    return size_buckets


def _make_new_file_name(original_file: str, new_id: str) -> str:
    suffix = Path(original_file).suffix
    return f"{new_id}{suffix}" if suffix else new_id


def _remap_analyzed_metrics_and_scores(
    metrics_path: Path,
    src_dir: Path,
    dest_dir: Path,
    run_tag: str,
    id_map: dict[tuple[Path, str], str],
) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]]]:
    # Preserve identifiers and valid "NA" sequences when merging.
    sequence_columns = [
        column
        for column in pd.read_csv(metrics_path, nrows=0).columns
        if column == "designed_chain_sequence"
        or column.startswith(("designed_sequence", "full_sequence_"))
    ]
    df = pd.read_csv(
        metrics_path,
        converters={
            "id": str,
            "file_name": str,
            **dict.fromkeys(sequence_columns, lambda value: value or None),
        },
    )
    if df.empty:
        message = (
            f"Metrics file contains no analyzed designs: {metrics_path}. "
            "Rerun analysis before merging this source."
        )
        raise ValueError(message)
    updated_rows = []
    source_mappings: list[tuple[str, str, str, str]] = []
    for _, row in df.iterrows():
        if "id" not in row or "file_name" not in row:
            raise ValueError(
                "aggregate_metrics_analyze.csv must contain 'id' and 'file_name' columns."
            )
        original_id = str(row["id"])
        original_file = str(row["file_name"])
        key = (src_dir.parent, original_id)
        new_id = id_map.setdefault(key, f"{run_tag}_{original_id}")
        new_file = _make_new_file_name(original_file, new_id)
        if pd.notna(row.get("esmfold2_input_hash")):
            from boltzgen.task.esmfold2.contract import SCORE_DIR, copy_renamed_result

            row["esmfold2_input_hash"] = copy_renamed_result(
                src_dir / SCORE_DIR,
                dest_dir / SCORE_DIR,
                original_id,
                new_id,
                src_dir / original_file,
            )
        updated_rows.append({**row, "id": new_id, "file_name": new_file})
        source_mappings.append((original_id, new_id, original_file, new_file))
    return pd.DataFrame(updated_rows), source_mappings


def merge_directories(
    sources: Sequence[Path], output: Path
) -> dict[str, dict[str, str]]:
    """
    Merge multiple BoltzGen output directories into a single destination directory so
    the filtering step can be rerun over the combined set of designs.
    """

    def _merge_design_dir(
        sources: list[Path],
        run_tags: dict[Path, str],
        dir_name: str,
        dest_dir: Path,
        id_map: dict[tuple[Path, str], str],
    ) -> int:
        metrics_frames: list[pd.DataFrame] = []
        seq_frames: list[pd.DataFrame] = []
        per_target_frames: list[pd.DataFrame] = []
        molecule_sources: dict[str, Path] = {}
        merged_count = 0

        for root in sources:
            src_dir = root / dir_name
            if not src_dir.exists():
                continue

            run_tag = run_tags[root]
            source_mappings: list[tuple[str, str, str, str]] = []

            metrics_path = src_dir / "aggregate_metrics_analyze.csv"
            if metrics_path.exists():
                metrics_frame, source_mappings = _remap_analyzed_metrics_and_scores(
                    metrics_path, src_dir, dest_dir, run_tag, id_map
                )
                metrics_frames.append(metrics_frame)
                merged_count += len(source_mappings)
            else:
                # Backbones and inverse-folded sequences have different IDs
                # when several sequences are generated per backbone.
                for path in sorted(src_dir.glob("*.cif")):
                    if not path.is_file() or path.stem.endswith("_native"):
                        continue
                    original_id = path.stem
                    new_id = id_map.setdefault(
                        (root, original_id), f"{run_tag}_{original_id}"
                    )
                    original_file = path.name
                    new_file = _make_new_file_name(original_file, new_id)
                    source_mappings.append(
                        (original_id, new_id, original_file, new_file)
                    )

            if not source_mappings:
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)

            for molecule in sorted((src_dir / const.molecules_dirname).glob("*.pkl")):
                destination = dest_dir / const.molecules_dirname / molecule.name
                previous = molecule_sources.get(molecule.name)
                if previous is None and destination.exists():
                    previous = destination
                if previous is not None:
                    previous_bytes = previous.read_bytes()
                    current_bytes = molecule.read_bytes()
                    if previous_bytes != current_bytes:
                        # Repeated SMILES parsing generates different reference
                        # conformers. Preserve all ordered chemistry/properties
                        # while allowing those stochastic coordinates to differ.
                        flags = (
                            Chem.PropertyPickleOptions.AllProps
                            | Chem.PropertyPickleOptions.NoConformers
                        )
                        previous_mol = pickle.loads(previous_bytes)  # noqa: S301
                        current_mol = pickle.loads(current_bytes)  # noqa: S301
                        if previous_mol.ToBinary(flags) != current_mol.ToBinary(flags):
                            message = (
                                f"Conflicting molecule definition for {molecule.stem}: "
                                f"{molecule} and {previous}"
                            )
                            raise ValueError(message)
                    if molecule.name in molecule_sources:
                        continue
                _copy_path(molecule, destination, required=True)
                molecule_sources[molecule.name] = molecule

            seq_path = src_dir / "ca_coords_sequences.pkl.gz"
            if seq_path.exists():
                seq_df = pd.read_pickle(seq_path)
                original_ids = [orig for orig, _, _, _ in source_mappings]
                seq_subset = seq_df[seq_df["id"].astype(str).isin(original_ids)].copy()
                if not seq_subset.empty:
                    id_lookup = {orig: new for orig, new, _, _ in source_mappings}
                    seq_subset["id"] = seq_subset["id"].astype(str).map(id_lookup)
                    seq_frames.append(seq_subset)

            per_target_path = src_dir / "per_target_metrics_analyze.csv"
            if per_target_path.exists():
                per_target_frames.append(pd.read_csv(per_target_path))

            for original_id, new_id, original_file, new_file in source_mappings:
                _copy_design_files(
                    src_dir=src_dir,
                    dest_dir=dest_dir,
                    original_id=original_id,
                    new_id=new_id,
                    original_file=original_file,
                    new_file=new_file,
                    include_refold=True,
                )

        if metrics_frames:
            pd.concat(metrics_frames, ignore_index=True).to_csv(
                dest_dir / "aggregate_metrics_analyze.csv", index=False
            )
        if seq_frames:
            pd.concat(seq_frames, ignore_index=True).to_pickle(
                dest_dir / "ca_coords_sequences.pkl.gz", compression="gzip"
            )
        if per_target_frames:
            pd.concat(per_target_frames, ignore_index=True).to_csv(
                dest_dir / "per_target_metrics_analyze.csv", index=False
            )

        return merged_count

    def _copy_design_files(
        src_dir: Path,
        dest_dir: Path,
        original_id: str,
        new_id: str,
        original_file: str,
        new_file: str,
        include_refold: bool,
    ) -> None:
        source_stem = Path(original_file).stem
        legacy_source = (
            source_stem.endswith("_gen")
            and not (src_dir / f"{source_stem}.npz").is_file()
        )
        source_prefix = source_stem[:-4] if legacy_source else original_id
        source_metadata_suffix = "_metadata.npz" if legacy_source else ".npz"
        source_metadata = src_dir / f"{source_prefix}{source_metadata_suffix}"
        if (
            legacy_source
            and source_metadata.is_file()
            and any(
                source_metadata.with_suffix(suffix).is_file()
                for suffix in (".cif", ".pdb")
            )
        ):
            message = (
                f"Ambiguous legacy metadata for {original_file}: {source_metadata} "
                "also belongs to a separate coordinate file. Restore the matching "
                "metadata before merging."
            )
            raise ValueError(message)
        if new_id.endswith("_gen") and not source_metadata.is_file():
            legacy_metadata = dest_dir / f"{new_id[:-4]}_metadata.npz"
            if legacy_metadata.is_file():
                # Ownership of this old alias is ambiguous; do not let the
                # reader fall back to it for newly copied coordinates.
                message = (
                    f"Cannot merge incomplete design {original_file} over legacy "
                    f"metadata {legacy_metadata}. Restore the source metadata "
                    "or use a fresh output directory."
                )
                raise ValueError(message)
        _copy_path(src_dir / original_file, dest_dir / new_file, required=True)
        # Publish one canonical layout even for legacy inputs, so replacing a
        # design cannot leave a competing metadata/native alias selected later.
        _copy_path(
            source_metadata,
            dest_dir / f"{new_id}.npz",
            required=False,
        )
        _copy_path(
            src_dir / f"{source_prefix}_native.cif",
            dest_dir / f"{new_id}_native.cif",
            required=False,
        )
        _copy_path(
            src_dir / f"{source_prefix}_native.pdb",
            dest_dir / f"{new_id}_native.pdb",
            required=False,
        )
        if include_refold:
            _copy_path(
                src_dir / const.refold_cif_dirname / original_file,
                dest_dir / const.refold_cif_dirname / new_file,
                required=False,
            )
            _copy_path(
                src_dir / const.refold_design_cif_dirname / original_file,
                dest_dir / const.refold_design_cif_dirname / new_file,
                required=False,
            )
            _copy_path(
                src_dir / const.affinity_dirname / f"{original_id}.npz",
                dest_dir / const.affinity_dirname / f"{new_id}.npz",
                required=False,
            )

    def _slugify_run_tag(path: Path, index: int) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "-", path.name).strip("-").lower()
        return slug or f"run{index}"

    def _copy_path(src: Path, dst: Path, *, required: bool) -> None:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            # Prediction writers can overwrite coordinates in place. A merged
            # run must not share writable files with its source runs.
            shutil.copy2(src, dst)
        elif required:
            raise FileNotFoundError(f"Required file missing during merge: {src}")
        else:
            # Missing companions in a replacement must not retain old data.
            dst.unlink(missing_ok=True)

    if not sources:
        raise ValueError("Provide at least one source directory to merge.")

    dest_root = Path(output).expanduser().resolve()

    source_roots: list[Path] = []
    for src in sources:
        root = Path(src).expanduser().resolve()
        if root == dest_root:
            print(f"Skipping {root} as it is the destination directory")
            continue
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Source directory not found: {root}")
        if root not in source_roots:
            source_roots.append(root)

    dest_root.mkdir(parents=True, exist_ok=True)

    base_tags = {
        root: _slugify_run_tag(root, idx + 1) for idx, root in enumerate(source_roots)
    }
    run_tags: dict[Path, str] = {}
    used_tags: set[str] = set()
    reserved_tags = set(base_tags.values())
    for root, base_tag in base_tags.items():
        tag = base_tag
        if tag in used_tags:
            # Preserve the names of sources whose base tags are already unique.
            index = 2
            while f"{base_tag}-{index}" in used_tags | reserved_tags:
                index += 1
            tag = f"{base_tag}-{index}"
        run_tags[root] = tag
        used_tags.add(tag)
    id_map: dict[tuple[Path, str], str] = {}

    total_designs = 0
    for dir_name in [
        "intermediate_designs_inverse_folded",
        "intermediate_designs",
    ]:
        dest_dir = dest_root / dir_name
        merged = _merge_design_dir(
            sources=source_roots,
            run_tags=run_tags,
            dir_name=dir_name,
            dest_dir=dest_dir,
            id_map=id_map,
        )
        if merged:
            total_designs += merged
            print(f"- merged {merged} designs into {dest_dir}")

    if total_designs == 0:
        print("No analyzed designs available for filtering.")
    else:
        print("===============================================")
        print(f"Merged {len(source_roots)} source(s) into {dest_root}")
        print(f"Total designs available for filtering: {total_designs}")

    return {
        new_id: {"source_run": str(root), "source_id": original_id}
        for (root, original_id), new_id in id_map.items()
    }


def run_task(config: omegaconf.DictConfig) -> None:
    """Instantiate and run one trusted built-in Hydra task in this process."""
    task = hydra.utils.instantiate(config)
    try:
        if not isinstance(task, Task):
            raise TypeError("Config must be an instance of Task.")
        task.run(config)
    finally:
        del task
        gc.collect()


def configure_pipeline(
    args: Any,
    *,
    resolve_artifact: Callable[..., Path] = get_artifact_path,
    load_molecules: Callable[..., Any] = load_canonicals,
    validate_specs: Callable[..., Any] = check_design_specs,
    pipeline_factory: Callable[..., BinderDesignPipeline] = BinderDesignPipeline,
    accelerator: Callable[[], str] = accelerator_type,
) -> None:
    """
    Generate **resolved per-step YAML configuration files** for the binder-design pipeline.

    This command constructs a `BinderDesignPipeline` from user and protocol parameters,
    validates design specifications, and writes out all configuration files required for running the pipeline. It does NOT run the pipeline.

    Outputs
    -------
    * `<output_dir>/config/<step>.yaml` — fully resolved config for each pipeline step.
    * `<output_dir>/steps.yaml` — manifest of steps and config file paths.

    This stage prepares the YAMLs used by `execute_command(...)`. Fresh polymer
    runs also prepare the isolated ESM runtime and check CUDA availability;
    they do not load model weights or run inference here.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen configure path/to/design.yaml --output out_dir --protocol peptide-anything
    """
    resolve_inverse_fold_model(args)
    moldir = resolve_artifact(args, args.moldir, repo_type="dataset")
    mols = load_molecules(moldir=moldir)

    # Setup output directory
    output_dir = args.output
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    else:
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)

    validate_specs(args, moldir, mols)

    pipeline = pipeline_factory(args, moldir)
    if args.steps is not None:
        pipeline.filter_steps(args.steps)

    pipeline.pretty_print()

    # Check that tasks can be instantiated
    for step in pipeline.steps:
        step.check()
        if step.name == "esmfold2_scoring":
            from boltzgen.task.esmfold2.contract import (
                validate_scoring_mode,
                validate_acceleration,
            )
            from boltzgen.task.esmfold2.runtime import resolve_python

            esm_config = step.get_config()
            validate_scoring_mode(
                esm_config.get("scoring_mode", "binder"),
                esm_config.get("scoring_target_chains"),
            )
            validate_acceleration(esm_config.get("acceleration", "auto"))
            # Reuse can finish entirely from saved scores. Provision that run's
            # runtime only if the scoring task finds work still to compute.
            if not esm_config.reuse:
                device_type = accelerator()
                if device_type not in ("cuda", "xpu", "cpu"):
                    raise RuntimeError(
                        "ESMFold2 scoring requires a CUDA, XPU, or CPU device"
                    )
                runtime_kwargs = (
                    {"require_xpu": True}
                    if device_type == "xpu"
                    else {"require_cpu": True}
                    if device_type == "cpu"
                    else {"require_cuda": True}
                )
                resolve_python(esm_config.python, **runtime_kwargs)

    # Make the config subdir in output
    config_dir = output_dir / "config"
    if config_dir.exists():
        # Rename it to be previous-config-XXX
        counter = 1
        while config_dir.with_name(f"previous-config-{counter}").exists():
            counter += 1
        prev_config_dir = config_dir.with_name(f"previous-config-{counter}")
        print(f"Renaming existing config directory to {prev_config_dir}")
        config_dir.rename(prev_config_dir)
    config_dir.mkdir(parents=True)

    # Prepare step configurations and collect step info
    steps_info = []
    for step in pipeline.steps:
        config = step.get_config()
        config_filename = f"{step.name}.yaml"
        config_path = config_dir / config_filename
        with open(config_path, "w") as f:
            omegaconf.OmegaConf.save(config, f)

        # Add step info for steps.yaml (use relative path from output directory)
        steps_info.append(
            {"name": step.name, "config_file": str(config_path.relative_to(output_dir))}
        )

    # Write steps.yaml file
    steps_yaml_path = output_dir / "steps.yaml"
    steps_data = {"steps": steps_info}

    with open(steps_yaml_path, "w") as f:
        yaml.dump(steps_data, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration complete. Configs written to {config_dir}")
    print(f"Steps manifest written to {steps_yaml_path}")


def execute_pipeline(args: Any) -> None:
    """
    Execute a **pre-configured binder design pipeline** from a directory of YAML files.

    Reads the `steps.yaml` manifest written by `configure_command(...)` and executes
    each step sequentially, either in subprocesses or directly in-process.

    Options
    --------
    * `--no_subprocess` : Run in the main process instead of spawning Python subprocesses.
    * `--steps` : Restrict to specific pipeline steps.
    * `--reuse` : Skip recomputation for existing results.

    Expected directory structure (produced by `configure_command(...)`):
        output_dir/
            ├── config/
            │     ├── design.yaml
            │     ├── folding.yaml
            │     └── ...
            └── steps.yaml

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen execute --output out_dir
    """
    config_dir = args.output

    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    # Resolve and propagate the timing file so every step (including any
    # subprocesses/DDP ranks it spawns) appends to the same experiment log.
    timing_filename = getattr(args, "timing_file", None) or DEFAULT_TIMING_FILENAME
    timing_path = Path(timing_filename)
    if not timing_path.is_absolute():
        timing_path = config_dir / timing_path
    set_timing_file(timing_path)
    print(f"Timing measurements will be appended to: {timing_path}")

    # Look for steps.yaml file
    steps_yaml_path = config_dir / "steps.yaml"
    if not steps_yaml_path.exists():
        raise FileNotFoundError(
            f"Steps manifest not found: {steps_yaml_path}. Run 'boltzgen configure' first."
        )

    # Load steps from steps.yaml
    with open(steps_yaml_path, "r") as f:
        steps_data = yaml.safe_load(f)

    if not isinstance(steps_data, dict) or "steps" not in steps_data:
        raise ValueError(f"Invalid steps.yaml format in {steps_yaml_path}")

    # Filter steps if specific steps are requested
    enabled_steps = set(args.steps) if args.steps else None
    resolved_steps: List[Tuple[str, Path]] = []

    for step_info in steps_data["steps"]:
        step_name = step_info["name"]
        config_filename = step_info["config_file"]

        # Skip if this step is not in the enabled steps list
        if enabled_steps is not None and step_name not in enabled_steps:
            continue

        # Build full path to config file
        config_path = config_dir / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        resolved_steps.append((step_name, config_path))

    if not resolved_steps:
        if enabled_steps:
            print(f"No matching steps found for: {', '.join(enabled_steps)}")
        else:
            print(f"No steps found in {steps_yaml_path}")
        return

    total_steps = len(resolved_steps)
    for index, (step_name, config_path) in enumerate(resolved_steps, start=1):
        print("**************************************************")
        print(f"Pipeline step {index} of {total_steps}: {step_name}")

        os.environ["BOLTZGEN_PIPELINE_PROGRESS"] = f"Step {index}/{total_steps}"
        os.environ["BOLTZGEN_PIPELINE_STEP"] = step_name

        start = time.time()
        if args.subprocess:
            command = [
                sys.executable,
                str(main_script),
                str(config_path),
            ]
            print(f"Running command: {shlex.join(command)}")
            subprocess.check_call(command)
        else:
            config = omegaconf.OmegaConf.load(config_path)
            run_task(config)

        elapsed = time.time() - start
        print(f"✓ Step {step_name} completed successfully in {elapsed:.1f}s")
        record_timing(
            "pipeline.step",
            elapsed,
            step_name=step_name,
            step_index=index,
            total_steps=total_steps,
            subprocess=args.subprocess,
        )

    if "BOLTZGEN_PIPELINE_PROGRESS" in os.environ:
        del os.environ["BOLTZGEN_PIPELINE_PROGRESS"]
    if "BOLTZGEN_PIPELINE_STEP" in os.environ:
        del os.environ["BOLTZGEN_PIPELINE_STEP"]

    # Persist this process's own roll-up (the per-step wall times recorded
    # above) without printing it in isolation, then print the complete,
    # merged overview across every pipeline step/subprocess that ran.
    flush_rollup(timing_path.with_suffix(".csv"), print_summary=False)
    print_timing_summary(timing_path.with_suffix(".csv"))
