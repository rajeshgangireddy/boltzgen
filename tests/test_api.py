"""Public in-process pipeline contract with lightweight stage writers."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import boltzgen.api as api
from boltzgen.api import (
    ArtifactRef,
    BoltzGenEngine,
    PipelinePlan,
    PipelineRequest,
    PipelineResumeError,
    PipelineRun,
    PipelineStageError,
    PipelineValidationError,
    StageResult,
)
from boltzgen.task.esmfold2.contract import (
    ESMC_REVISION,
    ESM_VERSION,
    MODEL_REVISION,
    PTM_KEY,
    REDESIGN_SCORE_KEY,
    SCHEMA_VERSION,
    SCORE_KEY,
    fingerprint,
)

PROTOCOLS = (
    "protein-anything",
    "peptide-anything",
    "nanobody-anything",
    "antibody-anything",
    "protein-small_molecule",
    "protein-redesign",
)


def test_installed_git_revision_is_preserved_in_pipeline_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "site-packages/boltzgen/api.py"
    module.parent.mkdir(parents=True)
    module.touch()
    revision = "a" * 40
    monkeypatch.setattr(api, "__file__", str(module))

    class Installed:
        def read_text(self, filename: str) -> str:
            assert filename == "direct_url.json"
            return json.dumps({"vcs_info": {"commit_id": revision}})

    monkeypatch.setattr(api, "distribution", lambda name: Installed())

    assert api._source_revision() == revision


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    spec = tmp_path / "input.yaml"
    spec.write_text("entities:\n  - protein: {id: A, sequence: 10}\n")
    moldir = tmp_path / "mols.zip"
    moldir.write_bytes(b"local molecule archive")
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"local weights")
    return dict(
        design_spec=spec,
        output_dir=tmp_path / "run",
        num_designs=2,
        budget=2,
        device="cpu",
        seed=13,
        design_checkpoints=(checkpoint, checkpoint),
        folding_checkpoint=checkpoint,
        solublempnn_checkpoint=checkpoint,
        boltzif_checkpoint=checkpoint,
        affinity_checkpoint=checkpoint,
        moldir=moldir,
        esmfold2_python=Path(sys.executable),
    )


def _sample(request: PipelineRequest) -> Path:
    return api._design_dir(request)


def _make_fake_stage(
    request: PipelineRequest,
    calls: list[str],
    *,
    fail: str | None = None,
):
    def execute(config) -> None:
        stage = os.environ["BOLTZGEN_PIPELINE_STEP"]
        calls.append(stage)
        if fail == stage:
            torch.set_grad_enabled(False)
            torch.set_float32_matmul_precision("medium")
            os.environ["PL_GLOBAL_SEED"] = "changed"
            raise RuntimeError("model stopped early")
        root = request.output_dir
        base = _sample(request)
        raw = root / "intermediate_designs"
        if stage in ("design", "inverse_folding"):
            destination = raw if stage == "design" else base
            destination.mkdir(parents=True, exist_ok=True)
            for index in range(request.num_designs):
                stem = f"input_{index}"
                (destination / f"{stem}.cif").write_text(f"candidate:{stem}")
                (destination / f"{stem}.npz").write_bytes(b"metadata")
        elif stage in ("folding", "design_folding", "affinity"):
            names = {
                "folding": ("fold_out_npz", "refold_cif"),
                "design_folding": ("fold_out_design_npz", "refold_design_cif"),
                "affinity": ("affinity_out_npz", None),
            }
            archive, cif = names[stage]
            (base / archive).mkdir(parents=True, exist_ok=True)
            if cif:
                (base / cif).mkdir(parents=True, exist_ok=True)
            for stem in api._candidate_files(base):
                if stage == "affinity":
                    np.savez(
                        base / archive / f"{stem}.npz",
                        affinity_probability_binary1=np.array([[0.71]]),
                    )
                else:
                    (base / archive / f"{stem}.npz").write_bytes(b"prediction")
                if cif:
                    (base / cif / f"{stem}.cif").write_text(f"refold:{stem}")
        elif stage == "esmfold2_scoring":
            directory = base / "esmfold2_scores"
            directory.mkdir(parents=True, exist_ok=True)
            for stem, cif in api._candidate_files(base).items():
                incoming = {
                    "design_id": stem,
                    "design_sha256": api._sha256(cif),
                    "scoring_mode": "redesign"
                    if request.protocol == "protein-redesign"
                    else "binder",
                }
                metrics = (
                    {REDESIGN_SCORE_KEY: 0.8, PTM_KEY: 0.8}
                    if request.protocol == "protein-redesign"
                    else {
                        SCORE_KEY: 0.8,
                        "esmfold2_design_to_target_ipsae": 0.8,
                        "esmfold2_target_to_design_ipsae": 0.8,
                    }
                )
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "model_revision": MODEL_REVISION,
                    "esmc_revision": ESMC_REVISION,
                    "esm_version": ESM_VERSION,
                    "scoring_mode": incoming["scoring_mode"],
                    "input_hash": fingerprint(incoming),
                    "metrics": metrics,
                }
                if request.protocol == "protein-redesign":
                    result["score_metric"] = PTM_KEY
                (directory / f"{stem}.input.json").write_text(json.dumps(incoming))
                (directory / f"{stem}.json").write_text(json.dumps(result))
                (directory / f"{stem}.cif").write_text(f"esm:{stem}")
                (directory / f"{stem}.npz").write_bytes(b"esm metadata")
        elif stage == "analysis":
            ids = sorted(api._candidate_files(base))
            rows = []
            for stem in ids:
                row = {"id": stem, "file_name": f"{stem}.cif"}
                if request.protocol == "protein-small_molecule":
                    row["affinity_probability_binary1"] = 0.71
                else:
                    row["esmfold2_input_hash"] = fingerprint(
                        json.loads(
                            (
                                base / "esmfold2_scores" / f"{stem}.input.json"
                            ).read_text()
                        )
                    )
                    if request.protocol == "protein-redesign":
                        row[REDESIGN_SCORE_KEY] = 0.8
                        row["esmfold2_score_metric"] = PTM_KEY
                    else:
                        row[SCORE_KEY] = 0.8
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                base / "aggregate_metrics_analyze.csv", index=False
            )
            pd.DataFrame([{"id": stem, "sequence": "GG"} for stem in ids]).to_pickle(
                base / "ca_coords_sequences.pkl.gz",
            )
        elif stage == "filtering":
            budget = int(config.budget)
            folder = Path(config.outdir) / "final_ranked_designs"
            final = folder / f"final_{budget}_designs"
            (final / "before_refolding").mkdir(parents=True, exist_ok=True)
            ids = sorted(api._candidate_files(base))
            analyzed = pd.read_csv(base / "aggregate_metrics_analyze.csv")
            pool = []
            width = len(str(len(ids)))
            if config.alpha == 0.2:
                ids.reverse()
            for rank, stem in enumerate(ids, 1):
                name = f"rank{rank:0{width}d}_{stem}.cif"
                if rank <= budget:
                    (final / name).write_text(
                        (base / "refold_cif" / f"{stem}.cif").read_text()
                    )
                    (final / "before_refolding" / name).write_text(
                        (base / f"{stem}.cif").read_text(),
                    )
                original = analyzed[analyzed["id"] == stem].iloc[0].to_dict()
                pool.append(
                    {
                        **original,
                        "final_rank": rank,
                        "quality_score": 1 - (rank - 1) / (len(ids) - 1)
                        if len(ids) > 1
                        else 1.0,
                        "pass_filter_rmsd_filter": True,
                    }
                )
            pd.DataFrame(pool[:budget]).to_csv(
                folder / f"final_designs_metrics_{budget}.csv", index=False
            )
            pd.DataFrame(pool).to_csv(folder / "all_designs_metrics.csv", index=False)
        else:
            raise AssertionError(stage)

    return execute


def _stub_real_work(
    monkeypatch: pytest.MonkeyPatch, request: PipelineRequest, calls: list[str]
) -> None:
    monkeypatch.setattr(api, "_validate_native_spec", lambda _request: None)
    monkeypatch.setattr(
        api,
        "_esm_assets",
        lambda _request: {"esmfold2_python": Path(sys.executable)},
    )
    monkeypatch.setattr(api, "run_task", _make_fake_stage(request, calls))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_plan_and_recipe_use_all_protocol_defaults(inputs: dict, protocol: str) -> None:
    request = PipelineRequest(protocol=protocol, **inputs)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        assert plan.stages[0] == "design"
        assert plan.stages[-2:] == ("analysis", "filtering")
        assert ("design_folding" in plan.stages) == (
            protocol in ("protein-anything", "protein-small_molecule")
        )
        assert ("affinity" in plan.stages) == (protocol == "protein-small_molecule")
        assert ("esmfold2_scoring" in plan.stages) == (
            protocol != "protein-small_molecule"
        )
        configs = api._recipe(request, plan.stages)
        assert list(configs) == list(plan.stages)
        assert configs["folding"].trainer.accelerator == "cpu"
        assert configs["folding"].trainer.devices == 1
        assert configs["design"].seed == 13
        assert configs["filtering"].random_state == 13
        assert configs["analysis"].num_processes == 1
        if protocol == "protein-small_molecule":
            assert configs["inverse_folding"]._target_.endswith("predict.Predict")
            assert configs["analysis"].affinity_metrics
            assert configs["filtering"].use_affinity
        else:
            assert configs["inverse_folding"]._target_.endswith("PredictSolubleMPNN")
            assert configs["esmfold2_scoring"].seed == 13
            assert configs["esmfold2_scoring"].scoring_mode == (
                "redesign" if protocol == "protein-redesign" else "binder"
            )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_public_pipeline_completes_each_protocol_with_stage_writers(
    inputs: dict, monkeypatch: pytest.MonkeyPatch, protocol: str
) -> None:
    request = PipelineRequest(protocol=protocol, **inputs)
    calls: list[str] = []
    _stub_real_work(monkeypatch, request, calls)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        assert isinstance(plan, PipelinePlan)
        run = engine.run(plan)
    assert isinstance(run, PipelineRun)
    assert run.completed and run.final_csv.is_file()
    assert calls == list(plan.stages)
    assert [stage.name for stage in run.stages] == calls
    assert all(
        isinstance(stage, StageResult)
        and stage.status == "completed"
        and stage.files
        and stage.elapsed_seconds >= 0
        and stage.device
        == ("cpu" if stage.name in ("analysis", "filtering") else request.device)
        and all(
            isinstance(file, ArtifactRef) and file.path.is_file()
            for file in stage.files
        )
        for stage in run.stages
    )
    expected_score = (
        "affinity_probability_binary1"
        if protocol == "protein-small_molecule"
        else REDESIGN_SCORE_KEY
        if protocol == "protein-redesign"
        else SCORE_KEY
    )
    assert expected_score in pd.read_csv(run.final_csv).columns
    assert len(run.selected) == request.budget
    for selected in run.selected:
        assert selected.source_id == selected.id
        assert selected.source_run == request.output_dir
        assert selected.cif.read_text() == f"refold:{selected.id}"
        assert selected.before_refolding_cif.read_text() == f"candidate:{selected.id}"
        if protocol == "protein-small_molecule":
            assert selected.esmfold2_cif is None
            assert selected.esmfold2_json is None
            assert selected.affinity_npz == (
                _sample(request) / "affinity_out_npz" / f"{selected.id}.npz"
            )
            with np.load(selected.affinity_npz) as evidence:
                assert evidence["affinity_probability_binary1"].shape == (1, 1)
                score = (
                    pd.read_csv(run.final_csv)
                    .set_index("id")
                    .loc[selected.id, expected_score]
                )
                assert evidence[expected_score].item() == pytest.approx(score)
        else:
            assert selected.esmfold2_cif.read_text() == f"esm:{selected.id}"
            assert selected.esmfold2_json.is_file()
            assert selected.affinity_npz is None


def test_native_affinity_writer_preserves_probability_key_and_shape(
    tmp_path: Path,
) -> None:
    from boltzgen.task.predict.writer import AffinityWriter

    writer = AffinityWriter(str(tmp_path))
    writer.write_on_batch_end(
        prediction={"affinity_probability_binary1": torch.tensor([[0.71]])},
        batch={"id": ["input_0"]},
    )
    archive = tmp_path / "affinity_out_npz" / "input_0.npz"
    with np.load(archive) as evidence:
        assert evidence.files == ["affinity_probability_binary1"]
        assert evidence["affinity_probability_binary1"].shape == (1, 1)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_api_recipe_preserves_cli_protocol_defaults(
    inputs: dict, protocol: str
) -> None:
    from boltzgen.cli import boltzgen as cli

    request = PipelineRequest(protocol=protocol, **inputs)
    args = cli.build_parser().parse_args(
        [
            "configure",
            str(request.design_spec),
            "--output",
            str(request.output_dir),
            "--moldir",
            str(request.moldir),
            "--protocol",
            protocol,
            "--devices",
            "1",
            "--seed",
            str(request.seed),
            "--num_designs",
            str(request.num_designs),
            "--budget",
            str(request.budget),
            "--design_checkpoints",
            *map(str, request.design_checkpoints),
            "--folding_checkpoint",
            str(request.folding_checkpoint),
            "--solublempnn_checkpoint",
            str(request.solublempnn_checkpoint),
            "--inverse_fold_checkpoint",
            str(request.boltzif_checkpoint),
            "--affinity_checkpoint",
            str(request.affinity_checkpoint),
            "--esmfold2_python",
            str(request.esmfold2_python),
        ]
    )
    cli_configs = {
        step.name: step.get_config()
        for step in cli.BinderDesignPipeline(args, request.moldir).steps
    }
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        direct_configs = api._recipe(request, plan.stages)
    assert tuple(cli_configs) == plan.stages
    for stage in plan.stages:
        cli_config = cli_configs[stage]
        direct = direct_configs[stage]
        assert cli_config._target_ == direct._target_
        if stage in (
            "design",
            "inverse_folding",
            "folding",
            "design_folding",
            "affinity",
        ):
            assert cli_config.checkpoint == direct.checkpoint
            assert cli_config.trainer.devices == direct.trainer.devices
        if stage == "analysis":
            assert cli_config.esmfold2_metrics == direct.esmfold2_metrics
            assert cli_config.affinity_metrics == direct.affinity_metrics
            assert cli_config.designfolding_metrics == direct.designfolding_metrics
        if stage == "filtering":
            assert cli_config.use_affinity == direct.use_affinity
            assert cli_config.get("esmfold2_redesign", False) == direct.get(
                "esmfold2_redesign", False
            )
            assert cli_config.metrics_override == direct.metrics_override
        if stage == "esmfold2_scoring":
            assert cli_config.scoring_mode == direct.scoring_mode
            assert cli_config.seed == direct.seed


def test_api_rejects_unsafe_configuration_before_weights(inputs: dict) -> None:
    request = PipelineRequest(
        **inputs, step_options={"design": {"_target_": "os.system"}}
    )
    with BoltzGenEngine() as engine, pytest.raises(
        PipelineValidationError, match="Unreviewed"
    ):
        engine.plan(request)
    request = PipelineRequest(
        **inputs, step_options={"folding": {"trainer.devices": 2}}
    )
    with BoltzGenEngine() as engine, pytest.raises(
        PipelineValidationError, match="Unreviewed"
    ):
        engine.plan(request)
    with pytest.raises(TypeError):
        request.step_options["folding"]["trainer.devices"] = 3
    with pytest.raises(FrozenInstanceError):
        request.seed = 12
    request = PipelineRequest(**{**inputs, "device": "cuda:0,cuda:1"})
    with BoltzGenEngine() as engine, pytest.raises(
        PipelineValidationError, match="multiple"
    ):
        engine.plan(request)


def test_rejects_ambiguous_typed_options_and_hydra_path_interpolation(
    inputs: dict,
) -> None:
    cases = (
        ({"step_options": {"analysis": {"num_processes": 2}}}, "Unreviewed"),
        ({"step_options": {"filtering": {"peptide_type": ["cyclic"]}}}, "Invalid"),
        ({"step_options": {"design": {"sampling_steps": True}}}, "integer"),
        ({"additional_filters": ("score>nan",)}, "finite"),
        ({"size_buckets": ("20-10:2",)}, "valid ranges"),
        ({"precision": "16-mixed"}, "silent bf16 fallback"),
        (
            {"output_dir": inputs["output_dir"].parent / "${oc.env:HOME}"},
            "interpolation",
        ),
        ({"filter_biased": "false"}, "boolean"),
    )
    with BoltzGenEngine() as engine:
        for overrides, message in cases:
            with pytest.raises(PipelineValidationError, match=message):
                engine.plan(PipelineRequest(**{**inputs, **overrides}))


def test_explicit_selected_device_and_reviewed_step_options(inputs: dict) -> None:
    request = PipelineRequest(
        **inputs,
        step_options={
            "design": {"recycling_steps": 2},
            "esmfold2_scoring": {"lm_dropout": 0.25, "scoring_target_chains": ("A",)},
            "filtering": {"peptide_type": "cyclic"},
        },
    )
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        config = api._recipe(request, plan.stages)
    assert config["design"].recycling_steps == 2
    assert config["esmfold2_scoring"].scoring_target_chains == ["A"]
    assert config["esmfold2_scoring"].lm_dropout == 0.25
    assert config["filtering"].peptide_type == "cyclic"
    assert config["folding"].trainer.accelerator == "cpu"
    assert config["folding"].trainer.devices == 1


def test_ligand_default_boltzif_accepts_inverse_folding_options(inputs: dict) -> None:
    request = PipelineRequest(
        **inputs,
        protocol="protein-small_molecule",
        step_options={"inverse_folding": {"recycling_steps": 2}},
    )
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        config = api._recipe(request, plan.stages)
    assert config["inverse_folding"].recycling_steps == 2


def test_quoted_design_spec_and_checkpoint_paths_keep_identity(inputs: dict) -> None:
    folder = inputs["design_spec"].parent / "quote's,[] space"
    folder.mkdir()
    spec = folder / "target,yaml.yaml"
    spec.write_text("entities: []\n")
    checkpoint = folder / "weights' set,2.ckpt"
    checkpoint.write_bytes(b"new weights")
    request = PipelineRequest(
        **{
            **inputs,
            "design_spec": spec,
            "design_checkpoints": (inputs["design_checkpoints"][0], checkpoint),
        }
    )
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        config = api._recipe(request, plan.stages)
    assert config["design"].data.cfg.yaml_path == [str(spec)]
    assert config["design"].override.checkpoints.checkpoint_list[
        0
    ].checkpoint.path == str(checkpoint)


def test_partial_resume_filter_only_and_stale_artifact(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    calls: list[str] = []
    _stub_real_work(monkeypatch, request, calls)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        partial = engine.run(plan, through="inverse_folding")
        assert not partial.completed and partial.final_csv is None
        assert calls == ["design", "inverse_folding"]
        assert all(stage.status == "completed" for stage in partial.stages)
        assert partial.resume_handle.is_file()
        assert partial.stages[0].files[0].path.is_file()
        done = engine.run(plan, resume_from=partial)
        assert done.completed and done.final_csv.is_file()
        assert len(done.selected) == request.budget
        assert all(
            item.source_id == item.id and item.esmfold2_json.is_file()
            for item in done.selected
        )
        assert all(
            item.cif.is_file() and item.before_refolding_cif.is_file()
            for item in done.selected
        )
        assert any(stage.status == "reused" for stage in done.stages)
        assert calls.count("design") == 1
        previous_csv = done.final_csv.read_bytes()
        previous_cif = done.selected[0].cif.read_bytes()
        previous_manifest = done.resume_handle.read_bytes()
        assert done.final_csv == (
            request.output_dir
            / "filter_runs/v0001/final_ranked_designs/final_designs_metrics_2.csv"
        )
        updated = replace(request, alpha=0.2, budget=1)
        reranked = engine.run(
            engine.plan(updated),
            stages=("filtering",),
            resume_from=done.resume_handle,
        )
        assert reranked.completed and len(reranked.selected) == 1
        assert (
            reranked.final_csv
            == request.output_dir
            / "filter_runs/v0002/final_ranked_designs/final_designs_metrics_1.csv"
        )
        assert calls[-1] == "filtering" and calls.count("analysis") == 1
        assert done.final_csv.read_bytes() == previous_csv
        assert done.selected[0].cif.read_bytes() == previous_cif
        assert done.resume_handle.read_bytes() == previous_manifest
        assert reranked.selected[0].id != done.selected[0].id
        with pytest.raises(PipelineResumeError, match="older filtering snapshot"):
            engine.run(engine.plan(updated), resume_from=done.resume_handle)
        (request.output_dir / "intermediate_designs/input_0.cif").write_text("swapped")
        with pytest.raises(PipelineResumeError, match="changed or was swapped"):
            engine.run(engine.plan(updated), resume_from=reranked)


def test_rejects_malformed_or_swapped_manifest_inputs(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        engine.run(plan, through="design")
        path = request.output_dir / "pipeline-manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text("[]")
        with pytest.raises(PipelineResumeError, match="pinned inputs"):
            engine.run(plan, resume_from=request.output_dir)
        manifest["inputs"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with pytest.raises(PipelineResumeError, match="pinned inputs"):
            engine.run(plan, resume_from=request.output_dir)


def test_only_inverse_fold_and_skip_inverse_folding_have_distinct_dependencies(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BoltzGenEngine() as engine:
        only = PipelineRequest(
            **{**inputs, "only_inverse_fold": True, "protocol": "protein-redesign"}
        )
        _stub_real_work(monkeypatch, only, [])
        plan = engine.plan(only)
        assert plan.stages[:2] == ("inverse_folding", "folding")
        run = engine.run(plan, through="inverse_folding")
        assert [stage.name for stage in run.stages] == ["inverse_folding"]
        assert run.final_csv is None and not run.completed

        skip = PipelineRequest(
            **{
                **inputs,
                "output_dir": inputs["output_dir"].parent / "skipped",
                "skip_inverse_folding": True,
                "protocol": "peptide-anything",
            }
        )
        calls: list[str] = []
        _stub_real_work(monkeypatch, skip, calls)
        plan = engine.plan(skip)
        assert plan.stages[:2] == ("design", "folding")
        run = engine.run(plan)
        assert run.completed and "inverse_folding" not in calls
        assert all(item.cif.is_file() for item in run.selected)
        with pytest.raises(PipelineValidationError, match="cannot be combined"):
            engine.plan(replace(skip, only_inverse_fold=True))


def test_asset_fingerprints_require_explicit_stage_reruns(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    calls: list[str] = []
    _stub_real_work(monkeypatch, request, calls)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        partial = engine.run(plan, through="design")
        inputs["design_checkpoints"][0].write_bytes(b"different local weights")
        with pytest.raises(PipelineResumeError, match="pinned asset.*changed"):
            engine.run(plan, resume_from=partial)
        rerun = engine.run(plan, stages=("design",), resume_from=partial)
        assert rerun.stages[0].status == "completed"
        assert calls == ["design", "design"]
        manifest = json.loads(rerun.resume_handle.read_text())
        assert manifest["stage_receipts"]["design"]["assets"]["design_checkpoints.0"][
            "sha256"
        ] == api._sha256(inputs["design_checkpoints"][0])
        done = engine.run(plan, resume_from=rerun)
        assert done.completed


def test_polymer_inverse_folding_checkpoint_path_is_pinned(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    alternate = request.design_spec.parent / "alternate-solublempnn.pt"
    alternate.write_bytes(request.solublempnn_checkpoint.read_bytes())
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        partial = engine.run(plan, through="inverse_folding")
        updated = replace(request, solublempnn_checkpoint=alternate)
        assert engine.plan(updated).fingerprint == plan.fingerprint
        with pytest.raises(
            PipelineResumeError, match="inverse_folding: settings changed"
        ):
            engine.run(engine.plan(updated), resume_from=partial)


def test_plan_identity_defers_stage_settings_and_assets_to_receipts(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(protocol="protein-small_molecule", **inputs)
    alternate = request.design_spec.parent / "alternate.ckpt"
    alternate.write_bytes(request.folding_checkpoint.read_bytes())
    filtering = replace(
        request, budget=1, step_options={"filtering": {"filter_cysteine": True}}
    )
    changed_stages = {
        "design": (
            replace(
                request, design_checkpoints=(alternate, *request.design_checkpoints[1:])
            ),
            replace(request, step_options={"design": {"sampling_steps": 3}}),
        ),
        "inverse_folding": (
            replace(request, boltzif_checkpoint=alternate),
            replace(request, step_options={"inverse_folding": {"recycling_steps": 2}}),
        ),
        "folding": (
            replace(request, folding_checkpoint=alternate),
            replace(request, step_options={"folding": {"sampling_steps": 3}}),
        ),
        "affinity": (
            replace(request, affinity_checkpoint=alternate),
            replace(request, step_options={"affinity": {"sampling_steps": 3}}),
        ),
    }
    changed_seed = replace(request, seed=request.seed + 1)
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        assert all(
            engine.plan(updated).fingerprint == plan.fingerprint
            for updated in (
                filtering,
                *(
                    variant
                    for changes in changed_stages.values()
                    for variant in changes
                ),
            )
        )
        assert engine.plan(changed_seed).fingerprint != plan.fingerprint
        incomplete = engine.run(plan, through="design")
        with pytest.raises(
            PipelineResumeError, match="filtering: missing completed dependencies"
        ):
            engine.run(
                engine.plan(filtering), stages=("filtering",), resume_from=incomplete
            )
        assert not (request.output_dir / "filter_runs").exists()
        partial = engine.run(plan, through="analysis", resume_from=incomplete)
        with pytest.raises(PipelineResumeError, match="Resume protocol, pinned inputs"):
            engine.run(engine.plan(changed_seed), resume_from=partial)
        for stage, changes in changed_stages.items():
            for updated in changes:
                affected = "folding|design_folding" if stage == "folding" else stage
                with pytest.raises(
                    PipelineResumeError, match=rf"^(?:{affected}): settings changed"
                ):
                    engine.run(engine.plan(updated), resume_from=partial)
        filtered = engine.run(engine.plan(filtering), resume_from=partial)
        assert filtered.completed
        assert filtered.final_csv == (
            request.output_dir
            / "filter_runs/v0001/final_ranked_designs/final_designs_metrics_1.csv"
        )
        with pytest.raises(PipelineResumeError, match="Resume protocol, pinned inputs"):
            engine.run(
                engine.plan(changed_seed),
                stages=("filtering",),
                resume_from=filtered,
            )
        refilter = replace(filtering, budget=2)
        analysis_csv = _sample(request) / "aggregate_metrics_analyze.csv"
        original = analysis_csv.read_bytes()
        analysis_csv.write_text("stale analysis")
        with pytest.raises(
            PipelineResumeError, match="analysis: artifact changed or was swapped"
        ):
            engine.run(
                engine.plan(refilter), stages=("filtering",), resume_from=filtered
            )
        analysis_csv.write_bytes(original)
        reranked = engine.run(
            engine.plan(refilter), stages=("filtering",), resume_from=filtered
        )
        assert reranked.completed
        assert reranked.final_csv == (
            request.output_dir
            / "filter_runs/v0002/final_ranked_designs/final_designs_metrics_2.csv"
        )
        for stage, changes in changed_stages.items():
            for updated in changes:
                with pytest.raises(
                    PipelineResumeError,
                    match="settings changed|Immutable filtering results",
                ):
                    engine.run(
                        engine.plan(updated),
                        stages=(stage,),
                        resume_from=reranked,
                    )
        with pytest.raises(PipelineResumeError, match="Immutable filtering results"):
            engine.run(engine.plan(refilter), stages=("folding",), resume_from=reranked)


def test_scoring_rerun_invalidates_downstream_receipts(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    calls: list[str] = []
    _stub_real_work(monkeypatch, request, calls)
    with BoltzGenEngine() as engine:
        done = engine.run(engine.plan(request), through="analysis")
        rescoring = replace(
            request,
            step_options={"esmfold2_scoring": {"num_loops": 3}},
        )
        plan = engine.plan(rescoring)
        with pytest.raises(PipelineResumeError, match="settings changed"):
            engine.run(plan, resume_from=done)
        partial = engine.run(plan, stages=("esmfold2_scoring",), resume_from=done)
        assert not partial.completed and partial.final_csv is None
        assert [stage.name for stage in partial.stages] == list(plan.stages[:-2])
        restored = engine.run(plan, resume_from=partial)
        assert restored.completed
        assert calls.count("analysis") == 2
        assert calls.count("folding") == 1
        with pytest.raises(PipelineResumeError, match="Immutable filtering results"):
            engine.run(plan, stages=("esmfold2_scoring",), resume_from=restored)


def test_refilter_history_detects_old_export_and_snapshot_tampering(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        first = engine.run(engine.plan(request))
        updated = replace(request, alpha=0.2)
        second = engine.run(
            engine.plan(updated), stages=("filtering",), resume_from=first
        )
        assert first.final_csv != second.final_csv
        assert first.resume_handle != second.resume_handle
        old_cif = first.selected[0].cif
        contents = old_cif.read_bytes()
        old_cif.write_text("stale selected structure")
        with pytest.raises(
            PipelineResumeError, match="artifact changed or was swapped"
        ):
            engine.run(engine.plan(updated), resume_from=second)
        old_cif.write_bytes(contents)

        original = first.resume_handle.read_bytes()
        first.resume_handle.write_bytes(original + b"\n")
        with pytest.raises(PipelineResumeError, match="filtering snapshot changed"):
            engine.run(engine.plan(updated), resume_from=second)


def test_partial_artifacts_and_incomplete_manifest_cannot_resume(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        partial = engine.run(plan, through="design")
        orphan = request.output_dir / "intermediate_designs_inverse_folded/input_0.cif"
        orphan.parent.mkdir(parents=True)
        orphan.write_text("interrupted inverse folding")
        with pytest.raises(PipelineResumeError, match="unreceipted partial artifacts"):
            engine.run(plan, resume_from=partial)
        orphan.unlink()
        next_manifest = partial.resume_handle.with_name(".pipeline-manifest.json.next")
        next_manifest.write_text("{}")
        with pytest.raises(PipelineResumeError, match="Incomplete manifest"):
            engine.run(plan, resume_from=partial)


def test_failed_filter_scores_cannot_be_reported_as_completed(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    calls: list[str] = []
    fake = _make_fake_stage(request, calls)

    def bad_score(config) -> None:
        fake(config)
        if os.environ["BOLTZGEN_PIPELINE_STEP"] == "filtering":
            csv = (
                Path(config.outdir) / "final_ranked_designs/final_designs_metrics_2.csv"
            )
            table = pd.read_csv(csv)
            table.loc[0, SCORE_KEY] = 0.01
            table.to_csv(csv, index=False)

    _stub_real_work(monkeypatch, request, calls)
    monkeypatch.setattr(api, "run_task", bad_score)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        with pytest.raises(
            PipelineStageError, match="stale esmfold2_ipsae_min"
        ) as error:
            engine.run(plan)
        assert error.value.result.status == "failed"
        assert (
            "filtering"
            not in json.loads(
                (request.output_dir / "pipeline-manifest.json").read_text()
            )["stage_receipts"]
        )
        monkeypatch.setattr(api, "run_task", fake)
        resumed = engine.run(plan, resume_from=request.output_dir)
        assert resumed.completed and len(resumed.selected) == request.budget


def test_ligand_analysis_rejects_stale_affinity_score(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(protocol="protein-small_molecule", **inputs)
    calls: list[str] = []
    fake = _make_fake_stage(request, calls)
    _stub_real_work(monkeypatch, request, calls)

    def bad_analysis(config) -> None:
        fake(config)
        if os.environ["BOLTZGEN_PIPELINE_STEP"] == "analysis":
            metrics = _sample(request) / "aggregate_metrics_analyze.csv"
            table = pd.read_csv(metrics)
            table.loc[0, "affinity_probability_binary1"] = 0.2
            table.to_csv(metrics, index=False)

    monkeypatch.setattr(api, "run_task", bad_analysis)
    with BoltzGenEngine() as engine, pytest.raises(
        PipelineStageError, match="stale affinity_probability_binary1"
    ) as error:
        engine.run(engine.plan(request), through="analysis")
    assert error.value.stage == "analysis"


def test_ligand_merge_preserves_affinity_score_evidence(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    with BoltzGenEngine() as engine:
        for index in range(2):
            source = PipelineRequest(
                **{
                    **inputs,
                    "protocol": "protein-small_molecule",
                    "output_dir": inputs["output_dir"].parent / f"ligand-{index}",
                    "seed": index,
                }
            )
            _stub_real_work(monkeypatch, source, [])
            runs.append(engine.run(engine.plan(source), through="analysis"))
        destination = replace(source, output_dir=inputs["output_dir"])
        merged = engine.merge(destination, runs=runs)
        for design_id in merged.stages[0].design_ids:
            archive = (
                destination.output_dir
                / "intermediate_designs_inverse_folded"
                / "affinity_out_npz"
                / f"{design_id}.npz"
            )
            assert archive.is_file()
            with np.load(archive) as evidence:
                assert evidence["affinity_probability_binary1"].shape == (1, 1)
                assert evidence["affinity_probability_binary1"].item() == 0.71
        assert sum(
            "affinity_out_npz" in file.name for file in merged.stages[0].files
        ) == len(merged.stages[0].design_ids)
        _stub_real_work(monkeypatch, destination, [])
        filtered = engine.run(
            engine.plan(destination), stages=("filtering",), resume_from=merged
        )
        assert filtered.completed and filtered.final_csv.is_file()
        assert filtered.stages[0].status == "reused"
        for item in filtered.selected:
            assert item.affinity_npz == (
                destination.output_dir
                / "intermediate_designs_inverse_folded"
                / "affinity_out_npz"
                / f"{item.id}.npz"
            )
            assert item.affinity_npz.is_file()
            with np.load(item.affinity_npz) as evidence:
                assert evidence["affinity_probability_binary1"].shape == (1, 1)
        old_csv = filtered.final_csv.read_bytes()
        archive = next(
            file.path
            for file in merged.stages[0].files
            if "affinity_out_npz" in file.name
        )
        original = archive.read_bytes()
        archive.write_bytes(b"changed prediction")
        with pytest.raises(
            PipelineResumeError, match="artifact changed or was swapped"
        ):
            engine.run(engine.plan(destination), resume_from=filtered)
        archive.write_bytes(original)
        updated = replace(destination, budget=1, alpha=0.2)
        reranked = engine.run(
            engine.plan(updated), stages=("filtering",), resume_from=filtered
        )
        assert reranked.completed
        assert reranked.final_csv == (
            destination.output_dir
            / "filter_runs/v0002/final_ranked_designs/final_designs_metrics_1.csv"
        )
        assert filtered.final_csv.read_bytes() == old_csv
        for item in reranked.selected:
            assert item.affinity_npz == (
                destination.output_dir
                / "intermediate_designs_inverse_folded"
                / "affinity_out_npz"
                / f"{item.id}.npz"
            )
            assert item.affinity_npz.is_file()
            with np.load(item.affinity_npz) as evidence:
                assert evidence["affinity_probability_binary1"].shape == (1, 1)
                assert evidence["affinity_probability_binary1"].item() == 0.71


def test_offline_flags_rng_and_timing_restore_even_on_failure(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("UV_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("BOLTZGEN_TIMING_FILE", "caller-timing.jsonl")
    environment = {
        key: os.environ.get(key)
        for key in (
            "HF_HUB_OFFLINE",
            "UV_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "BOLTZGEN_TIMING_FILE",
            "PL_GLOBAL_SEED",
            "CUEQ_NEW_TEST_FLAG",
            "UNRELATED_TEST_FLAG",
        )
    }
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    grad = torch.is_grad_enabled()
    matmul = torch.get_float32_matmul_precision()

    def assert_restored() -> None:
        assert {key: os.environ.get(key) for key in environment} == environment
        assert random.getstate() == py_rng
        assert np.array_equal(np.random.get_state()[1], np_rng[1])
        assert torch.equal(torch.random.get_rng_state(), torch_rng)
        assert torch.is_grad_enabled() == grad
        assert torch.get_float32_matmul_precision() == matmul

    request = PipelineRequest(**inputs)
    calls: list[str] = []
    fake = _make_fake_stage(request, calls)
    _stub_real_work(monkeypatch, request, calls)

    def preflight(_request: PipelineRequest) -> dict[str, Path]:
        assert all(
            os.environ[key] == "1"
            for key in ("HF_HUB_OFFLINE", "UV_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        return {"esmfold2_python": Path(sys.executable)}

    def run_stage(config) -> None:
        assert all(
            os.environ[key] == "1"
            for key in ("HF_HUB_OFFLINE", "UV_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        os.environ["CUEQ_NEW_TEST_FLAG"] = "set during run"
        os.environ["UNRELATED_TEST_FLAG"] = "set during run"
        random.random()
        np.random.random()
        torch.rand(1)
        fake(config)

    monkeypatch.setattr(api, "_esm_assets", preflight)
    monkeypatch.setattr(api, "run_task", run_stage)
    with BoltzGenEngine() as engine:
        assert engine.run(engine.plan(request), through="esmfold2_scoring").stages
    assert_restored()

    failed = replace(request, output_dir=request.output_dir.parent / "failed-run")
    broken = _make_fake_stage(failed, [], fail="design")

    def failed_stage(config) -> None:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        os.environ["CUEQ_NEW_TEST_FLAG"] = "failed"
        os.environ["UNRELATED_TEST_FLAG"] = "failed"
        broken(config)

    def failed_rollup() -> None:
        raise RuntimeError("timing storage failed")

    monkeypatch.setattr(api, "run_task", failed_stage)
    monkeypatch.setattr(api, "flush_rollup", failed_rollup)
    with BoltzGenEngine() as engine, pytest.raises(
        PipelineStageError, match="model stopped early"
    ):
        engine.run(engine.plan(failed), through="design")
    assert_restored()


def test_cpu_run_does_not_seed_unused_accelerators(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(protocol="protein-small_molecule", **inputs)
    _stub_real_work(monkeypatch, request, [])
    seeded: list[tuple[str, int]] = []
    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        if backend is not None:
            method = "manual_seed" if name == "mps" else "manual_seed_all"
            monkeypatch.setattr(
                backend,
                method,
                lambda seed, backend_name=name: seeded.append((backend_name, seed)),
            )
    with BoltzGenEngine() as engine:
        run = engine.run(engine.plan(request), through="design")
    assert run.stages[0].status == "completed"
    assert seeded == []


def test_esm_preflight_uses_only_the_offline_worker(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    ccd = request.design_spec.parent / "ccd.pkl"
    ccd.write_bytes(b"ccd")
    model = request.design_spec.parent / "esmfold2"
    esmc = request.design_spec.parent / "esmc"
    for directory in (model, esmc):
        directory.mkdir()
        (directory / "weights.safetensors").write_bytes(b"weights")
    calls: list[list[str]] = []

    def local_ccd(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        return str(ccd)

    def probe(command, **kwargs):
        calls.append(command)
        assert all(
            os.environ[key] == "1"
            for key in ("HF_HUB_OFFLINE", "UV_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        assert kwargs["check"] is True and kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0)

    from boltzgen.task.esmfold2.contract import ESMC_REPO, MODEL_REPO

    def local_model(repo, **kwargs):
        assert kwargs["local_files_only"] is True
        assert repo in (MODEL_REPO, ESMC_REPO)
        return str(model if repo == MODEL_REPO else esmc)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", local_ccd)
    monkeypatch.setattr("huggingface_hub.snapshot_download", local_model)
    monkeypatch.setattr(api.subprocess, "run", probe)
    with api._run_state(request):
        assets = api._esm_assets(request)
    assert assets["esmfold2_ccd"] == ccd
    assert assets["esmfold2_model"] == model
    assert assets["esmc_model"] == esmc
    assert len(calls) == 1
    assert calls[0][:2] == [str(request.esmfold2_python), "-I"]
    assert Path(calls[0][2]).name == "worker.py"
    assert calls[0][-3:] == ["--device", "cpu", "--check-runtime"]


def test_stage_selection_checks_dependencies_and_failed_state(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = PipelineRequest(**inputs)
    calls: list[str] = []
    _stub_real_work(monkeypatch, request, calls)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        with pytest.raises(PipelineResumeError, match="missing completed dependencies"):
            engine.run(plan, stages=("filtering",))
        monkeypatch.setattr(
            api, "run_task", _make_fake_stage(request, calls, fail="design")
        )
        with pytest.raises(PipelineStageError) as error:
            engine.run(plan, through="design", resume_from=request.output_dir)
        assert error.value.stage == "design" and error.value.result.status == "failed"
        assert torch.is_grad_enabled()
        assert "PL_GLOBAL_SEED" not in os.environ
        monkeypatch.setattr(api, "run_task", _make_fake_stage(request, calls))
        resumed = engine.run(plan, through="design", resume_from=request.output_dir)
        assert not resumed.completed and resumed.stages[0].design_ids == (
            "input_0",
            "input_1",
        )


def test_api_does_not_import_cli_or_launch_pipeline_process(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    command = [
        sys.executable,
        "-c",
        "import sys; from boltzgen.api import BoltzGenEngine; "
        "assert not any(name.startswith('boltzgen.cli') for name in sys.modules)",
    ]
    subprocess.run(command, env=env, check=True)
    request = PipelineRequest(protocol="protein-small_molecule", **inputs)
    _stub_real_work(monkeypatch, request, [])
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("No pipeline subprocess"),
    )
    with BoltzGenEngine() as engine:
        run = engine.run(engine.plan(request), through="design")
        assert run.stages[0].status == "completed"


def test_compatible_merge_preserves_source_identity(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    calls = []
    with BoltzGenEngine() as engine:
        for index in range(2):
            request = PipelineRequest(
                **{
                    **inputs,
                    "output_dir": inputs["output_dir"].parent / f"source-{index}",
                    "seed": index,
                }
            )
            _stub_real_work(monkeypatch, request, calls)
            runs.append(
                engine.run(
                    engine.plan(request), through="analysis" if index == 0 else None
                )
            )
        merged_request = replace(
            request, output_dir=inputs["output_dir"].parent / "combined", budget=3
        )
        with pytest.raises(PipelineValidationError, match="design_spec.*required"):
            engine.merge(replace(merged_request, design_spec=None), runs=runs)
        merged = engine.merge(merged_request, runs=runs)
        assert not merged.completed and merged.final_csv is None
        assert merged.stages[0].status == "merged"
        assert len(merged.stages[0].design_ids) == 4
        _stub_real_work(monkeypatch, merged_request, calls)
        filtered = engine.run(
            engine.plan(merged_request),
            stages=("filtering",),
            resume_from=merged,
        )
        assert filtered.completed
        assert filtered.final_csv.is_file()
        assert filtered.stages[0].status == "reused"
        assert len(filtered.selected) == 3
        assert all(
            item.source_id in {"input_0", "input_1"} for item in filtered.selected
        )
        assert {item.source_run for item in filtered.selected} <= {
            run.output_dir for run in runs
        }
        final = pd.read_csv(filtered.final_csv).set_index("id")
        for item in filtered.selected:
            assert item.affinity_npz is None
            assert item.esmfold2_cif.read_text() == f"esm:{item.source_id}"
            incoming = json.loads(
                (item.esmfold2_json.parent / f"{item.id}.input.json").read_text()
            )
            score = json.loads(item.esmfold2_json.read_text())
            assert incoming["design_sha256"] == api._sha256(
                _sample(merged_request) / f"{item.id}.cif"
            )
            assert score["input_hash"] == fingerprint(incoming)
            assert score["merged_from"]["design_id"] == item.source_id
            assert score["metrics"][SCORE_KEY] == pytest.approx(
                final.loc[item.id, SCORE_KEY]
            )
        evidence = filtered.selected[0].esmfold2_json
        original = evidence.read_bytes()
        evidence.write_text("stale score")
        with pytest.raises(
            PipelineResumeError, match="artifact changed or was swapped"
        ):
            engine.run(engine.plan(merged_request), resume_from=filtered)
        evidence.write_bytes(original)
        refilter_request = replace(merged_request, budget=4)
        with pytest.raises(PipelineValidationError, match="design_spec.*required"):
            engine.plan(replace(refilter_request, design_spec=None))
        with BoltzGenEngine() as restarted:
            reranked = restarted.run(
                restarted.plan(refilter_request),
                stages=("filtering",),
                resume_from=filtered.resume_handle,
            )
        assert reranked.final_csv == (
            merged_request.output_dir
            / "filter_runs/v0002/final_ranked_designs/final_designs_metrics_4.csv"
        )
        assert len(reranked.selected) == 4
        assert filtered.final_csv.is_file()


def test_merge_checks_source_compatibility_and_freezes_lineage(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BoltzGenEngine() as engine:
        source = PipelineRequest(
            **{**inputs, "output_dir": inputs["output_dir"].parent / "source"}
        )
        _stub_real_work(monkeypatch, source, [])
        run = engine.run(engine.plan(source), through="analysis")
        destination = replace(source, output_dir=inputs["output_dir"].parent / "merged")
        with pytest.raises(PipelineResumeError, match="protocol and inputs"):
            engine.merge(
                replace(destination, protocol="protein-small_molecule"), runs=[run]
            )
        merged = engine.merge(destination, runs=[run])
        assert merged.stages[0].status == "merged"
        assert any(file.source_id == "input_0" for file in merged.stages[0].files)
        with pytest.raises(PipelineResumeError, match="nested merges"):
            engine.merge(
                replace(
                    destination, output_dir=destination.output_dir.parent / "nested"
                ),
                runs=[merged],
            )

        manifest = json.loads(merged.resume_handle.read_text())
        manifest["source_runs"]["source_input_0"]["source_id"] = "swapped"
        merged.resume_handle.write_text(json.dumps(manifest))
        _stub_real_work(monkeypatch, destination, [])
        with pytest.raises(PipelineResumeError, match="merged source lineage changed"):
            engine.run(
                engine.plan(destination), stages=("filtering",), resume_from=merged
            )


def test_nested_native_spec_change_invalidates_resume(
    inputs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure = inputs["design_spec"].with_suffix(".cif")
    structure.write_text("original")
    inputs["design_spec"].write_text(
        "entities:\n  - file: {path: input.cif, include: all}\n",
    )
    request = PipelineRequest(**inputs)
    _stub_real_work(monkeypatch, request, [])
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        assert any(file.path == structure for file in plan.input_files)
        run = engine.run(plan, through="design")
        structure.write_text("changed")
        with pytest.raises(
            PipelineValidationError, match="Plan or pinned inputs changed"
        ):
            engine.run(plan, resume_from=run)


def test_nested_native_spec_and_structure_inputs_are_fingerprinted(
    inputs: dict,
) -> None:
    inner = inputs["design_spec"].parent / "inner.yaml"
    structure = inputs["design_spec"].parent / "structure.cif"
    structure.write_text("fixture structure")
    inner.write_text("entities:\n  - file: {path: structure.cif}\n")
    inputs["design_spec"].write_text("entities:\n  - file: {path: inner.yaml}\n")
    request = PipelineRequest(**inputs)
    with BoltzGenEngine() as engine:
        plan = engine.plan(request)
        assert {file.path for file in plan.input_files} >= {
            inputs["design_spec"],
            inner,
            structure,
            inputs["moldir"],
        }
        inner.write_text("entities:\n  - file: {path: missing.cif}\n")
        with pytest.raises(PipelineValidationError, match="Missing local design input"):
            engine.plan(request)
