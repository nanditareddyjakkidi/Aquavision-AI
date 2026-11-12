#!/usr/bin/env python3
"""Write comparison artifacts for an already completed BatchNorm ablation.

This is evaluation-only: it loads the existing epoch-2/3/4 checkpoints and the
original matching checkpoints, runs MPS inference on frozen validation pairs,
and writes no model or dataset files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only finalizer.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch


PROJECT = Path(__file__).resolve().parents[1]
PREPARATION = PROJECT / "preparation"
RUNS_ROOT = PROJECT / "training_runs/funie"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"

sys.path.insert(0, str(PREPARATION))
from run_funie_batchnorm_ablation import (  # noqa: E402
    FrozenUIEBPairs,
    compare_epoch,
    read_frozen_rows,
    require_mps,
    sha256,
    write_comparison_report,
)


def verify_frozen_batchnorm_checkpoint(checkpoint: Path, epoch: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) + 1 != epoch or not payload.get("batchnorm_statistics_frozen"):
        raise RuntimeError(f"{checkpoint.name} is not the expected BatchNorm-frozen epoch-{epoch} checkpoint")
    for model_name, expected_key in (("generator", "batchnorm_epoch1_generator_state"), ("discriminator", "batchnorm_epoch1_discriminator_state")):
        expected = payload.get(expected_key)
        state = payload.get(model_name)
        if not isinstance(expected, dict) or not isinstance(state, dict):
            raise RuntimeError(f"{checkpoint.name} lacks recorded BatchNorm provenance")
        for name, buffers in expected.items():
            for field, expected_value in buffers.items():
                state_name = f"{name}.{field}"
                if state_name not in state or not torch.equal(state[state_name].detach().cpu(), expected_value.detach().cpu()):
                    raise RuntimeError(f"{checkpoint.name} changed {model_name} BatchNorm {state_name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed separate BatchNorm ablation run")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    run_dir = RUNS_ROOT / args.run_id
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoints = run_dir / "checkpoints"
    comparison_dir = run_dir / "comparison_against_original"
    if comparison_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing comparison artifacts: {comparison_dir}")
    if not (config_path.is_file() and metrics_path.is_file() and FREEZE_FILE.is_file()):
        raise SystemExit("Expected completed ablation configuration, metrics, and frozen manifest")
    configuration = json.loads(config_path.read_text())
    if configuration.get("format") != "aqua_vision_funie_batchnorm_frozen_v1" or configuration.get("frozen_manifest", {}).get("sha256") != sha256(FREEZE_FILE):
        raise RuntimeError("Run configuration does not match the expected frozen BatchNorm ablation")
    source_run_dir = PROJECT / configuration["source_run"]["path"]
    source_checkpoints = source_run_dir / "checkpoints"
    if not source_run_dir.is_dir():
        raise RuntimeError("Recorded original run directory is unavailable")
    copied_source = run_dir / "source_checkpoint/epoch_001.pt"
    if not copied_source.is_file() or sha256(copied_source) != configuration["source_run"]["checkpoint_sha256"]:
        raise RuntimeError("Separate-run copy of the epoch-1 source checkpoint is missing or changed")
    recorded_epochs = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if [int(row["epoch"]) for row in recorded_epochs] != [2, 3, 4]:
        raise RuntimeError("Ablation run does not contain exactly completed epochs 2–4")
    for epoch in (2, 3, 4):
        ablation_checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
        original_checkpoint = source_checkpoints / f"epoch_{epoch:03d}.pt"
        if not original_checkpoint.is_file():
            raise RuntimeError(f"Original epoch-{epoch} checkpoint is unavailable")
        verify_frozen_batchnorm_checkpoint(ablation_checkpoint, epoch)

    validation_rows = read_frozen_rows("val")
    if len(validation_rows) != 134:
        raise RuntimeError("Unexpected frozen validation-pair count")
    source_metrics = [json.loads(line) for line in (source_run_dir / "metrics.jsonl").read_text().splitlines() if line]
    fixed_names = list(source_metrics[0]["comparison_filenames"])
    color_failure_names = ["725_img_.png", "809_img_.png", "832_img_.png", "727_img_.png", "920_img_.png"]
    available_names = {row["filename"] for row in validation_rows}
    if not set(fixed_names + color_failure_names).issubset(available_names):
        raise RuntimeError("A requested fixed-preview or color-shift case is absent from frozen validation")

    device = require_mps()
    comparison_dir.mkdir(parents=True)
    validation_dataset = FrozenUIEBPairs(validation_rows, train=False, seed=int(configuration["seed"]))
    comparisons = [
        compare_epoch(epoch, source_checkpoints / f"epoch_{epoch:03d}.pt", checkpoints / f"epoch_{epoch:03d}.pt", validation_dataset, fixed_names, color_failure_names, device, comparison_dir)
        for epoch in (2, 3, 4)
    ]
    summary = {
        "source_run": str(source_run_dir.relative_to(PROJECT)),
        "source_epoch1_checkpoint_sha256": configuration["source_run"]["checkpoint_sha256"],
        "baseline": configuration["raw_validation_baseline"],
        "comparisons": comparisons,
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "note": "Evaluation-only finalization; no model optimization occurred. BatchNorm epoch-1 buffers were verified unchanged in every ablation checkpoint.",
    }
    (comparison_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_comparison_report(comparison_dir / "FINDINGS.md", summary["baseline"], comparisons, summary["source_run"])
    print(json.dumps({"status": "passed", "comparison_dir": str(comparison_dir.relative_to(PROJECT)), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
