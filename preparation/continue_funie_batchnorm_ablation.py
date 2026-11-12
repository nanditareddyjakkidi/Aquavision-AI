#!/usr/bin/env python3
"""Continue the frozen-BatchNorm FUnIE-GAN ablation from epoch 4 through 20.

This MPS-only runner accepts only the existing separate BatchNorm ablation
run, restores its completed epoch-4 checkpoint exactly, and trains epochs
5--20.  It preserves every original recipe setting; reapplying BatchNorm eval
mode after each model.train() call remains the sole experimental change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only continuation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
PREPARATION = PROJECT / "preparation"
FUNIE_PYTORCH = PROJECT / "third_party/FUnIE-GAN/PyTorch"
RUNS_ROOT = PROJECT / "training_runs/funie"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
EXPECTED_COMPLETED_EPOCH = 4
FINAL_EPOCH = 20
COLOR_SHIFT_CASES = ["725_img_.png", "809_img_.png", "832_img_.png", "727_img_.png", "920_img_.png"]

sys.path.insert(0, str(PREPARATION))
from run_funie_batchnorm_ablation import (  # noqa: E402
    FrozenUIEBPairs,
    affine_gradient_summary,
    assert_batchnorm_unchanged,
    batchnorm_modules,
    checkpoint_payload,
    compare_epoch,
    freeze_batchnorm_statistics,
    read_frozen_rows,
    save_fixed_preview,
    snapshot_batchnorm,
    verify_ablation_checkpoint,
)
from run_funie_staged import (  # noqa: E402
    atomic_torch_save,
    git_revision,
    metrics_for_pair,
    module_state_matches,
    mps_memory,
    optimizer_resume_summary,
    require_finite,
    require_mps,
    restore_rng,
    sha256,
)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_recorded_bn_buffers(payload: dict[str, Any], model_name: str, expected_key: str) -> None:
    recorded = payload.get(expected_key)
    state = payload.get(model_name)
    if not isinstance(recorded, dict) or not isinstance(state, dict):
        raise RuntimeError(f"Resume checkpoint lacks {model_name} BatchNorm provenance")
    for name, buffers in recorded.items():
        if not isinstance(buffers, dict):
            raise RuntimeError(f"Resume checkpoint has malformed BatchNorm provenance for {name}")
        for field, expected in buffers.items():
            value = state.get(f"{name}.{field}")
            if not isinstance(value, torch.Tensor) or not isinstance(expected, torch.Tensor) or not torch.equal(value.detach().cpu(), expected.detach().cpu()):
                raise RuntimeError(f"Resume checkpoint changed {model_name} BatchNorm {name}.{field}")


def train_with_frozen_batchnorm(
    model: torch.nn.Module,
    expected: dict[str, dict[str, torch.Tensor]],
    label: str,
) -> dict[str, Any]:
    """Apply the experimental behavior after every model.train() call."""
    model.train()
    freeze_batchnorm_statistics(model)
    assert_batchnorm_unchanged(model, expected, f"{label} immediately after model.train()")
    frozen_layers = batchnorm_modules(model)
    if not frozen_layers or any(module.training for _, module in frozen_layers):
        raise RuntimeError(f"{label} BatchNorm modules are not all in eval mode after model.train()")
    if any(not module.affine or module.weight is None or module.bias is None or not module.weight.requires_grad or not module.bias.requires_grad for _, module in frozen_layers):
        raise RuntimeError(f"{label} BatchNorm affine parameters are not trainable")
    return {"model_train_called": True, "batchnorm_layers": len(frozen_layers), "all_batchnorm_eval": True, "affine_parameters_trainable": True}


def write_regression_file(comparison_dir: Path, epoch: int) -> dict[str, Any]:
    metrics_path = comparison_dir / f"per_image_metrics_epoch_{epoch:03d}.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"Comparison did not create {metrics_path.name}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 134:
        raise RuntimeError(f"Expected 134 per-image comparison rows for epoch {epoch}, found {len(rows)}")
    regressions = [
        row
        for row in rows
        if float(row["ablation_psnr_delta"]) < 0.0 or float(row["ablation_ssim_delta"]) < 0.0
    ]
    output = comparison_dir / f"per_image_regressions_epoch_{epoch:03d}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(regressions)
    psnr_deltas = [float(row["ablation_psnr_delta"]) for row in rows]
    ssim_deltas = [float(row["ablation_ssim_delta"]) for row in rows]
    return {
        "per_image_metrics_csv": str(metrics_path.relative_to(PROJECT)),
        "per_image_regressions_csv": str(output.relative_to(PROJECT)),
        "rows": len(rows),
        "regression_rows_any_metric": len(regressions),
        "raw_baseline": {
            "mean_psnr_delta": float(np.mean(psnr_deltas)),
            "median_psnr_delta": float(median(psnr_deltas)),
            "psnr_improved": sum(value > 0.0 for value in psnr_deltas),
            "psnr_worsened": sum(value < 0.0 for value in psnr_deltas),
            "psnr_unchanged": sum(value == 0.0 for value in psnr_deltas),
            "mean_ssim_delta": float(np.mean(ssim_deltas)),
            "median_ssim_delta": float(median(ssim_deltas)),
            "ssim_improved": sum(value > 0.0 for value in ssim_deltas),
            "ssim_worsened": sum(value < 0.0 for value in ssim_deltas),
            "ssim_unchanged": sum(value == 0.0 for value in ssim_deltas),
        },
    }


def source_metric_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_json_lines(path)
    result = {int(row["epoch"]): row for row in rows}
    if set(range(1, FINAL_EPOCH + 1)) - result.keys():
        raise RuntimeError("Original run does not contain every matching epoch through 20")
    return result


def write_continuation_reports(
    run_dir: Path,
    config: dict[str, Any],
    comparisons: list[dict[str, Any]],
    original_metrics: dict[int, dict[str, Any]],
    ablation_metrics: dict[int, dict[str, Any]],
    baseline: dict[str, Any],
) -> None:
    comparison_dir = run_dir / "comparison_against_original"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen-BatchNorm continuation only: resumed epoch 4 and trained epochs 5--20; no extension beyond epoch 20.",
        "source_run": config["source_run"]["path"],
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "raw_validation_baseline": baseline,
        "comparisons": comparisons,
        "note": "BatchNorm running buffers remain frozen at epoch-1 values while affine parameters remain trainable. This targeted ablation does not conclusively establish the cause of the original instability.",
    }
    (comparison_dir / "continuation_summary_epoch_005_to_020.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    curve_rows: list[dict[str, Any]] = []
    table_rows: list[str] = []
    comparison_by_epoch = {int(row["epoch"]): row for row in comparisons}
    for epoch in range(5, FINAL_EPOCH + 1):
        original = original_metrics[epoch]["validation"]
        ablation = ablation_metrics[epoch]["validation"]
        detail = comparison_by_epoch[epoch]
        raw = detail["per_image_regressions"]["raw_baseline"]
        curve_rows.append(
            {
                "epoch": epoch,
                "original_psnr": original["psnr_mean"],
                "ablation_psnr": ablation["psnr_mean"],
                "raw_psnr": baseline["psnr_mean"],
                "original_ssim": original["ssim_mean"],
                "ablation_ssim": ablation["ssim_mean"],
                "raw_ssim": baseline["ssim_mean"],
            }
        )
        table_rows.append(
            "| {epoch} | {apsnr:.4f} | {op:.4f} | {dpsnr:+.4f} | {assim:.5f} | {os:.5f} | {dssim:+.5f} | {rawp}/{raws} | [{csv}]({csv}) |".format(
                epoch=epoch,
                apsnr=ablation["psnr_mean"],
                op=original["psnr_mean"],
                dpsnr=detail["ablation_minus_original"]["mean_psnr"],
                assim=ablation["ssim_mean"],
                os=original["ssim_mean"],
                dssim=detail["ablation_minus_original"]["mean_ssim"],
                rawp=raw["psnr_worsened"],
                raws=raw["ssim_worsened"],
                csv=detail["per_image_regressions"]["per_image_regressions_csv"],
            )
        )
    with (comparison_dir / "continuation_validation_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    report = f"""# Frozen-BatchNorm continuation: epochs 5--20

This report covers the bounded continuation from the preserved BatchNorm
ablation epoch-4 checkpoint through epoch 20. It did not train beyond epoch
20. The original FUnIE-GAN run remains unchanged.

- Frozen UIEB validation pairs: 134; manifest SHA-256:
  `{sha256(FREEZE_FILE)}`.
- Raw baseline on those pairs: {baseline['psnr_mean']:.4f} PSNR and
  {baseline['ssim_mean']:.5f} SSIM.
- The only altered behavior remains frozen BatchNorm running statistics; the
  BatchNorm affine weight/bias parameters remained trainable and were audited.
- Fixed raw/original/BN-frozen/reference sheets and the known color-shift
  cases are in `comparison_sheets/`. Per-image raw-baseline regressions are
  explicitly listed in the CSVs linked below.

| Epoch | BN-frozen PSNR | Original PSNR | Mean Δ PSNR | BN-frozen SSIM | Original SSIM | Mean Δ SSIM | Raw regressions (PSNR/SSIM) | Per-image regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(table_rows)}

Interpretation remains limited: freezing epoch-1 BatchNorm statistics tests a
specific batch-size-one mechanism, but does not by itself prove or disprove the
underlying cause of the original run's instability.
"""
    (comparison_dir / "CONTINUATION_EPOCHS_005_TO_020.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Existing frozen-BatchNorm ablation run directory name")
    parser.add_argument("--resume", type=Path, required=True, help="Its preserved completed epoch-4 checkpoint")
    parser.add_argument("--epochs", type=int, default=FINAL_EPOCH, help="Must be 20; total epoch cap")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--preview-count", type=int, default=5)
    args = parser.parse_args()
    if args.epochs != FINAL_EPOCH or args.preview_count != 5:
        raise SystemExit("This bounded continuation requires --epochs 20 and --preview-count 5")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")

    run_dir = RUNS_ROOT / args.run_id
    checkpoints = run_dir / "checkpoints"
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"
    continuation_config_path = run_dir / "continuation_config_epoch_005_to_020.json"
    resume_events_path = run_dir / "continuation_resume_events.jsonl"
    comparison_dir = run_dir / "comparison_against_original"
    if not (run_dir.is_dir() and checkpoints.is_dir() and config_path.is_file() and metrics_path.is_file() and FREEZE_FILE.is_file()):
        raise SystemExit("Expected the existing frozen-BatchNorm run, checkpoints, metrics, and frozen manifest")
    if continuation_config_path.exists():
        raise SystemExit("This bounded continuation has already been initialized; refusing to overwrite its artifacts")
    if not args.resume.is_file() or run_dir.resolve() not in args.resume.resolve().parents:
        raise SystemExit("--resume must be a checkpoint inside the requested frozen-BatchNorm run")

    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    current_freeze_sha = sha256(FREEZE_FILE)
    if (
        configuration.get("format") != "aqua_vision_funie_batchnorm_frozen_v1"
        or configuration.get("run_id") != args.run_id
        or configuration.get("seed") != args.seed
        or configuration.get("batch_size") != 1
        or configuration.get("frozen_manifest", {}).get("sha256") != current_freeze_sha
        or configuration.get("data", {}).get("train_pairs") != 623
        or configuration.get("data", {}).get("validation_pairs") != 134
    ):
        raise RuntimeError("Saved ablation configuration does not match the controlled frozen UIEB recipe")
    expected_only_change = "After every generator.train()/discriminator.train(), all BatchNorm2d modules are reset to eval mode; affine weight/bias remain requires_grad=True and are optimized."
    if configuration.get("only_change") != expected_only_change:
        raise RuntimeError("The existing run does not have the expected single BatchNorm-only change")
    recorded_metrics = read_json_lines(metrics_path)
    if [int(row["epoch"]) for row in recorded_metrics] != [2, 3, 4]:
        raise RuntimeError("Continuation requires exactly the completed epochs 2--4 and no pre-existing continuation rows")
    if any(row.get("batchnorm", {}).get("statistics_frozen") is not True for row in recorded_metrics):
        raise RuntimeError("Existing ablation metrics lack the required frozen-BatchNorm audit")

    resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    resume_hash = sha256(args.resume)
    canonical_epoch4 = checkpoints / "epoch_004.pt"
    if not canonical_epoch4.is_file() or resume_hash != sha256(canonical_epoch4):
        raise RuntimeError("--resume must be the byte-identical preserved epoch-4 checkpoint (or its latest.pt copy)")
    if (
        resume_payload.get("format") != "aqua_vision_funie_batchnorm_frozen_v1"
        or resume_payload.get("run_id") != args.run_id
        or int(resume_payload.get("epoch", -1)) != EXPECTED_COMPLETED_EPOCH - 1
        or int(resume_payload.get("global_step", -1)) != EXPECTED_COMPLETED_EPOCH * 623
        or resume_payload.get("frozen_manifest_sha256") != current_freeze_sha
        or resume_payload.get("source_revision") != git_revision(FUNIE_PYTORCH.parent)
        or resume_payload.get("source_epoch1_checkpoint_sha256") != configuration["source_run"]["checkpoint_sha256"]
        or resume_payload.get("batchnorm_statistics_frozen") is not True
    ):
        raise RuntimeError("Resume checkpoint is not the verified completed epoch-4 frozen-BatchNorm state")
    assert_recorded_bn_buffers(resume_payload, "generator", "batchnorm_epoch1_generator_state")
    assert_recorded_bn_buffers(resume_payload, "discriminator", "batchnorm_epoch1_discriminator_state")
    copied_source = run_dir / "source_checkpoint/epoch_001.pt"
    if not copied_source.is_file() or sha256(copied_source) != configuration["source_run"]["checkpoint_sha256"]:
        raise RuntimeError("The preserved source epoch-1 copy is missing or altered")
    if resume_payload.get("rng", {}).get("mps") is None:
        raise RuntimeError("Epoch-4 checkpoint lacks MPS RNG state; refusing an unverifiable continuation")

    source_run_dir = PROJECT / configuration["source_run"]["path"]
    source_checkpoints = source_run_dir / "checkpoints"
    source_metrics_path = source_run_dir / "metrics.jsonl"
    if not (source_run_dir.is_dir() and source_metrics_path.is_file()):
        raise RuntimeError("The preserved original run is unavailable for matched comparisons")
    for epoch in range(5, FINAL_EPOCH + 1):
        if not (source_checkpoints / f"epoch_{epoch:03d}.pt").is_file():
            raise RuntimeError(f"Original matching epoch-{epoch} checkpoint is unavailable")
    original_metrics = source_metric_rows(source_metrics_path)

    train_rows, validation_rows = read_frozen_rows("train"), read_frozen_rows("val")
    if len(train_rows) != 623 or len(validation_rows) != 134:
        raise RuntimeError("Unexpected frozen UIEB v2 train/validation pair counts")
    preview_names = list(original_metrics[1].get("comparison_filenames", []))
    available_names = {str(row["filename"]) for row in validation_rows}
    if len(preview_names) != 5 or not set(preview_names + COLOR_SHIFT_CASES).issubset(available_names):
        raise RuntimeError("Fixed preview or known color-shift case is absent from frozen validation")
    baseline_path = run_dir / "validation_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("pairs") != 134 or baseline.get("frozen_manifest_sha256") != current_freeze_sha:
        raise RuntimeError("Raw validation baseline does not match the frozen validation manifest")

    device = require_mps()
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.commons import VGG19_PercepLoss
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    discriminator = DiscriminatorFunieGAN().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    perceptual = VGG19_PercepLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    vgg_path = Path(torch.hub.get_dir()) / "checkpoints/vgg19-dcbb9e9d.pth"
    if not vgg_path.is_file() or sha256(vgg_path) != configuration.get("vgg19", {}).get("sha256"):
        raise RuntimeError("Pinned VGG19 perceptual weights do not match the existing ablation configuration")

    generator.load_state_dict(resume_payload["generator"], strict=True)
    discriminator.load_state_dict(resume_payload["discriminator"], strict=True)
    optimizer_g.load_state_dict(resume_payload["optimizer_g"])
    optimizer_d.load_state_dict(resume_payload["optimizer_d"])
    generator_bn_epoch1 = resume_payload["batchnorm_epoch1_generator_state"]
    discriminator_bn_epoch1 = resume_payload["batchnorm_epoch1_discriminator_state"]
    resume_apply_g = train_with_frozen_batchnorm(generator, generator_bn_epoch1, "generator resume")
    resume_apply_d = train_with_frozen_batchnorm(discriminator, discriminator_bn_epoch1, "discriminator resume")
    rng_restored = restore_rng(resume_payload["rng"])
    if not all(rng_restored.values()):
        raise RuntimeError(f"Incomplete RNG restoration: {rng_restored}")
    resume_verification = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resume_checkpoint": str(args.resume.resolve()),
        "checkpoint_sha256": resume_hash,
        "restored_epoch": int(resume_payload["epoch"]) + 1,
        "next_epoch": EXPECTED_COMPLETED_EPOCH + 1,
        "restored_global_step": int(resume_payload["global_step"]),
        "requested_total_epochs": FINAL_EPOCH,
        "generator_state_matches": module_state_matches(generator, resume_payload["generator"]),
        "discriminator_state_matches": module_state_matches(discriminator, resume_payload["discriminator"]),
        "optimizer_g": optimizer_resume_summary(optimizer_g, resume_payload["optimizer_g"]),
        "optimizer_d": optimizer_resume_summary(optimizer_d, resume_payload["optimizer_d"]),
        "rng_restored": rng_restored,
        "frozen_manifest_sha256": current_freeze_sha,
        "batchnorm_epoch1_state_verified": True,
        "batchnorm_reapplied_after_train": {"generator": resume_apply_g, "discriminator": resume_apply_d},
    }
    if not (
        resume_verification["generator_state_matches"]
        and resume_verification["discriminator_state_matches"]
        and resume_verification["optimizer_g"]["entry_count_matches"]
        and resume_verification["optimizer_g"]["group_count_matches"]
        and resume_verification["optimizer_g"]["state_values_match"]
        and resume_verification["optimizer_d"]["entry_count_matches"]
        and resume_verification["optimizer_d"]["group_count_matches"]
        and resume_verification["optimizer_d"]["state_values_match"]
    ):
        raise RuntimeError("Generator, discriminator, or optimizer state did not restore exactly")
    resume_events_path.write_text(json.dumps(resume_verification) + "\n", encoding="utf-8")
    continuation_configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "aqua_vision_funie_batchnorm_frozen_continuation_v1",
        "run_id": args.run_id,
        "resume": resume_verification,
        "bounded_epochs": {"completed_at_resume": EXPECTED_COMPLETED_EPOCH, "run": list(range(5, FINAL_EPOCH + 1)), "total": FINAL_EPOCH},
        "frozen_manifest_sha256": current_freeze_sha,
        "single_experimental_change": configuration["only_change"],
        "batchnorm_behavior": "After every generator.train()/discriminator.train() call, all BatchNorm2d modules are returned to eval mode; their affine parameters stay trainable and their epoch-1 buffers are asserted unchanged.",
        "raw_baseline": baseline,
        "fixed_previews": preview_names,
        "color_shift_cases": COLOR_SHIFT_CASES,
    }
    continuation_config_path.write_text(json.dumps(continuation_configuration, indent=2) + "\n", encoding="utf-8")

    train_dataset = FrozenUIEBPairs(train_rows, train=True, seed=args.seed)
    validation_dataset = FrozenUIEBPairs(validation_rows, train=False, seed=args.seed)
    adversarial, l1 = torch.nn.MSELoss(), torch.nn.L1Loss()
    global_step = int(resume_payload["global_step"])
    best_psnr = float(resume_payload["best_validation_psnr"])
    started = time.perf_counter()
    ablation_metrics = {int(row["epoch"]): row for row in recorded_metrics}
    continuation_comparisons: list[dict[str, Any]] = []

    for epoch_index in range(EXPECTED_COMPLETED_EPOCH, FINAL_EPOCH):
        epoch = epoch_index + 1
        epoch_started = time.perf_counter()
        generator_train_audit = train_with_frozen_batchnorm(generator, generator_bn_epoch1, f"generator epoch {epoch}")
        discriminator_train_audit = train_with_frozen_batchnorm(discriminator, discriminator_bn_epoch1, f"discriminator epoch {epoch}")
        train_dataset.set_epoch(epoch_index)
        data_generator = torch.Generator().manual_seed(args.seed + epoch_index)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, generator=data_generator)
        losses: list[dict[str, float]] = []
        first_gradient_audit: dict[str, Any] | None = None
        for batch_index, batch in enumerate(train_loader, start=1):
            raw, reference = batch["raw"].to(device), batch["reference"].to(device)
            if raw.device.type != "mps" or reference.device.type != "mps":
                raise RuntimeError("Training tensors did not reach MPS")
            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_for_d = generator(raw)
            prediction_real = discriminator(reference, raw)
            prediction_fake = discriminator(fake_for_d, raw)
            d_loss = 0.5 * (
                adversarial(prediction_real, torch.ones_like(prediction_real))
                + adversarial(prediction_fake, torch.zeros_like(prediction_fake))
            ) * 10.0
            require_finite(d_loss, "discriminator loss")
            d_loss.backward()
            discriminator_gradients = affine_gradient_summary(discriminator, "discriminator")
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            generated = generator(raw)
            prediction_generated = discriminator(generated, raw)
            gan_loss = adversarial(prediction_generated, torch.ones_like(prediction_generated))
            l1_loss = l1(generated, reference)
            perceptual_loss = perceptual(generated, reference)
            g_loss = gan_loss + 7.0 * l1_loss + 3.0 * perceptual_loss
            require_finite(g_loss, "generator loss")
            g_loss.backward()
            generator_gradients = affine_gradient_summary(generator, "generator")
            optimizer_g.step()
            if batch_index == 1:
                first_gradient_audit = {"generator": generator_gradients, "discriminator": discriminator_gradients}
                assert_batchnorm_unchanged(generator, generator_bn_epoch1, f"generator epoch {epoch} after first batch")
                assert_batchnorm_unchanged(discriminator, discriminator_bn_epoch1, f"discriminator epoch {epoch} after first batch")
            losses.append(
                {
                    "d_loss": require_finite(d_loss, "recorded discriminator loss"),
                    "g_loss": require_finite(g_loss, "recorded generator loss"),
                    "gan_loss": require_finite(gan_loss, "recorded GAN loss"),
                    "l1_loss": require_finite(l1_loss, "recorded L1 loss"),
                    "perceptual_loss": require_finite(perceptual_loss, "recorded perceptual loss"),
                }
            )
            global_step += 1
        if len(losses) != 623:
            raise RuntimeError(f"Epoch {epoch} did not complete the 623 frozen training pairs")
        assert_batchnorm_unchanged(generator, generator_bn_epoch1, f"generator epoch {epoch} after epoch")
        assert_batchnorm_unchanged(discriminator, discriminator_bn_epoch1, f"discriminator epoch {epoch} after epoch")

        generator.eval()
        validation_psnr: list[float] = []
        validation_ssim: list[float] = []
        with torch.inference_mode():
            for index in range(len(validation_dataset)):
                item = validation_dataset[index]
                raw, reference = item["raw"].unsqueeze(0).to(device), item["reference"].unsqueeze(0).to(device)
                generated = generator(raw)
                psnr, ssim = metrics_for_pair(generated, reference)
                validation_psnr.append(psnr)
                validation_ssim.append(ssim)
        mean_psnr, mean_ssim = float(np.mean(validation_psnr)), float(np.mean(validation_ssim))
        if not (math.isfinite(mean_psnr) and math.isfinite(mean_ssim)):
            raise RuntimeError(f"Epoch {epoch} produced non-finite validation metrics")
        preview_path = run_dir / "validation_comparisons" / f"epoch_{epoch:03d}.png"
        color_preview_path = run_dir / "color_shift_previews" / f"epoch_{epoch:03d}.png"
        save_fixed_preview(generator, validation_dataset, preview_names, device, preview_path)
        save_fixed_preview(generator, validation_dataset, COLOR_SHIFT_CASES, device, color_preview_path)
        previous_best = best_psnr
        best_psnr = max(best_psnr, mean_psnr)
        payload = checkpoint_payload(
            epoch_index,
            global_step,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            best_psnr,
            args.run_id,
            configuration["source_run"]["checkpoint_sha256"],
            generator_bn_epoch1,
            discriminator_bn_epoch1,
        )
        checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
        atomic_torch_save(payload, checkpoint)
        shutil.copy2(checkpoint, checkpoints / "latest.pt")
        if mean_psnr > previous_best:
            shutil.copy2(checkpoint, checkpoints / "best_validation_psnr.pt")
        first_raw = validation_dataset[0]["raw"].unsqueeze(0).to(device)
        checkpoint_reload = verify_ablation_checkpoint(
            checkpoint,
            epoch,
            first_raw,
            device,
            generator_bn_epoch1,
            discriminator_bn_epoch1,
        )
        summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_steps": len(losses),
            "loss_mean": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
            "validation": {"pairs": len(validation_psnr), "psnr_mean": mean_psnr, "ssim_mean": mean_ssim},
            "batchnorm": {
                "statistics_frozen": True,
                "generator_state_unchanged": True,
                "discriminator_state_unchanged": True,
                "reapplied_after_every_train_call": {"generator": generator_train_audit, "discriminator": discriminator_train_audit},
                "first_batch_affine_gradient_audit": first_gradient_audit,
            },
            "comparison_image": str(preview_path.relative_to(PROJECT)),
            "color_shift_comparison_image": str(color_preview_path.relative_to(PROJECT)),
            "checkpoint": str((checkpoints / "latest.pt").relative_to(PROJECT)),
            "checkpoint_reload": checkpoint_reload,
            "device": str(device),
            "mps_memory": mps_memory(),
            "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "epoch_elapsed_seconds": round(time.perf_counter() - epoch_started, 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "frozen_manifest_sha256": current_freeze_sha,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        ablation_metrics[epoch] = summary
        comparison = compare_epoch(
            epoch,
            source_checkpoints / f"epoch_{epoch:03d}.pt",
            checkpoint,
            validation_dataset,
            preview_names,
            COLOR_SHIFT_CASES,
            device,
            comparison_dir,
        )
        comparison["per_image_regressions"] = write_regression_file(comparison_dir, epoch)
        continuation_comparisons.append(comparison)
        print(json.dumps({"epoch": epoch, "training": summary, "comparison": comparison}, indent=2))

    if set(range(5, FINAL_EPOCH + 1)) != set(ablation_metrics) - {2, 3, 4}:
        raise RuntimeError("Continuation did not record exactly epochs 5--20")
    write_continuation_reports(run_dir, configuration, continuation_comparisons, original_metrics, ablation_metrics, baseline)
    print(
        json.dumps(
            {
                "status": "passed",
                "run_dir": str(run_dir.relative_to(PROJECT)),
                "completed_epochs": list(range(5, FINAL_EPOCH + 1)),
                "latest_checkpoint": str((checkpoints / "latest.pt").relative_to(PROJECT)),
                "best_psnr_checkpoint": str((checkpoints / "best_validation_psnr.pt").relative_to(PROJECT)),
                "report": str((comparison_dir / "CONTINUATION_EPOCHS_005_TO_020.md").relative_to(PROJECT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
