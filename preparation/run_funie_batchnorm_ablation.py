#!/usr/bin/env python3
"""MPS-only BatchNorm-statistics ablation for the frozen UIEB FUnIE-GAN run.

This runner copies an existing epoch-1 checkpoint into a separate run directory
and trains only epochs 2–4. It preserves the original data, seed, losses,
optimizer states, batch size, and paired preprocessing. The sole experimental
change is keeping generator/discriminator BatchNorm modules in evaluation mode
after every model.train() call, while their affine parameters remain trainable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only ablation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
PREPARATION = PROJECT / "preparation"
FUNIE_PYTORCH = PROJECT / "third_party/FUnIE-GAN/PyTorch"
RUNS_ROOT = PROJECT / "training_runs/funie"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"

sys.path.insert(0, str(PREPARATION))
from run_funie_staged import (  # noqa: E402
    FrozenUIEBPairs,
    atomic_torch_save,
    capture_rng,
    git_revision,
    metrics_for_pair,
    mps_memory,
    read_frozen_rows,
    require_finite,
    require_mps,
    restore_rng,
    sha256,
    tensor_to_image,
    validation_baseline,
)


def batchnorm_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.BatchNorm2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]


def freeze_batchnorm_statistics(model: torch.nn.Module) -> None:
    """Set only BatchNorm modules to eval; retain affine weights/biases for training."""
    modules = batchnorm_modules(model)
    if not modules:
        raise RuntimeError("Expected BatchNorm layers were not found")
    for _, module in modules:
        module.eval()
        if module.training or not module.affine:
            raise RuntimeError("Failed to place a BatchNorm module in evaluation mode")
        if module.weight is None or module.bias is None or not module.weight.requires_grad or not module.bias.requires_grad:
            raise RuntimeError("BatchNorm affine parameters are not trainable")


def snapshot_batchnorm(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "running_mean": module.running_mean.detach().cpu().clone(),
            "running_var": module.running_var.detach().cpu().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().cpu().clone(),
        }
        for name, module in batchnorm_modules(model)
    }


def assert_batchnorm_unchanged(model: torch.nn.Module, expected: dict[str, dict[str, torch.Tensor]], label: str) -> None:
    actual = snapshot_batchnorm(model)
    if actual.keys() != expected.keys():
        raise RuntimeError(f"{label} BatchNorm layer set changed")
    for name in actual:
        for field in ("running_mean", "running_var", "num_batches_tracked"):
            if not torch.equal(actual[name][field], expected[name][field]):
                raise RuntimeError(f"{label} BatchNorm {field} changed in {name}")


def affine_gradient_summary(model: torch.nn.Module, label: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, module in batchnorm_modules(model):
        if module.weight is None or module.bias is None or module.weight.grad is None or module.bias.grad is None:
            raise RuntimeError(f"{label} BatchNorm affine parameters did not receive gradients in {name}")
        weight_gradient, bias_gradient = module.weight.grad.detach(), module.bias.grad.detach()
        if not torch.isfinite(weight_gradient).all() or not torch.isfinite(bias_gradient).all():
            raise RuntimeError(f"{label} BatchNorm affine gradients are non-finite in {name}")
        weight_norm, bias_norm = float(weight_gradient.norm().cpu()), float(bias_gradient.norm().cpu())
        if weight_norm == 0.0 and bias_norm == 0.0:
            raise RuntimeError(f"{label} BatchNorm affine gradients are all zero in {name}")
        summary[name] = {
            "weight_grad_norm": weight_norm,
            "bias_grad_norm": bias_norm,
        }
    return summary


def checkpoint_payload(
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    best_psnr: float,
    run_id: str,
    source_checkpoint_sha256: str,
    generator_batchnorm: dict[str, dict[str, torch.Tensor]],
    discriminator_batchnorm: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "format": "aqua_vision_funie_batchnorm_frozen_v1",
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_psnr": best_psnr,
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "source_revision": git_revision(FUNIE_PYTORCH.parent),
        "source_epoch1_checkpoint_sha256": source_checkpoint_sha256,
        "batchnorm_statistics_frozen": True,
        "batchnorm_epoch1_generator_state": generator_batchnorm,
        "batchnorm_epoch1_discriminator_state": discriminator_batchnorm,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "rng": capture_rng(),
    }


def save_fixed_preview(
    generator: torch.nn.Module,
    dataset: FrozenUIEBPairs,
    names: list[str],
    device: torch.device,
    output: Path,
) -> None:
    rows_by_name = {str(dataset[index]["filename"]): index for index in range(len(dataset))}
    font = ImageFont.load_default()
    label_height, image_size = 18, 256
    canvas = Image.new("RGB", (image_size * len(names), label_height + image_size * 3), "white")
    draw = ImageDraw.Draw(canvas)
    generator.eval()
    with torch.inference_mode():
        for column, name in enumerate(names):
            item = dataset[rows_by_name[name]]
            raw = item["raw"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            generated = generator(raw)
            draw.text((column * image_size + 3, 3), name, fill="black", font=font)
            for row, image in enumerate((tensor_to_image(raw[0]), tensor_to_image(generated[0]), tensor_to_image(reference[0]))):
                canvas.paste(image, (column * image_size, label_height + row * image_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def load_generator(checkpoint: Path, expected_epoch: int, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) + 1 != expected_epoch:
        raise RuntimeError(f"{checkpoint} is not epoch {expected_epoch}")
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.funiegan import GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    generator.load_state_dict(payload["generator"], strict=True)
    generator.eval()
    return generator


def verify_ablation_checkpoint(
    checkpoint: Path,
    expected_epoch: int,
    raw: torch.Tensor,
    device: torch.device,
    generator_batchnorm_epoch1: dict[str, dict[str, torch.Tensor]],
    discriminator_batchnorm_epoch1: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) + 1 != expected_epoch or not payload.get("batchnorm_statistics_frozen"):
        raise RuntimeError("Reloaded checkpoint does not have the expected ablation provenance")
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    generator, discriminator = GeneratorFunieGAN().to(device), DiscriminatorFunieGAN().to(device)
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    generator.eval()
    discriminator.eval()
    freeze_batchnorm_statistics(generator)
    freeze_batchnorm_statistics(discriminator)
    assert_batchnorm_unchanged(generator, generator_batchnorm_epoch1, "reloaded generator")
    assert_batchnorm_unchanged(discriminator, discriminator_batchnorm_epoch1, "reloaded discriminator")
    with torch.inference_mode():
        output = generator(raw)
    if output.device.type != "mps" or not torch.isfinite(output).all():
        raise RuntimeError("Reloaded ablation checkpoint inference failed on MPS")
    return {"passed": True, "output_shape": list(output.shape), "checkpoint_sha256": sha256(checkpoint), "batchnorm_state_unchanged": True}


def make_comparison_sheet(title: str, rows: list[dict[str, Any]], images: dict[str, tuple[Image.Image, Image.Image, Image.Image, Image.Image]], output: Path) -> None:
    font = ImageFont.load_default()
    image_size, title_height, label_height = 256, 50, 32
    canvas = Image.new("RGB", (image_size * 4, title_height + len(rows) * (label_height + image_size)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), title, fill="black", font=font)
    headers = ("raw input", "original output", "BN-frozen output", "reference")
    for column, header in enumerate(headers):
        draw.rectangle((column * image_size, 22, (column + 1) * image_size, title_height), outline="black")
        draw.text((column * image_size + 5, 30), header, fill="black", font=font)
    y = title_height
    for row in rows:
        name = str(row["filename"])
        label = (
            f"{name} | original dPSNR={row['original_psnr_delta']:+.3f}, dSSIM={row['original_ssim_delta']:+.4f} | "
            f"BN-frozen dPSNR={row['ablation_psnr_delta']:+.3f}, dSSIM={row['ablation_ssim_delta']:+.4f}"
        )
        draw.text((5, y + 4), label, fill="black", font=font)
        for column, image in enumerate(images[name]):
            canvas.paste(image, (column * image_size, y + label_height))
        y += label_height + image_size
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def counts(values: list[float]) -> dict[str, int]:
    return {"improved": sum(value > 0.0 for value in values), "worsened": sum(value < 0.0 for value in values), "unchanged": sum(value == 0.0 for value in values)}


def compare_epoch(
    epoch: int,
    original_checkpoint: Path,
    ablation_checkpoint: Path,
    dataset: FrozenUIEBPairs,
    fixed_names: list[str],
    color_failure_names: list[str],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_generator(original_checkpoint, epoch, device)
    ablation = load_generator(ablation_checkpoint, epoch, device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            raw = item["raw"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            original_output, ablation_output = original(raw), ablation(raw)
            if not torch.isfinite(original_output).all() or not torch.isfinite(ablation_output).all():
                raise RuntimeError(f"Non-finite output in epoch {epoch} comparison")
            raw_psnr, raw_ssim = metrics_for_pair(raw, reference)
            original_psnr, original_ssim = metrics_for_pair(original_output, reference)
            ablation_psnr, ablation_ssim = metrics_for_pair(ablation_output, reference)
            records.append(
                {
                    "filename": str(item["filename"]),
                    "raw_psnr": raw_psnr,
                    "raw_ssim": raw_ssim,
                    "original_psnr": original_psnr,
                    "original_ssim": original_ssim,
                    "original_psnr_delta": original_psnr - raw_psnr,
                    "original_ssim_delta": original_ssim - raw_ssim,
                    "ablation_psnr": ablation_psnr,
                    "ablation_ssim": ablation_ssim,
                    "ablation_psnr_delta": ablation_psnr - raw_psnr,
                    "ablation_ssim_delta": ablation_ssim - raw_ssim,
                    "ablation_minus_original_psnr": ablation_psnr - original_psnr,
                    "ablation_minus_original_ssim": ablation_ssim - original_ssim,
                }
            )
    torch.mps.synchronize()
    with (output_dir / f"per_image_metrics_epoch_{epoch:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_name = {str(dataset[index]["filename"]): index for index in range(len(dataset))}
    requested = set(fixed_names + color_failure_names)
    images: dict[str, tuple[Image.Image, Image.Image, Image.Image, Image.Image]] = {}
    with torch.inference_mode():
        for name in requested:
            item = dataset[by_name[name]]
            raw = item["raw"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            images[name] = (tensor_to_image(raw[0]), tensor_to_image(original(raw)[0]), tensor_to_image(ablation(raw)[0]), tensor_to_image(reference[0]))
    by_record = {row["filename"]: row for row in records}
    sheet_dir = output_dir / "comparison_sheets"
    make_comparison_sheet(f"epoch {epoch} fixed previews", [by_record[name] for name in fixed_names], images, sheet_dir / f"epoch_{epoch:03d}_fixed_previews.png")
    make_comparison_sheet(f"epoch {epoch} prior color-shift failure cases", [by_record[name] for name in color_failure_names], images, sheet_dir / f"epoch_{epoch:03d}_color_shift_cases.png")
    psnr_differences = [float(row["ablation_minus_original_psnr"]) for row in records]
    ssim_differences = [float(row["ablation_minus_original_ssim"]) for row in records]
    return {
        "epoch": epoch,
        "pairs": len(records),
        "ablation_minus_original": {
            "mean_psnr": float(np.mean(psnr_differences)),
            "median_psnr": float(median(psnr_differences)),
            "psnr_counts": counts(psnr_differences),
            "mean_ssim": float(np.mean(ssim_differences)),
            "median_ssim": float(median(ssim_differences)),
            "ssim_counts": counts(ssim_differences),
        },
        "ablation_vs_raw": {
            "mean_psnr_delta": float(np.mean([row["ablation_psnr_delta"] for row in records])),
            "mean_ssim_delta": float(np.mean([row["ablation_ssim_delta"] for row in records])),
        },
    }


def write_comparison_report(output: Path, baseline: dict[str, Any], comparisons: list[dict[str, Any]], source_run: str) -> None:
    rows = "\n".join(
        f"| {item['epoch']} | {item['ablation_minus_original']['mean_psnr']:+.4f} | {item['ablation_minus_original']['median_psnr']:+.4f} | {item['ablation_minus_original']['psnr_counts']['improved']} / {item['ablation_minus_original']['psnr_counts']['worsened']} | {item['ablation_minus_original']['mean_ssim']:+.5f} | {item['ablation_minus_original']['median_ssim']:+.5f} | {item['ablation_minus_original']['ssim_counts']['improved']} / {item['ablation_minus_original']['ssim_counts']['worsened']} |"
        for item in comparisons
    )
    output.write_text(
        f"""# BatchNorm-statistics ablation versus original epochs 2–4

- Original run: `{source_run}`.
- The raw validation baseline is {baseline['psnr_mean']:.4f} dB PSNR and
  {baseline['ssim_mean']:.5f} SSIM across the same 134 frozen validation pairs.
- The only training change in this run is BatchNorm behavior: generator and
  discriminator BatchNorm modules use their epoch-1 running statistics in eval
  mode while their affine weight and bias parameters remain trainable.

| Epoch | Mean ablation−original PSNR | Median | Improved / worsened | Mean ablation−original SSIM | Median | Improved / worsened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Per-image metrics and raw/original/BN-frozen/reference sheets are saved beside
this report. The sheets include the fixed validation previews and previously
identified color-shift failure cases: `725_img_.png`, `809_img_.png`,
`832_img_.png`, `727_img_.png`, and `920_img_.png`.

This experiment freezes epoch-1 BatchNorm statistics. It can test whether that
specific batch-size-one mechanism contributes to instability, but cannot by
itself prove or disprove the underlying cause.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New, separate ablation run directory name")
    parser.add_argument("--source-checkpoint", type=Path, required=True, help="Preserved original epoch_001.pt")
    parser.add_argument("--epochs", type=int, default=4, help="Total epoch number; must be 4 for epochs 2–4")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--preview-count", type=int, default=5)
    args = parser.parse_args()
    if args.epochs != 4 or args.preview_count != 5:
        raise SystemExit("This controlled experiment requires --epochs 4 and --preview-count 5")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    if not args.source_checkpoint.is_file() or args.source_checkpoint.name != "epoch_001.pt":
        raise SystemExit("--source-checkpoint must be the preserved original epoch_001.pt")

    device = require_mps()
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing ablation run: {run_dir}")
    source_checkpoint = args.source_checkpoint.resolve()
    source_run_dir = source_checkpoint.parents[1]
    source_config_path = source_run_dir / "run_config.json"
    source_metrics_path = source_run_dir / "metrics.jsonl"
    if not (source_config_path.is_file() and source_metrics_path.is_file()):
        raise RuntimeError("Source checkpoint must belong to a recorded FUnIE-GAN run")
    source_config = json.loads(source_config_path.read_text())
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_hash = sha256(source_checkpoint)
    if int(source_payload.get("epoch", -1)) != 0 or int(source_payload.get("global_step", -1)) != 623:
        raise RuntimeError("Source checkpoint is not the expected completed epoch-1 state")
    if source_payload.get("frozen_manifest_sha256") != sha256(FREEZE_FILE):
        raise RuntimeError("Source checkpoint does not match the frozen manifest")
    if source_payload.get("source_revision") != git_revision(FUNIE_PYTORCH.parent):
        raise RuntimeError("Source checkpoint does not match the pinned FUnIE-GAN revision")
    if source_config.get("seed") != args.seed or source_config.get("pilot_arguments", {}).get("preview_count") != args.preview_count or source_config.get("pilot_arguments", {}).get("max_train_steps") is not None:
        raise RuntimeError("Source run seed, preview count, or train-step setting does not match this controlled ablation")

    train_rows, validation_rows = read_frozen_rows("train"), read_frozen_rows("val")
    if len(train_rows) != 623 or len(validation_rows) != 134:
        raise RuntimeError("Unexpected frozen UIEB train/validation pair counts")
    train_dataset = FrozenUIEBPairs(train_rows, train=True, seed=args.seed)
    validation_dataset = FrozenUIEBPairs(validation_rows, train=False, seed=args.seed)
    preview_names = list(json.loads(source_metrics_path.read_text().splitlines()[0])["comparison_filenames"])
    if len(preview_names) != 5:
        raise RuntimeError("Source fixed-preview list is not the expected five validation pairs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.commons import VGG19_PercepLoss
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    discriminator = DiscriminatorFunieGAN().to(device)
    generator.load_state_dict(source_payload["generator"], strict=True)
    discriminator.load_state_dict(source_payload["discriminator"], strict=True)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    optimizer_g.load_state_dict(source_payload["optimizer_g"])
    optimizer_d.load_state_dict(source_payload["optimizer_d"])
    perceptual = VGG19_PercepLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    vgg_path = Path(torch.hub.get_dir()) / "checkpoints/vgg19-dcbb9e9d.pth"
    expected_vgg_hash = source_config.get("vgg19", {}).get("sha256")
    if expected_vgg_hash is None or not vgg_path.is_file() or sha256(vgg_path) != expected_vgg_hash:
        raise RuntimeError("Pinned VGG19 perceptual weights do not match the original run")
    # Match the original resume order: model, optimizer, and VGG construction
    # complete before restoring the epoch-1 RNG snapshot for the epoch-2 loop.
    rng_restored = restore_rng(source_payload.get("rng", {}))
    adversarial, l1 = torch.nn.MSELoss(), torch.nn.L1Loss()

    generator.train()
    discriminator.train()
    freeze_batchnorm_statistics(generator)
    freeze_batchnorm_statistics(discriminator)
    generator_batchnorm_epoch1 = snapshot_batchnorm(generator)
    discriminator_batchnorm_epoch1 = snapshot_batchnorm(discriminator)
    run_dir.mkdir(parents=True)
    copied_source = run_dir / "source_checkpoint/epoch_001.pt"
    copied_source.parent.mkdir(parents=True)
    shutil.copy2(source_checkpoint, copied_source)
    if sha256(copied_source) != source_hash:
        raise RuntimeError("Copied source checkpoint hash does not match the preserved source")
    baseline = validation_baseline(validation_dataset, device)
    (run_dir / "validation_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    configuration = {
        "format": "aqua_vision_funie_batchnorm_frozen_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "device_requirement": "MPS only; PYTORCH_ENABLE_MPS_FALLBACK=0",
        "seed": args.seed,
        "source_run": {"path": str(source_run_dir.relative_to(PROJECT)), "run_id": source_payload.get("run_id"), "epoch": 1, "checkpoint_sha256": source_hash, "copied_checkpoint": str(copied_source.relative_to(PROJECT))},
        "frozen_manifest": {"path": str(FREEZE_FILE.relative_to(PROJECT)), "sha256": sha256(FREEZE_FILE)},
        "data": source_config["data"],
        "loss_recipe": source_config["loss_recipe"],
        "vgg19": source_config["vgg19"],
        "optimizer": {"name": "Adam", "lr": 0.0003, "betas": [0.5, 0.99], "source_optimizer_states_restored": True},
        "rng_restored": rng_restored,
        "batch_size": 1,
        "epochs": {"source_completed": 1, "run": [2, 3, 4]},
        "only_change": "After every generator.train()/discriminator.train(), all BatchNorm2d modules are reset to eval mode; affine weight/bias remain requires_grad=True and are optimized.",
        "caveat": "This freezes epoch-1 BatchNorm statistics and does not conclusively prove or disprove the cause of instability.",
        "raw_validation_baseline": baseline,
    }
    (run_dir / "run_config.json").write_text(json.dumps(configuration, indent=2) + "\n")

    checkpoints = run_dir / "checkpoints"
    comparisons = run_dir / "validation_comparisons"
    metrics_path = run_dir / "metrics.jsonl"
    global_step = int(source_payload["global_step"])
    best_psnr = -math.inf
    run_started = time.perf_counter()
    for epoch in range(1, args.epochs):
        epoch_started = time.perf_counter()
        generator.train()
        discriminator.train()
        freeze_batchnorm_statistics(generator)
        freeze_batchnorm_statistics(discriminator)
        assert_batchnorm_unchanged(generator, generator_batchnorm_epoch1, "generator before epoch")
        assert_batchnorm_unchanged(discriminator, discriminator_batchnorm_epoch1, "discriminator before epoch")
        train_dataset.set_epoch(epoch)
        data_generator = torch.Generator().manual_seed(args.seed + epoch)
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
            prediction_real, prediction_fake = discriminator(reference, raw), discriminator(fake_for_d, raw)
            d_loss = 0.5 * (adversarial(prediction_real, torch.ones_like(prediction_real)) + adversarial(prediction_fake, torch.zeros_like(prediction_fake))) * 10.0
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
                assert_batchnorm_unchanged(generator, generator_batchnorm_epoch1, "generator after first batch")
                assert_batchnorm_unchanged(discriminator, discriminator_batchnorm_epoch1, "discriminator after first batch")
            losses.append({
                "d_loss": require_finite(d_loss, "recorded discriminator loss"),
                "g_loss": require_finite(g_loss, "recorded generator loss"),
                "gan_loss": require_finite(gan_loss, "recorded GAN loss"),
                "l1_loss": require_finite(l1_loss, "recorded L1 loss"),
                "perceptual_loss": require_finite(perceptual_loss, "recorded perceptual loss"),
            })
            global_step += 1
        assert_batchnorm_unchanged(generator, generator_batchnorm_epoch1, "generator after epoch")
        assert_batchnorm_unchanged(discriminator, discriminator_batchnorm_epoch1, "discriminator after epoch")

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
            raise RuntimeError("Non-finite validation metric")
        save_fixed_preview(generator, validation_dataset, preview_names, device, comparisons / f"epoch_{epoch + 1:03d}.png")
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
        payload = checkpoint_payload(epoch, global_step, generator, discriminator, optimizer_g, optimizer_d, best_psnr, args.run_id, source_hash, generator_batchnorm_epoch1, discriminator_batchnorm_epoch1)
        checkpoint = checkpoints / f"epoch_{epoch + 1:03d}.pt"
        atomic_torch_save(payload, checkpoint)
        shutil.copy2(checkpoint, checkpoints / "latest.pt")
        if mean_psnr >= best_psnr:
            shutil.copy2(checkpoint, checkpoints / "best_validation_psnr.pt")
        first_raw = validation_dataset[0]["raw"].unsqueeze(0).to(device)
        checkpoint_reload = verify_ablation_checkpoint(checkpoint, epoch + 1, first_raw, device, generator_batchnorm_epoch1, discriminator_batchnorm_epoch1)
        summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_steps": len(losses),
            "loss_mean": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
            "validation": {"pairs": len(validation_psnr), "psnr_mean": mean_psnr, "ssim_mean": mean_ssim},
            "batchnorm": {
                "statistics_frozen": True,
                "generator_state_unchanged": True,
                "discriminator_state_unchanged": True,
                "first_batch_affine_gradient_audit": first_gradient_audit,
            },
            "comparison_image": str((comparisons / f"epoch_{epoch + 1:03d}.png").relative_to(PROJECT)),
            "checkpoint": str((checkpoints / "latest.pt").relative_to(PROJECT)),
            "checkpoint_reload": checkpoint_reload,
            "device": str(device),
            "mps_memory": mps_memory(),
            "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "epoch_elapsed_seconds": round(time.perf_counter() - epoch_started, 3),
            "elapsed_seconds": round(time.perf_counter() - run_started, 3),
            "frozen_manifest_sha256": sha256(FREEZE_FILE),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary, indent=2))

    comparison_dir = run_dir / "comparison_against_original"
    original_checkpoints = source_checkpoint.parent
    color_failure_names = ["725_img_.png", "809_img_.png", "832_img_.png", "727_img_.png", "920_img_.png"]
    comparisons = [
        compare_epoch(epoch, original_checkpoints / f"epoch_{epoch:03d}.pt", checkpoints / f"epoch_{epoch:03d}.pt", validation_dataset, preview_names, color_failure_names, device, comparison_dir)
        for epoch in (2, 3, 4)
    ]
    comparison_summary = {
        "source_run": str(source_run_dir.relative_to(PROJECT)),
        "source_epoch1_checkpoint_sha256": source_hash,
        "baseline": baseline,
        "comparisons": comparisons,
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "note": "Only BatchNorm running-stat behavior changed; epoch-1 statistics were frozen and this result does not conclusively prove or disprove the underlying cause.",
    }
    (comparison_dir / "summary.json").write_text(json.dumps(comparison_summary, indent=2) + "\n")
    write_comparison_report(comparison_dir / "FINDINGS.md", baseline, comparisons, str(source_run_dir.relative_to(PROJECT)))
    print(json.dumps({"status": "passed", "run_dir": str(run_dir.relative_to(PROJECT)), "comparison_dir": str(comparison_dir.relative_to(PROJECT)), "baseline": baseline, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
