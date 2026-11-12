#!/usr/bin/env python3
"""MPS-only, read-only FUnIE-GAN validation review.

This script never trains, updates checkpoints, or reads UIEB test/challenging
images. It compares the saved best-PSNR and latest generators on exactly the
frozen validation split, then writes review-only artifacts below that run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only review.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
PREPARATION = PROJECT / "preparation"
FUNIE_PYTORCH = PROJECT / "third_party/FUnIE-GAN/PyTorch"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
PAIR_MANIFEST = PROJECT / "datasets/processed/uieb_enhancement_v2_leakage_resolved/pair_split_manifest.csv"

sys.path.insert(0, str(PREPARATION))
from run_funie_staged import FrozenUIEBPairs, metrics_for_pair, mps_memory, read_frozen_rows, require_mps, tensor_to_image  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def finite_output(output: torch.Tensor, label: str) -> None:
    if output.device.type != "mps" or not torch.isfinite(output).all():
        raise RuntimeError(f"{label} did not produce a finite MPS output")


def checkpoint_models(checkpoint: Path, expected_epoch: int, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) + 1 != expected_epoch:
        raise RuntimeError(f"{checkpoint.name} is not the expected epoch {expected_epoch} checkpoint")
    if payload.get("frozen_manifest_sha256") != sha256(FREEZE_FILE):
        raise RuntimeError(f"{checkpoint.name} does not match the frozen manifest")
    if payload.get("source_revision") != git_revision(FUNIE_PYTORCH.parent):
        raise RuntimeError(f"{checkpoint.name} does not match the pinned FUnIE-GAN source revision")
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    discriminator = DiscriminatorFunieGAN().to(device)
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    generator.eval()
    discriminator.eval()
    if generator.training:
        raise RuntimeError(f"{checkpoint.name} generator did not enter evaluation mode")
    return generator, discriminator, payload


def batchnorm_summary(model: torch.nn.Module) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.BatchNorm2d):
            continue
        running_mean = module.running_mean.detach().float().cpu()
        running_var = module.running_var.detach().float().cpu()
        summaries.append(
            {
                "name": name,
                "num_features": module.num_features,
                "momentum": module.momentum,
                "track_running_stats": module.track_running_stats,
                "num_batches_tracked": int(module.num_batches_tracked.detach().cpu()),
                "running_mean": {"mean": float(running_mean.mean()), "std": float(running_mean.std()), "min": float(running_mean.min()), "max": float(running_mean.max())},
                "running_var": {"mean": float(running_var.mean()), "std": float(running_var.std()), "min": float(running_var.min()), "max": float(running_var.max())},
            }
        )
    return summaries


def delta_counts(values: list[float]) -> dict[str, int]:
    return {"improved": sum(value > 0.0 for value in values), "worsened": sum(value < 0.0 for value in values), "unchanged": sum(value == 0.0 for value in values)}


def metric_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    psnr_deltas = [float(row[f"{prefix}_psnr_delta"]) for row in rows]
    ssim_deltas = [float(row[f"{prefix}_ssim_delta"]) for row in rows]
    return {
        "mean_psnr_delta": float(np.mean(psnr_deltas)),
        "median_psnr_delta": float(median(psnr_deltas)),
        "psnr_counts": delta_counts(psnr_deltas),
        "mean_ssim_delta": float(np.mean(ssim_deltas)),
        "median_ssim_delta": float(median(ssim_deltas)),
        "ssim_counts": delta_counts(ssim_deltas),
    }


def sheet_image(title: str, selected: list[dict[str, Any]], images: dict[str, tuple[Image.Image, Image.Image, Image.Image, Image.Image]], output: Path) -> None:
    font = ImageFont.load_default()
    image_size, label_height, title_height = 256, 32, 50
    width = image_size * 4
    height = title_height + len(selected) * (label_height + image_size)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), title, fill="black", font=font)
    y = title_height
    for row in selected:
        name = str(row["filename"])
        label = (
            f"{name} | e3 dPSNR={row['epoch3_psnr_delta']:+.3f}, dSSIM={row['epoch3_ssim_delta']:+.4f} | "
            f"e20 dPSNR={row['epoch20_psnr_delta']:+.3f}, dSSIM={row['epoch20_ssim_delta']:+.4f}"
        )
        draw.text((5, y + 4), label, fill="black", font=font)
        raw, epoch3, epoch20, reference = images[name]
        for index, image in enumerate((raw, epoch3, epoch20, reference)):
            canvas.paste(image, (index * image_size, y + label_height))
        y += label_height + image_size
    headers = ("raw input", "epoch 3 output", "epoch 20 output", "reference")
    for index, header in enumerate(headers):
        draw.rectangle((index * image_size, 22, (index + 1) * image_size, title_height), outline="black")
        draw.text((index * image_size + 5, 30), header, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def read_training_rows(run_dir: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    return {int(row["epoch"]): row for row in rows}


def write_report(
    output: Path,
    run_dir: Path,
    checkpoint_data: dict[str, Any],
    audit: dict[str, Any],
    summary: dict[str, Any],
    epoch_rows: dict[int, dict[str, Any]],
) -> None:
    epoch12, epoch14 = epoch_rows[12], epoch_rows[14]
    preview12 = run_dir / "validation_comparisons/epoch_012.png"
    preview14 = run_dir / "validation_comparisons/epoch_014.png"
    checkpoint12 = run_dir / "checkpoints/epoch_012.pt"
    checkpoint14 = run_dir / "checkpoints/epoch_014.pt"
    generator_counts = (audit["epoch3_generator_batchnorm"][0]["num_batches_tracked"], audit["epoch20_generator_batchnorm"][0]["num_batches_tracked"])
    discriminator_counts = (audit["epoch3_discriminator_batchnorm"][0]["num_batches_tracked"], audit["epoch20_discriminator_batchnorm"][0]["num_batches_tracked"])
    text = f"""# FUnIE-GAN 20-epoch review

## Scope

- Run: `{run_dir.name}`.
- This review reads only the frozen UIEB validation split (134 pairs). It does
  not read test pairs or challenging images, train a model, alter checkpoints,
  or interact with the website or RF-DETR.
- Review device: `{summary['device']}` with CPU fallback disabled.

## Checkpoint preservation and verified provenance

- `best_validation_psnr.pt` exists and is byte-identical to epoch 3:
  `{checkpoint_data['best_hash']}`.
- `latest.pt` exists and is byte-identical to epoch 20:
  `{checkpoint_data['latest_hash']}`.
- Both checkpoints carry frozen-manifest SHA-256
  `{checkpoint_data['freeze_sha256']}` and pinned FUnIE-GAN revision
  `{checkpoint_data['source_revision']}`.

## Same-split comparison

The metric definition is exactly the staged runner's validation definition:
both images are bicubically resized to 256x256, represented in `[-1,1]`, then
mapped and clamped to `[0,1]`; PSNR uses RGB MSE and SSIM uses its 11x11,
sigma-1.5, zero-padded Gaussian window. These are internally comparable
metrics, not a claim of equivalence to a separate benchmark implementation.

| Checkpoint | Median dPSNR (dB) | PSNR improved / worsened | Median dSSIM | SSIM improved / worsened |
| --- | ---: | ---: | ---: | ---: |
| Epoch 3 | {summary['epoch3']['median_psnr_delta']:+.4f} | {summary['epoch3']['psnr_counts']['improved']} / {summary['epoch3']['psnr_counts']['worsened']} | {summary['epoch3']['median_ssim_delta']:+.5f} | {summary['epoch3']['ssim_counts']['improved']} / {summary['epoch3']['ssim_counts']['worsened']} |
| Epoch 20 | {summary['epoch20']['median_psnr_delta']:+.4f} | {summary['epoch20']['psnr_counts']['improved']} / {summary['epoch20']['psnr_counts']['worsened']} | {summary['epoch20']['median_ssim_delta']:+.5f} | {summary['epoch20']['ssim_counts']['improved']} / {summary['epoch20']['ssim_counts']['worsened']} |

Per-image values and changes are in `per_image_metrics.csv`. Labeled sheets
are in `comparison_sheets/`; each has raw input, epoch-3 output, epoch-20
output, and reference in that order.

## Audit: verified behavior

- **Train/validation modes:** the trainer calls `generator.train()` and
  `discriminator.train()` before each training epoch, then `generator.eval()`
  before validation and preview generation. Validation invokes only the
  generator under inference mode. This mode switch is present and correct.
- **BatchNorm exposure at batch size 1:** the generator has 7 BatchNorm2d
  layers and the discriminator has 3, all with momentum 0.8. During each
  training batch, generator BatchNorm statistics update for the no-grad fake
  pass used in the discriminator update and again for the generator update.
  Discriminator BatchNorm statistics update for real, fake, and generator-loss
  forwards. `no_grad` prevents gradients but does not prevent BatchNorm
  running-stat updates in train mode. The saved counter values rise from
  {generator_counts[0]} to {generator_counts[1]} for a generator BatchNorm
  layer and from {discriminator_counts[0]} to {discriminator_counts[1]} for a
  discriminator layer. Full summaries are in
  `batchnorm_running_statistics.json`.
- **Pairs and normalization:** raw/reference images are independently RGB
  converted and bicubically resized to the same 256x256 dimensions. The train
  flip is aligned across the pair; validation uses `train=False`, so it does
  not flip. Tensors use `pixel / 127.5 - 1.0`.
- **VGG feature loss:** the frozen VGG19 is set to evaluation mode and receives
  those `[-1,1]` tensors directly. That faithfully matches the pinned official
  source. It does not add torchvision ImageNet mean/std normalization.
- **Metrics:** raw-input baseline and generated-output values use the same
  preprocessing and metric function. All evaluated outputs were finite.

## Hypotheses, not confirmed bugs

- Batch size 1 plus BatchNorm momentum 0.8 creates highly volatile running
  statistics; it is a credible contributor to the abrupt evaluation swings,
  but this review does not prove causation.
- The VGG19 input convention is faithful to the official source, but differs
  from the conventional ImageNet normalization expected by torchvision's
  pretrained VGG. Its influence on this run is unmeasured.
- GAN dynamics may also be unstable: the discriminator loss falls substantially
  while generator loss flattens, but loss values alone cannot identify a fault.

## Epochs 12 and 14 sharp drops

- Both preview and checkpoint artifacts exist: epoch 12 preview={preview12.is_file()}, checkpoint={checkpoint12.is_file()}; epoch 14 preview={preview14.is_file()}, checkpoint={checkpoint14.is_file()}.
- Epoch 12 recorded PSNR/SSIM {epoch12['validation']['psnr_mean']:.4f} / {epoch12['validation']['ssim_mean']:.4f}; its preview shows strong dark/green saturation relative to the reference examples.
- Epoch 14 recorded PSNR/SSIM {epoch14['validation']['psnr_mean']:.4f} / {epoch14['validation']['ssim_mean']:.4f}; its preview shows widespread clipped bright cyan/white output.
- These artifacts confirm transient visual failure at those epochs. They do
  not identify the root cause; neither memory pressure nor a missing artifact
  is supported by the logged evidence.

## One justified next experiment — not launched

From the preserved epoch-1 checkpoint, run a short, fresh epochs-2-to-4
ablation that keeps the same frozen train/validation data, seed, losses, batch
size, and MPS-only requirement, but forces all generator and discriminator
BatchNorm modules to evaluation mode during training while leaving their affine
parameters trainable. Compare its validation distribution directly with the
original epochs 2–4. This isolates the highest-risk batch-1 mechanism without
touching held-out images. Do not run it until explicitly approved.
"""
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if not (FREEZE_FILE.is_file() and PAIR_MANIFEST.is_file()):
        raise SystemExit("Frozen UIEB validation manifest is required")

    run_dir = PROJECT / "training_runs/funie" / args.run_id
    checkpoints = run_dir / "checkpoints"
    best_checkpoint = checkpoints / "best_validation_psnr.pt"
    latest_checkpoint = checkpoints / "latest.pt"
    epoch3_checkpoint = checkpoints / "epoch_003.pt"
    epoch20_checkpoint = checkpoints / "epoch_020.pt"
    output_dir = run_dir / "review_epoch_003_vs_020"
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing review: {output_dir}")
    if not (best_checkpoint.is_file() and latest_checkpoint.is_file() and epoch3_checkpoint.is_file() and epoch20_checkpoint.is_file() and (run_dir / "metrics.jsonl").is_file()):
        raise SystemExit("Expected saved best/latest, epoch 3/20 checkpoints, and metrics.jsonl")
    if sha256(best_checkpoint) != sha256(epoch3_checkpoint):
        raise RuntimeError("best_validation_psnr.pt is not byte-identical to epoch_003.pt")
    if sha256(latest_checkpoint) != sha256(epoch20_checkpoint):
        raise RuntimeError("latest.pt is not byte-identical to epoch_020.pt")

    device = require_mps()
    validation_rows = read_frozen_rows("val")
    if len(validation_rows) != 134:
        raise RuntimeError("Unexpected frozen UIEB validation-pair count")
    dataset = FrozenUIEBPairs(validation_rows, train=False, seed=0)
    epoch_rows = read_training_rows(run_dir)
    if not all(epoch in epoch_rows for epoch in (3, 12, 14, 20)):
        raise RuntimeError("Expected epoch 3, 12, 14, and 20 training records")

    epoch3, discriminator3, payload3 = checkpoint_models(best_checkpoint, 3, device)
    epoch20, discriminator20, payload20 = checkpoint_models(latest_checkpoint, 20, device)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            raw = item["raw"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            output3 = epoch3(raw)
            output20 = epoch20(raw)
            finite_output(output3, "epoch 3")
            finite_output(output20, "epoch 20")
            raw_psnr, raw_ssim = metrics_for_pair(raw, reference)
            epoch3_psnr, epoch3_ssim = metrics_for_pair(output3, reference)
            epoch20_psnr, epoch20_ssim = metrics_for_pair(output20, reference)
            records.append(
                {
                    "filename": str(item["filename"]),
                    "raw_psnr": raw_psnr,
                    "raw_ssim": raw_ssim,
                    "epoch3_psnr": epoch3_psnr,
                    "epoch3_ssim": epoch3_ssim,
                    "epoch3_psnr_delta": epoch3_psnr - raw_psnr,
                    "epoch3_ssim_delta": epoch3_ssim - raw_ssim,
                    "epoch20_psnr": epoch20_psnr,
                    "epoch20_ssim": epoch20_ssim,
                    "epoch20_psnr_delta": epoch20_psnr - raw_psnr,
                    "epoch20_ssim_delta": epoch20_ssim - raw_ssim,
                }
            )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    fieldnames = list(records[0])
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    fixed_names = list(epoch_rows[1]["comparison_filenames"])
    selected: dict[str, list[dict[str, Any]]] = {
        "fixed_previews": [next(row for row in records if row["filename"] == name) for name in fixed_names],
        "epoch3_largest_psnr_improvements": sorted(records, key=lambda row: row["epoch3_psnr_delta"], reverse=True)[: args.top_k],
        "epoch3_largest_psnr_regressions": sorted(records, key=lambda row: row["epoch3_psnr_delta"])[: args.top_k],
        "epoch20_largest_psnr_improvements": sorted(records, key=lambda row: row["epoch20_psnr_delta"], reverse=True)[: args.top_k],
        "epoch20_largest_psnr_regressions": sorted(records, key=lambda row: row["epoch20_psnr_delta"])[: args.top_k],
    }
    needed = {row["filename"] for rows in selected.values() for row in rows}
    images: dict[str, tuple[Image.Image, Image.Image, Image.Image, Image.Image]] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            name = str(item["filename"])
            if name not in needed:
                continue
            raw = item["raw"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            images[name] = (tensor_to_image(raw[0]), tensor_to_image(epoch3(raw)[0]), tensor_to_image(epoch20(raw)[0]), tensor_to_image(reference[0]))
    torch.mps.synchronize()
    sheet_dir = output_dir / "comparison_sheets"
    for name, rows in selected.items():
        sheet_image(name.replace("_", " "), rows, images, sheet_dir / f"{name}.png")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "pairs": len(records),
        "freeze_sha256": sha256(FREEZE_FILE),
        "epoch3": metric_summary(records, "epoch3"),
        "epoch20": metric_summary(records, "epoch20"),
        "elapsed_seconds": round(elapsed, 3),
        "mps_memory": mps_memory(),
        "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    checkpoint_data = {
        "best_hash": sha256(best_checkpoint),
        "latest_hash": sha256(latest_checkpoint),
        "freeze_sha256": sha256(FREEZE_FILE),
        "source_revision": payload3["source_revision"],
        "best_payload_epoch": int(payload3["epoch"]) + 1,
        "latest_payload_epoch": int(payload20["epoch"]) + 1,
    }
    audit = {
        "epoch3_generator_batchnorm": batchnorm_summary(epoch3),
        "epoch20_generator_batchnorm": batchnorm_summary(epoch20),
        "epoch3_discriminator_batchnorm": batchnorm_summary(discriminator3),
        "epoch20_discriminator_batchnorm": batchnorm_summary(discriminator20),
        "generator_batchnorm_layers": 7,
        "discriminator_batchnorm_layers": 3,
        "batchnorm_momentum": 0.8,
    }
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary, "checkpoints": checkpoint_data}, indent=2) + "\n")
    (output_dir / "batchnorm_running_statistics.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_report(output_dir / "FINDINGS.md", run_dir, checkpoint_data, audit, summary, epoch_rows)
    print(json.dumps({"status": "passed", "review_dir": str(output_dir.relative_to(PROJECT)), "summary": summary, "checkpoints": checkpoint_data}, indent=2))


if __name__ == "__main__":
    main()
