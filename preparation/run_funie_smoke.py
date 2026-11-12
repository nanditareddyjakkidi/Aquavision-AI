#!/usr/bin/env python3
"""Bounded FUnIE-GAN architecture compatibility test for frozen UIEB train pairs.

This is deliberately not the legacy repository's full training entry point: that
entry point only knows EUVP/UFO-120 layouts and selects CUDA/CPU itself.  The
test uses its GeneratorFunieGAN and DiscriminatorFunieGAN classes directly,
requires an explicit CPU opt-in when MPS is unavailable, and writes only under
smoke_tests/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.json"
PAIR_MANIFEST = PROJECT / "datasets/processed/uieb_enhancement_v2_leakage_resolved/pair_split_manifest.csv"
FUNIE_SOURCE = PROJECT / "third_party/FUnIE-GAN/PyTorch"


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def finite(value: torch.Tensor, name: str) -> float:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite {name}")
    return float(value.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--allow-cpu-compatibility", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        raise SystemExit("--steps must be between 1 and 10")
    if args.image_size < 32 or args.image_size % 32:
        raise SystemExit("--image-size must be a multiple of 32 and at least 32")
    if not FROZEN_MANIFEST.is_file() or not PAIR_MANIFEST.is_file():
        raise SystemExit("Frozen v2 manifests are required before a smoke test can run")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        device = torch.device("mps")
        device_note = "MPS selected"
    elif args.allow_cpu_compatibility:
        device = torch.device("cpu")
        device_note = "CPU explicitly selected because MPS is unavailable at runtime"
    else:
        raise SystemExit("MPS is unavailable. Re-run with --allow-cpu-compatibility to perform a labelled CPU test.")

    rows: list[dict[str, str]] = []
    with PAIR_MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["derived_split"] == "train":
                rows.append(row)
    if len(rows) < args.steps:
        raise SystemExit("Not enough frozen UIEB train pairs for requested smoke-test steps")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("funie_smoke_%Y%m%dT%H%M%SZ")
    run_dir = PROJECT / "smoke_tests" / "funie" / run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing smoke-test output: {run_dir}")
    run_dir.mkdir(parents=True)

    sys.path.insert(0, str(FUNIE_SOURCE))
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    torch.manual_seed(20260930)
    generator = GeneratorFunieGAN().to(device).train()
    discriminator = DiscriminatorFunieGAN().to(device).train()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    parameter_before = next(generator.parameters()).detach().clone()
    start = time.perf_counter()
    losses: list[dict[str, float | int | str]] = []

    for step, row in enumerate(rows[: args.steps], start=1):
        raw = image_tensor(PROJECT / row["raw_path"], args.image_size, device)
        reference = image_tensor(PROJECT / row["reference_path"], args.image_size, device)

        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_for_d = generator(raw)
        prediction_real = discriminator(reference, raw)
        prediction_fake = discriminator(fake_for_d, raw)
        d_loss = (mse(prediction_real, torch.ones_like(prediction_real)) + mse(prediction_fake, torch.zeros_like(prediction_fake))) / 2
        finite(d_loss, "discriminator loss")
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        generated = generator(raw)
        prediction_generated = discriminator(generated, raw)
        adversarial_loss = mse(prediction_generated, torch.ones_like(prediction_generated))
        reconstruction_loss = l1(generated, reference)
        g_loss = adversarial_loss + 7.0 * reconstruction_loss
        finite(g_loss, "generator loss")
        g_loss.backward()
        optimizer_g.step()
        losses.append({
            "step": step,
            "filename": row["filename"],
            "d_loss": finite(d_loss, "discriminator loss after update"),
            "g_loss": finite(g_loss, "generator loss after update"),
            "adversarial_loss": finite(adversarial_loss, "adversarial loss"),
            "l1_loss": finite(reconstruction_loss, "L1 loss"),
        })

    optimizer_changed_generator = not torch.equal(parameter_before, next(generator.parameters()).detach())
    if not optimizer_changed_generator:
        raise RuntimeError("generator parameter did not change after optimizer updates")

    checkpoint = run_dir / "funie_smoke_checkpoint.pt"
    torch.save({
        "format": "aqua_vision_funie_smoke_v1",
        "source_revision": git_revision(FUNIE_SOURCE.parent),
        "model_architecture": "FUnIE-GAN PyTorch GeneratorFunieGAN + DiscriminatorFunieGAN",
        "pretrained_checkpoint": None,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "steps": args.steps,
    }, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    restored_generator = GeneratorFunieGAN().to(device)
    restored_discriminator = DiscriminatorFunieGAN().to(device)
    restored_generator.load_state_dict(payload["generator"])
    restored_discriminator.load_state_dict(payload["discriminator"])
    restored_generator.eval()
    with torch.inference_mode():
        output = restored_generator(image_tensor(PROJECT / rows[0]["raw_path"], args.image_size, device))
    if output.shape != (1, 3, args.image_size, args.image_size) or not torch.isfinite(output).all():
        raise RuntimeError(f"reloaded generator inference failed, got shape {tuple(output.shape)}")

    report = {
        "status": "passed",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "UIEB frozen derived train pairs only; challenging and held-out validation/test pairs excluded",
        "device": str(device),
        "device_note": device_note,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "steps": args.steps,
        "batch_size": 1,
        "image_size": args.image_size,
        "pairs_used": [{"filename": row["filename"], "raw_path": row["raw_path"], "reference_path": row["reference_path"]} for row in rows[: args.steps]],
        "losses": losses,
        "forward_pass": True,
        "finite_loss": True,
        "backward_pass": True,
        "optimizer_update": optimizer_changed_generator,
        "checkpoint": str(checkpoint.relative_to(PROJECT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": True,
        "inference": {"shape": list(output.shape), "finite": True},
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "frozen_manifest_sha256": sha256(PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"),
        "method_note": "Core GAN adversarial + L1 losses only. The official legacy entry point's VGG perceptual path and its CUDA/CPU selector were intentionally not used for this bounded compatibility test.",
    }
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
