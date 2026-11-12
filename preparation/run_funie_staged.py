#!/usr/bin/env python3
"""MPS-only staged FUnIE-GAN trainer using frozen AquaVision UIEB manifests.

This runner deliberately uses the complete loss structure in the pinned official
FUnIE-GAN PyTorch source.  It is separate from smoke_tests/ and never reads the
derived UIEB test split or the challenging set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# MPS CPU fallback must be disabled before torch is imported.  The runner never
# chooses a CPU device; an unavailable/unsupported MPS execution fails instead.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only runner.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
FUNIE_PYTORCH = PROJECT / "third_party/FUnIE-GAN/PyTorch"
PAIR_MANIFEST = PROJECT / "datasets/processed/uieb_enhancement_v2_leakage_resolved/pair_split_manifest.csv"
CHALLENGING_MANIFEST = PROJECT / "datasets/processed/uieb_enhancement_v2_leakage_resolved/challenging_manifest.csv"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
FREEZE_RECORD = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.json"
RUNS_ROOT = PROJECT / "training_runs/funie"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_frozen_rows(split: str) -> list[dict[str, str]]:
    with PAIR_MANIFEST.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["derived_split"] == split]
    if not rows:
        raise RuntimeError(f"No UIEB rows found for frozen split {split!r}")
    return sorted(rows, key=lambda row: row["filename"])


class FrozenUIEBPairs(Dataset[dict[str, torch.Tensor | str]]):
    """Manifest-driven paired data with deterministic aligned augmentation."""

    def __init__(self, rows: list[dict[str, str]], train: bool, seed: int) -> None:
        self.rows = rows
        self.train = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        with Image.open(PROJECT / row["raw_path"]) as raw_handle:
            raw = raw_handle.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        with Image.open(PROJECT / row["reference_path"]) as reference_handle:
            reference = reference_handle.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        # The official loader flips both members of a pair together with p=0.5.
        # Deterministic per-epoch sampling makes pilot and resumed runs repeatable.
        if self.train and random.Random(self.seed + self.epoch * 1_000_003 + index).random() < 0.5:
            raw = raw.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            reference = reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {"raw": self._to_tensor(raw), "reference": self._to_tensor(reference), "filename": row["filename"]}

    def __len__(self) -> int:
        return len(self.rows)


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built():
        raise SystemExit("This PyTorch installation was not built with MPS support; refusing CPU fallback.")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; refusing CPU fallback. Run from native macOS Terminal with Metal access.")
    device = torch.device("mps")
    probe = torch.ones(1, device=device)
    if probe.device.type != "mps":
        raise RuntimeError(f"MPS probe returned unexpected device {probe.device}")
    return device


def require_finite(value: torch.Tensor, label: str) -> float:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite {label}")
    return float(value.detach().cpu())


def mps_memory() -> dict[str, int | None]:
    observed: dict[str, int | None] = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        method = getattr(torch.mps, name, None)
        if method is None:
            observed[name] = None
            continue
        try:
            observed[name] = int(method())
        except RuntimeError:
            observed[name] = None
    return observed


def capture_rng() -> dict[str, Any]:
    """Capture every RNG state used by this runner for future exact resumes."""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    # The pilot checkpoint predates MPS RNG capture. The recipe has no stochastic
    # MPS layers, and data order/augmentation are epoch/index deterministic, so
    # it can continue faithfully; later checkpoints also retain this state.
    get_mps_rng_state = getattr(torch.mps, "get_rng_state", None)
    try:
        state["mps"] = get_mps_rng_state() if get_mps_rng_state is not None else None
    except RuntimeError:
        state["mps"] = None
    return state


def restore_rng(state: dict[str, Any]) -> dict[str, bool]:
    restored = {"python": False, "numpy": False, "torch": False, "mps": False}

    def cpu_byte_tensor(value: Any, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} RNG state is not a torch tensor")
        # Checkpoints are otherwise loaded onto MPS for model restoration, but
        # both RNG setter APIs require their serialized CPU ByteTensor state.
        return value.detach().to(device="cpu", dtype=torch.uint8)

    if "python" in state:
        random.setstate(state["python"])
        restored["python"] = True
    if "numpy" in state:
        np.random.set_state(state["numpy"])
        restored["numpy"] = True
    if "torch" in state:
        torch.set_rng_state(cpu_byte_tensor(state["torch"], "CPU torch"))
        restored["torch"] = True
    set_mps_rng_state = getattr(torch.mps, "set_rng_state", None)
    if state.get("mps") is not None and set_mps_rng_state is not None:
        set_mps_rng_state(cpu_byte_tensor(state["mps"], "MPS"))
        restored["mps"] = True
    return restored


def module_state_matches(module: torch.nn.Module, expected: dict[str, torch.Tensor]) -> bool:
    actual = module.state_dict()
    if actual.keys() != expected.keys():
        return False
    return all(torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu()) for name in actual)


def state_values_match(actual: Any, expected: Any) -> bool:
    """Compare serialized state after moving tensor values to CPU."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(state_values_match(actual[key], expected[key]) for key in actual)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(state_values_match(left, right) for left, right in zip(actual, expected))
    return actual == expected


def optimizer_resume_summary(optimizer: torch.optim.Optimizer, expected: dict[str, Any]) -> dict[str, int | bool]:
    actual = optimizer.state_dict()
    return {
        "state_entries_expected": len(expected["state"]),
        "state_entries_restored": len(actual["state"]),
        "parameter_groups_expected": len(expected["param_groups"]),
        "parameter_groups_restored": len(actual["param_groups"]),
        "entry_count_matches": len(actual["state"]) == len(expected["state"]),
        "group_count_matches": len(actual["param_groups"]) == len(expected["param_groups"]),
        "state_values_match": state_values_match(actual, expected),
    }


def gaussian_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean 11x11 Gaussian-window SSIM for image tensors in [0, 1]."""
    coordinates = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel_1d = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11).contiguous()
    mu_pred = F.conv2d(prediction, kernel, padding=5, groups=3)
    mu_target = F.conv2d(target, kernel, padding=5, groups=3)
    mu_pred_sq, mu_target_sq, mu_cross = mu_pred.square(), mu_target.square(), mu_pred * mu_target
    sigma_pred = F.conv2d(prediction.square(), kernel, padding=5, groups=3) - mu_pred_sq
    sigma_target = F.conv2d(target.square(), kernel, padding=5, groups=3) - mu_target_sq
    sigma_cross = F.conv2d(prediction * target, kernel, padding=5, groups=3) - mu_cross
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2))
    return score.mean()


def metrics_for_pair(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = ((prediction.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    target = ((target.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = F.mse_loss(prediction, target)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    ssim = gaussian_ssim(prediction, target)
    return require_finite(psnr, "validation PSNR"), require_finite(ssim, "validation SSIM")


def validation_baseline(validation_dataset: Dataset[Any], device: torch.device) -> dict[str, Any]:
    """Evaluate unenhanced raw input against the reference with pilot metrics."""
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with torch.inference_mode():
        for batch in DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0):
            raw = batch["raw"].to(device)
            reference = batch["reference"].to(device)
            psnr, ssim = metrics_for_pair(raw, reference)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    return {
        "pairs": len(psnr_values),
        "psnr_mean": float(np.mean(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "metric_preprocessing": "same 256x256 bicubic RGB resize and [-1,1] to [0,1] conversion as enhanced validation metrics",
        "device": str(device),
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
    }


def save_validation_curves(metrics_path: Path, baseline: dict[str, Any], output_dir: Path) -> None:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    rows.sort(key=lambda row: int(row["epoch"]))
    curve_data = {
        "baseline": baseline,
        "epochs": [int(row["epoch"]) for row in rows],
        "validation_psnr": [float(row["validation"]["psnr_mean"]) for row in rows],
        "validation_ssim": [float(row["validation"]["ssim_mean"]) for row in rows],
        "generator_loss": [float(row["loss_mean"]["g_loss"]) for row in rows],
        "discriminator_loss": [float(row["loss_mean"]["d_loss"]) for row in rows],
        "epoch_elapsed_seconds": [row.get("epoch_elapsed_seconds") for row in rows],
        "mps_current_allocated_memory": [row.get("mps_memory", {}).get("current_allocated_memory") for row in rows],
        "mps_driver_allocated_memory": [row.get("mps_memory", {}).get("driver_allocated_memory") for row in rows],
        "process_max_rss": [row.get("process_max_rss") for row in rows],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_curves.json").write_text(json.dumps(curve_data, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = curve_data["epochs"]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(epochs, curve_data["validation_psnr"], marker="o", label="enhanced PSNR")
    axes[0].axhline(float(baseline["psnr_mean"]), color="gray", linestyle="--", label="raw-input PSNR baseline")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    right_axis = axes[0].twinx()
    right_axis.plot(epochs, curve_data["validation_ssim"], marker="s", color="tab:green", label="enhanced SSIM")
    right_axis.axhline(float(baseline["ssim_mean"]), color="tab:green", linestyle=":", label="raw-input SSIM baseline")
    right_axis.set_ylabel("SSIM")
    axes[1].plot(epochs, curve_data["generator_loss"], marker="o", label="generator loss")
    axes[1].plot(epochs, curve_data["discriminator_loss"], marker="s", label="discriminator loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean training loss")
    axes[1].legend()
    durations = [np.nan if value is None else float(value) for value in curve_data["epoch_elapsed_seconds"]]
    driver_gib = [np.nan if value is None else float(value) / 1024**3 for value in curve_data["mps_driver_allocated_memory"]]
    axes[2].plot(epochs, durations, marker="o", color="tab:purple", label="epoch time")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("seconds")
    memory_axis = axes[2].twinx()
    memory_axis.plot(epochs, driver_gib, marker="s", color="tab:orange", label="MPS driver allocation")
    memory_axis.set_ylabel("GiB")
    axes[2].legend(loc="upper left")
    memory_axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_curves.png", dpi=150)
    plt.close(figure)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    array = pixels.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_comparison(epoch: int, generator: torch.nn.Module, preview_loader: DataLoader[Any], device: torch.device, output: Path) -> list[str]:
    generator.eval()
    columns: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    names: list[str] = []
    with torch.inference_mode():
        for batch in preview_loader:
            raw = batch["raw"].to(device)
            reference = batch["reference"].to(device)
            generated = generator(raw)
            columns.append((tensor_to_image(raw[0]), tensor_to_image(generated[0]), tensor_to_image(reference[0])))
            names.append(str(batch["filename"][0]))
    canvas = Image.new("RGB", (256 * len(columns), 256 * 3))
    for index, column in enumerate(columns):
        for row, image in enumerate(column):
            canvas.paste(image, (index * 256, row * 256))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return names


def checkpoint_payload(
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    best_psnr: float,
    run_id: str,
) -> dict[str, Any]:
    return {
        "format": "aqua_vision_funie_full_recipe_v1",
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_psnr": best_psnr,
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "source_revision": git_revision(FUNIE_PYTORCH.parent),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "rng": capture_rng(),
    }


def verify_checkpoint(checkpoint: Path, raw: torch.Tensor, device: torch.device) -> dict[str, Any]:
    # The checkpoint is a locally generated run artifact. Resume/reload does not
    # accept arbitrary external checkpoints.
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    restored_generator = GeneratorFunieGAN().to(device)
    restored_discriminator = DiscriminatorFunieGAN().to(device)
    restored_generator.load_state_dict(payload["generator"])
    restored_discriminator.load_state_dict(payload["discriminator"])
    restored_generator.eval()
    with torch.inference_mode():
        output = restored_generator(raw)
    if output.device.type != "mps" or not torch.isfinite(output).all():
        raise RuntimeError("Reloaded checkpoint inference failed on MPS")
    return {"passed": True, "output_shape": list(output.shape), "checkpoint_sha256": sha256(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique directory name under training_runs/funie/")
    parser.add_argument("--epochs", type=int, default=1, help="Total epoch count; the pilot command uses 1")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional per-epoch cap; omit for the complete frozen training epoch")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--resume", type=Path, default=None, help="Local checkpoint from this same --run-id")
    args = parser.parse_args()
    if args.epochs < 1 or args.preview_count < 1:
        raise SystemExit("--epochs and --preview-count must be positive")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        raise SystemExit("--max-train-steps must be positive when provided")
    if not (PAIR_MANIFEST.is_file() and CHALLENGING_MANIFEST.is_file() and FREEZE_FILE.is_file() and FREEZE_RECORD.is_file()):
        raise SystemExit("Frozen UIEB manifests/freeze record are required")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")

    device = require_mps()
    run_dir = RUNS_ROOT / args.run_id
    checkpoints = run_dir / "checkpoints"
    comparisons = run_dir / "validation_comparisons"
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"
    baseline_path = run_dir / "validation_baseline.json"
    curves_dir = run_dir / "validation_curves"
    resume_events_path = run_dir / "resume_events.jsonl"
    train_rows, validation_rows = read_frozen_rows("train"), read_frozen_rows("val")
    if len(train_rows) != 623 or len(validation_rows) != 134:
        raise RuntimeError("Unexpected frozen UIEB v2 train/validation counts")
    if args.preview_count > len(validation_rows):
        raise SystemExit("--preview-count exceeds frozen validation-pair count")

    if args.resume is None:
        if run_dir.exists():
            raise SystemExit(f"Refusing to overwrite existing run: {run_dir}; use a new --run-id or --resume")
        run_dir.mkdir(parents=True)
    else:
        if not args.resume.is_file() or not run_dir.is_dir() or not config_path.is_file():
            raise SystemExit("--resume requires an existing local checkpoint and its matching --run-id directory")
        if run_dir.resolve() not in args.resume.resolve().parents:
            raise SystemExit("--resume checkpoint must be inside this run directory")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.commons import VGG19_PercepLoss, Weights_Normal
    from nets.funiegan import DiscriminatorFunieGAN, GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    discriminator = DiscriminatorFunieGAN().to(device)
    # The pinned source uses a pretrained VGG19 feature extractor at layer 30
    # (conv5_2), freezes its parameters, and applies no additional ImageNet
    # normalization after paired tensors were normalized to [-1, 1].
    perceptual = VGG19_PercepLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.99))
    adversarial = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    start_epoch, global_step, best_psnr = 0, 0, -math.inf
    vgg_path = Path(torch.hub.get_dir()) / "checkpoints" / "vgg19-dcbb9e9d.pth"

    if args.resume is None:
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)
        configuration = {
            "format": "aqua_vision_funie_full_recipe_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "device_requirement": "MPS only; PYTORCH_ENABLE_MPS_FALLBACK=0",
            "device": str(device),
            "seed": args.seed,
            "source": {"repository": "third_party/FUnIE-GAN", "revision": git_revision(FUNIE_PYTORCH.parent)},
            "frozen_manifest": {"path": str(FREEZE_FILE.relative_to(PROJECT)), "sha256": sha256(FREEZE_FILE)},
            "data": {
                "train_manifest": str(PAIR_MANIFEST.relative_to(PROJECT)),
                "validation_manifest": str(PAIR_MANIFEST.relative_to(PROJECT)),
                "train_pairs": len(train_rows),
                "validation_pairs": len(validation_rows),
                "excluded": ["derived UIEB test: 133 pairs", "UIEB challenging set: 60 images"],
                "preprocessing": "paired RGB resize 256x256 bicubic; tensors normalized to [-1,1]; deterministic aligned p=0.5 horizontal flip only for train",
            },
            "initialization": "official Weights_Normal on generator and discriminator; pretrained ImageNet VGG19 feature extractor frozen",
            "vgg19": {
                "source_class": "nets.commons.VGG19_PercepLoss",
                "feature_layer": "module 30 / conv5_2",
                "source_weight_url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
                "local_path": str(vgg_path.relative_to(PROJECT)) if vgg_path.is_relative_to(PROJECT) else str(vgg_path),
                "sha256": sha256(vgg_path) if vgg_path.is_file() else None,
                "input_semantics": "[-1,1] paired tensors passed directly, matching the pinned official implementation",
            },
            "loss_recipe": {
                "discriminator": "0.5 * (MSE(D(reference, raw), 1) + MSE(D(G(raw), raw), 0)) * 10",
                "generator": "MSE(D(G(raw), raw), 1) + 7 * L1(G(raw), reference) + 3 * VGG19_conv5_2_MSE(G(raw), reference)",
                "note": "The discriminator fake image is generated under no_grad during the D update; this preserves the objective while avoiding an unused generator gradient.",
            },
            "pilot_arguments": {"epochs": args.epochs, "batch_size": 1, "max_train_steps": args.max_train_steps, "preview_count": args.preview_count},
        }
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    else:
        # Keep serialized RNG byte states on CPU. load_state_dict transfers
        # model and optimizer tensors to their MPS parameters as required.
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("frozen_manifest_sha256") != sha256(FREEZE_FILE):
            raise RuntimeError("Resume checkpoint does not match the current frozen manifest")
        saved_config = json.loads(config_path.read_text())
        if saved_config.get("seed") != args.seed or saved_config.get("source", {}).get("revision") != git_revision(FUNIE_PYTORCH.parent):
            raise RuntimeError("Resume command does not match the saved seed or pinned FUnIE-GAN source revision")
        saved_launch = saved_config.get("pilot_arguments", {})
        if saved_launch.get("max_train_steps") != args.max_train_steps or saved_launch.get("preview_count") != args.preview_count:
            raise RuntimeError("Resume command must preserve the saved training-step cap and fixed validation-preview count")
        saved_vgg_hash = saved_config.get("vgg19", {}).get("sha256")
        if saved_vgg_hash is None or not vgg_path.is_file() or sha256(vgg_path) != saved_vgg_hash:
            raise RuntimeError("Resume cannot verify the pinned VGG19 perceptual-weight provenance")
        if payload.get("run_id") != args.run_id:
            raise RuntimeError("Resume checkpoint run ID does not match --run-id")
        generator.load_state_dict(payload["generator"])
        discriminator.load_state_dict(payload["discriminator"])
        optimizer_g.load_state_dict(payload["optimizer_g"])
        optimizer_d.load_state_dict(payload["optimizer_d"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_psnr = float(payload.get("best_validation_psnr", -math.inf))
        rng = payload.get("rng", {}) if isinstance(payload.get("rng", {}), dict) else {}
        rng_restored = restore_rng(rng)
        resume_verification = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "resume_checkpoint": str(args.resume.resolve()),
            "checkpoint_sha256": sha256(args.resume),
            "restored_epoch": int(payload["epoch"]),
            "next_epoch": start_epoch,
            "restored_global_step": global_step,
            "requested_total_epochs": args.epochs,
            "preview_count": args.preview_count,
            "max_train_steps": args.max_train_steps,
            "perceptual_vgg_sha256": saved_vgg_hash,
            "generator_state_matches": module_state_matches(generator, payload["generator"]),
            "discriminator_state_matches": module_state_matches(discriminator, payload["discriminator"]),
            "optimizer_g": optimizer_resume_summary(optimizer_g, payload["optimizer_g"]),
            "optimizer_d": optimizer_resume_summary(optimizer_d, payload["optimizer_d"]),
            "rng_restored": rng_restored,
            "mps_rng_was_present": rng.get("mps") is not None,
            "note": "The epoch-001 pilot predates MPS RNG capture. This FUnIE recipe has no stochastic MPS layers and uses deterministic per-epoch data order and aligned flips; future checkpoints include MPS RNG state.",
        }
        if not (resume_verification["generator_state_matches"] and resume_verification["discriminator_state_matches"] and resume_verification["optimizer_g"]["entry_count_matches"] and resume_verification["optimizer_g"]["group_count_matches"] and resume_verification["optimizer_g"]["state_values_match"] and resume_verification["optimizer_d"]["entry_count_matches"] and resume_verification["optimizer_d"]["group_count_matches"] and resume_verification["optimizer_d"]["state_values_match"]):
            raise RuntimeError("Resume state verification failed")
        with resume_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(resume_verification) + "\n")
        if start_epoch >= args.epochs:
            raise SystemExit("Resume checkpoint already reaches --epochs; increase --epochs to continue")

    train_dataset = FrozenUIEBPairs(train_rows, train=True, seed=args.seed)
    validation_dataset = FrozenUIEBPairs(validation_rows, train=False, seed=args.seed)
    preview_dataset = FrozenUIEBPairs(validation_rows[: args.preview_count], train=False, seed=args.seed)
    preview_loader = DataLoader(preview_dataset, batch_size=1, shuffle=False, num_workers=0)
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        if baseline.get("frozen_manifest_sha256") != sha256(FREEZE_FILE) or baseline.get("pairs") != len(validation_dataset):
            raise RuntimeError("Existing validation baseline does not match the frozen validation manifest")
    else:
        baseline = validation_baseline(validation_dataset, device)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        parameter_before_g = next(generator.parameters()).detach().clone()
        parameter_before_d = next(discriminator.parameters()).detach().clone()
        generator.train()
        discriminator.train()
        train_dataset.set_epoch(epoch)
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, generator=loader_generator)
        epoch_losses: list[dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            raw = batch["raw"].to(device)
            reference = batch["reference"].to(device)
            if raw.device.type != "mps" or reference.device.type != "mps":
                raise RuntimeError("A training tensor was not placed on MPS")

            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_for_d = generator(raw)
            prediction_real = discriminator(reference, raw)
            prediction_fake = discriminator(fake_for_d, raw)
            d_loss = 0.5 * (adversarial(prediction_real, torch.ones_like(prediction_real)) + adversarial(prediction_fake, torch.zeros_like(prediction_fake))) * 10.0
            require_finite(d_loss, "discriminator loss")
            d_loss.backward()
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
            optimizer_g.step()
            epoch_losses.append({
                "d_loss": require_finite(d_loss, "recorded discriminator loss"),
                "g_loss": require_finite(g_loss, "recorded generator loss"),
                "gan_loss": require_finite(gan_loss, "recorded GAN loss"),
                "l1_loss": require_finite(l1_loss, "recorded L1 loss"),
                "perceptual_loss": require_finite(perceptual_loss, "recorded perceptual loss"),
            })
            global_step += 1
            if args.max_train_steps is not None and batch_index >= args.max_train_steps:
                break

        if not epoch_losses:
            raise RuntimeError("Pilot completed no training steps")
        if torch.equal(parameter_before_g, next(generator.parameters()).detach()) or torch.equal(parameter_before_d, next(discriminator.parameters()).detach()):
            raise RuntimeError("A model parameter did not update during training")

        generator.eval()
        validation_psnr: list[float] = []
        validation_ssim: list[float] = []
        with torch.inference_mode():
            for batch in DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0):
                raw = batch["raw"].to(device)
                reference = batch["reference"].to(device)
                generated = generator(raw)
                psnr, ssim = metrics_for_pair(generated, reference)
                validation_psnr.append(psnr)
                validation_ssim.append(ssim)
        mean_psnr = float(np.mean(validation_psnr))
        mean_ssim = float(np.mean(validation_ssim))
        if not math.isfinite(mean_psnr) or not math.isfinite(mean_ssim):
            raise RuntimeError("Non-finite validation metric")
        comparison_path = comparisons / f"epoch_{epoch + 1:03d}.png"
        preview_names = save_comparison(epoch, generator, preview_loader, device, comparison_path)
        checkpoint = checkpoints / f"epoch_{epoch + 1:03d}.pt"
        payload = checkpoint_payload(epoch, global_step, generator, discriminator, optimizer_g, optimizer_d, best_psnr, args.run_id)
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
        payload["best_validation_psnr"] = best_psnr
        atomic_torch_save(payload, checkpoint)
        latest = checkpoints / "latest.pt"
        shutil.copy2(checkpoint, latest)
        if mean_psnr >= best_psnr:
            shutil.copy2(checkpoint, checkpoints / "best_validation_psnr.pt")
        # Reload before recording success so the one-epoch pilot validates the
        # saved artifact, not merely the in-memory model.
        fixed_raw = next(iter(preview_loader))["raw"].to(device)
        reload_result = verify_checkpoint(latest, fixed_raw, device)
        summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_steps": len(epoch_losses),
            "loss_mean": {key: float(np.mean([item[key] for item in epoch_losses])) for key in epoch_losses[0]},
            "validation": {"pairs": len(validation_psnr), "psnr_mean": mean_psnr, "ssim_mean": mean_ssim},
            "comparison_image": str(comparison_path.relative_to(PROJECT)),
            "comparison_filenames": preview_names,
            "checkpoint": str(latest.relative_to(PROJECT)),
            "checkpoint_reload": reload_result,
            "device": str(device),
            "mps_memory": mps_memory(),
            "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "epoch_elapsed_seconds": round(time.perf_counter() - epoch_started, 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "frozen_manifest_sha256": sha256(FREEZE_FILE),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        save_validation_curves(metrics_path, baseline, curves_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
