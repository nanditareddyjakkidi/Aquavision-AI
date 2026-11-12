#!/usr/bin/env python3
"""Record the preserved epoch-20 frozen-BatchNorm FUnIE-GAN as a model candidate.

This script is metadata-only.  It reads the completed run and its frozen
validation results, writes a candidate JSON record plus a per-image regression
summary, and never modifies a checkpoint, dataset, split, or website file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[1]
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
RUN_ID = "funie_batchnorm_frozen_e2to4_mps_20260930"
RUN_DIR = PROJECT / "training_runs/funie" / RUN_ID
CHECKPOINTS = RUN_DIR / "checkpoints"
COMPARISON_DIR = RUN_DIR / "comparison_against_original"
OUTPUT_DIR = PROJECT / "model_candidates"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="funie_gan_frozen_batchnorm_epoch20_mps")
    args = parser.parse_args()
    if "/" in args.candidate_id or args.candidate_id in {"", ".", ".."}:
        raise SystemExit("--candidate-id must be a simple filename stem")

    config_path = RUN_DIR / "run_config.json"
    metrics_path = RUN_DIR / "metrics.jsonl"
    best_checkpoint = CHECKPOINTS / "best_validation_psnr.pt"
    latest_checkpoint = CHECKPOINTS / "latest.pt"
    per_image_path = COMPARISON_DIR / "per_image_metrics_epoch_020.csv"
    required = [FREEZE_FILE, config_path, metrics_path, best_checkpoint, latest_checkpoint, per_image_path]
    if not all(path.is_file() for path in required):
        raise SystemExit("The completed epoch-20 BatchNorm-frozen FUnIE-GAN artifacts are required")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    epoch20 = next((row for row in metrics if int(row["epoch"]) == 20), None)
    if epoch20 is None or len(metrics) != 19:
        raise RuntimeError("Expected the complete separate BatchNorm run through epoch 20")
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    latest_payload = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    for name, payload in (("best", best_payload), ("latest", latest_payload)):
        if (
            int(payload.get("epoch", -1)) != 19
            or payload.get("format") != "aqua_vision_funie_batchnorm_frozen_v1"
            or payload.get("batchnorm_statistics_frozen") is not True
            or payload.get("frozen_manifest_sha256") != sha256(FREEZE_FILE)
        ):
            raise RuntimeError(f"The {name} checkpoint is not the preserved epoch-20 frozen-BatchNorm artifact")
    if sha256(best_checkpoint) != sha256(latest_checkpoint):
        raise RuntimeError("Best and latest epoch-20 checkpoints differ unexpectedly")

    with per_image_path.open(newline="", encoding="utf-8") as handle:
        per_image = list(csv.DictReader(handle))
    if len(per_image) != 134:
        raise RuntimeError("Expected exactly 134 frozen validation-pair metrics")
    for row in per_image:
        row["ablation_psnr_delta"] = float(row["ablation_psnr_delta"])
        row["ablation_ssim_delta"] = float(row["ablation_ssim_delta"])
    psnr_regressions = [row for row in per_image if row["ablation_psnr_delta"] < 0.0]
    ssim_regressions = [row for row in per_image if row["ablation_ssim_delta"] < 0.0]
    any_regressions = [row for row in per_image if row in psnr_regressions or row in ssim_regressions]
    any_regressions.sort(key=lambda row: (row["ablation_psnr_delta"] + 10.0 * row["ablation_ssim_delta"], row["filename"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record_path = OUTPUT_DIR / f"{args.candidate_id}.json"
    regression_path = OUTPUT_DIR / f"{args.candidate_id}_validation_regressions.md"
    if record_path.exists() or regression_path.exists():
        raise SystemExit("Candidate record already exists; refusing to overwrite it")

    psnr_deltas = floats(per_image, "ablation_psnr_delta")
    ssim_deltas = floats(per_image, "ablation_ssim_delta")
    regression_rows = "\n".join(
        f"| {row['filename']} | {row['ablation_psnr_delta']:+.4f} | {row['ablation_ssim_delta']:+.5f} |"
        for row in any_regressions
    )
    regression_path.write_text(
        f"""# Epoch-20 frozen-BatchNorm FUnIE-GAN validation regressions

The comparison uses all 134 frozen UIEB v2 validation pairs and the identical
raw-input metric preprocessing. These are relative-to-raw regressions, not
test-set results.

- PSNR: {len(psnr_regressions)} worsened, {sum(delta > 0 for delta in psnr_deltas)} improved;
  mean delta {sum(psnr_deltas) / len(psnr_deltas):+.4f} dB and median
  {median(psnr_deltas):+.4f} dB.
- SSIM: {len(ssim_regressions)} worsened, {sum(delta > 0 for delta in ssim_deltas)} improved;
  mean delta {sum(ssim_deltas) / len(ssim_deltas):+.5f} and median
  {median(ssim_deltas):+.5f}.
- {len(any_regressions)} images regressed on at least one metric. Their exact
  original/raw/output/reference metrics remain in
  `{per_image_path.relative_to(PROJECT)}`.

| Filename | Δ PSNR vs raw | Δ SSIM vs raw |
| --- | ---: | ---: |
{regression_rows}
""",
        encoding="utf-8",
    )

    record: dict[str, Any] = {
        "candidate_id": args.candidate_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_test_evaluated",
        "task": "underwater image enhancement",
        "model_description": "FUnIE-GAN with frozen BatchNorm running statistics (epoch-1 running means, variances, and counters frozen; affine weight/bias parameters remained trainable).",
        "checkpoint": {
            "path": str(best_checkpoint.relative_to(PROJECT)),
            "sha256": sha256(best_checkpoint),
            "epoch": int(best_payload["epoch"]) + 1,
            "global_step": int(best_payload["global_step"]),
            "best_validation_psnr": float(best_payload["best_validation_psnr"]),
            "latest_checkpoint_path": str(latest_checkpoint.relative_to(PROJECT)),
            "latest_checkpoint_sha256": sha256(latest_checkpoint),
        },
        "source_revision": best_payload["source_revision"],
        "frozen_manifest": {"path": str(FREEZE_FILE.relative_to(PROJECT)), "sha256": sha256(FREEZE_FILE)},
        "configuration": {
            "run_config_path": str(config_path.relative_to(PROJECT)),
            "seed": config["seed"],
            "batch_size": config["batch_size"],
            "loss_recipe": config["loss_recipe"],
            "preprocessing": config["data"]["preprocessing"],
            "excluded_data": config["data"]["excluded"],
            "only_experimental_change": config["only_change"],
            "vgg19": config["vgg19"],
        },
        "validation": {
            "pairs": epoch20["validation"]["pairs"],
            "psnr_mean": epoch20["validation"]["psnr_mean"],
            "ssim_mean": epoch20["validation"]["ssim_mean"],
            "raw_baseline": config["raw_validation_baseline"],
            "per_image_metrics": str(per_image_path.relative_to(PROJECT)),
            "remaining_regressions_summary": str(regression_path.relative_to(PROJECT)),
        },
        "remaining_per_image_regressions": {
            "psnr_worsened": len(psnr_regressions),
            "ssim_worsened": len(ssim_regressions),
            "any_metric_worsened": len(any_regressions),
            "psnr_mean_delta_vs_raw": sum(psnr_deltas) / len(psnr_deltas),
            "psnr_median_delta_vs_raw": median(psnr_deltas),
            "ssim_mean_delta_vs_raw": sum(ssim_deltas) / len(ssim_deltas),
            "ssim_median_delta_vs_raw": median(ssim_deltas),
        },
        "scope_guard": "UIEB test pairs and challenging images were not used for training or candidate selection. This record does not replace detector inputs with enhanced images.",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "recorded", "candidate_record": str(record_path.relative_to(PROJECT)), "regression_summary": str(regression_path.relative_to(PROJECT)), "checkpoint_sha256": record["checkpoint"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
