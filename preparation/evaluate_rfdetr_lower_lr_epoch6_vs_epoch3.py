#!/usr/bin/env python3
"""MPS-only validation comparison of lower-LR epoch 6 and no-jitter epoch 3.

The epoch-3 metrics, per-image threshold counts, and foreground predictions are
reused only after their checkpoint, data-freeze, and standalone-evaluation
provenance checks pass. The lower-LR epoch-6 EMA receives one inference-only
pass on the same 1,082 frozen validation images. No training or test-data
access occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only evaluation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_rfdetr_epoch5_ema_validation import SHARED_THRESHOLD, coco_area_counts
from evaluate_rfdetr_raw_vs_funie_enhanced import draw_variant, evaluate_detector_variant, finite, write_json
from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING, FREEZE_FILE, PROJECT, RFDETR_SOURCE, RUNS_ROOT, VAL_COCO,
    check_coco_payload, digest, frozen_entries, load_manifest, mps_memory,
    require_mps, verify_frozen_file,
)

ROOT = RUNS_ROOT / "evaluations"
LOWER_RUN = RUNS_ROOT / "rfdetr_no_jitter_epoch3_lower_lr_e4to8_mps_20261003T054821Z"
LOWER_CHECKPOINT = LOWER_RUN / "training_output/checkpoint_best_ema.pth"
LOWER_RESULT = LOWER_RUN / "lower_lr_result.json"
LOWER_SHA = "7670935cb63af62f4a9471dc8081b62dd34d41cf8ff1fdca442bbe9f79a0e753"
LOWER_AP = 0.18791550397872925

E3_RUN = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z"
E3_CHECKPOINT = E3_RUN / "training_output/checkpoint_best_ema.pth"
E3_SHA = "6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440"
E3_AP = 0.17765460908412933
E3_EVAL_DIR = ROOT / "rfdetr_scale_jitter_off_epoch3_ema_validation_20261002T161631Z"
E3_EVAL = E3_EVAL_DIR / "evaluation_result.json"
E3_PREDICTIONS = E3_EVAL_DIR / "foreground_coco_predictions.json"
E3_DETAIL_DIR = ROOT / "rfdetr_raw_vs_funie_enhanced_val_mps_20261002T163013Z"
E3_DETAIL = E3_DETAIL_DIR / "raw_vs_funie_enhanced_result.json"
E3_PER_IMAGE = E3_DETAIL_DIR / "per_image_shared_threshold_043.csv"
FIXED_PREVIEWS = LOWER_RUN / "preview_plan.json"

IMAGE_COUNT, ANNOTATION_COUNT, TOLERANCE = 1082, 289, 0.001
METRICS = (
    "AP_50_95", "AP50", "AP75",
    "AP_small", "AP_medium", "AP_large",
    "AR_small_500", "AR_medium_500", "AR_large_500",
)


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} does not contain an object")
    return payload


def pinned(path: Path, expected: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return {"path": str(path.relative_to(PROJECT)), "sha256": actual}


def aggregate(quality: dict[int, dict[str, Any]], annotations: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    tp = sum(int(item["tp"]) for item in quality.values())
    fp = sum(int(item["fp"]) for item in quality.values())
    fn = sum(int(item["fn"]) for item in quality.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negatives = [image_id for image_id in quality if not annotations.get(image_id, [])]
    negative_fp = sum(int(quality[image_id]["fp"]) for image_id in negatives)
    negative_detected = sum(int(quality[image_id]["fp"]) > 0 for image_id in negatives)
    return {
        "threshold": SHARED_THRESHOLD, "iou_threshold": 0.5,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "F1": f1,
        "negative_images": len(negatives),
        "false_positives_on_negative_images": negative_fp,
        "false_positives_per_negative_image": negative_fp / len(negatives) if negatives else 0.0,
        "negative_images_with_false_detection": negative_detected,
        "fraction_negative_images_with_false_detection": negative_detected / len(negatives) if negatives else 0.0,
    }


def load_epoch3(
    frozen: dict[str, str], images: dict[int, dict[str, Any]], annotations: dict[int, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    checkpoint = pinned(E3_CHECKPOINT, E3_SHA, "preserved no-jitter epoch-3 EMA checkpoint")
    standalone = load_json(E3_EVAL, "epoch-3 standalone evaluation")
    if standalone.get("status") != "passed" or standalone.get("checkpoint", {}).get("sha256") != E3_SHA:
        raise RuntimeError("Epoch-3 standalone evaluation does not match its preserved checkpoint")
    if standalone.get("frozen_checks") != {key: frozen[key] for key in standalone.get("frozen_checks", {})}:
        raise RuntimeError("Epoch-3 standalone evaluation uses different frozen files")
    if int(standalone.get("validation_images", -1)) != IMAGE_COUNT:
        raise RuntimeError("Epoch-3 standalone evaluation does not cover all validation images")
    reproduce = standalone.get("reproduction_check", {})
    if not reproduce.get("passed") or abs(float(reproduce.get("recorded_AP_50_95", math.nan)) - E3_AP) > 1e-12:
        raise RuntimeError("Epoch-3 standalone evaluation lacks a passing AP reproduction check")

    metrics = standalone.get("training_compatible_ema_metrics")
    shared = standalone.get("precision_recall_iou50_shared_threshold")
    best = standalone.get("precision_recall_iou50_validation_selected_best_f1")
    if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
        raise RuntimeError("Epoch-3 cache lacks required COCO metrics")
    if not isinstance(shared, dict) or float(shared.get("threshold", math.nan)) != SHARED_THRESHOLD:
        raise RuntimeError("Epoch-3 cache lacks the shared-threshold results")
    if not isinstance(best, dict) or "best_confidence_threshold" not in best:
        raise RuntimeError("Epoch-3 cache lacks best-F1 selection")
    for key in METRICS:
        finite(metrics[key], f"epoch-3 {key}")

    detail = load_json(E3_DETAIL, "epoch-3 threshold-detail evaluation")
    if detail.get("status") != "passed" or detail.get("detector", {}).get("sha256") != E3_SHA:
        raise RuntimeError("Epoch-3 threshold detail has different checkpoint provenance")
    if detail.get("frozen_checks") != frozen:
        raise RuntimeError("Epoch-3 threshold detail uses different frozen files")
    if detail.get("raw", {}).get("training_compatible_metrics") != metrics:
        raise RuntimeError("Epoch-3 standalone metrics and threshold-detail metrics disagree")

    quality: dict[int, dict[str, Any]] = {}
    if not E3_PER_IMAGE.is_file():
        raise RuntimeError("Missing epoch-3 per-image threshold cache")
    with E3_PER_IMAGE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = int(row["image_id"])
            if image_id in quality or image_id not in images:
                raise RuntimeError("Epoch-3 per-image cache has an invalid image ID")
            if row["source_image_path"] != str(images[image_id]["file_name"]):
                raise RuntimeError("Epoch-3 per-image cache has a different source image")
            if int(row["ground_truth_instances"]) != len(annotations.get(image_id, [])):
                raise RuntimeError("Epoch-3 per-image cache has different annotations")
            quality[image_id] = {
                "tp": int(row["raw_tp"]), "fp": int(row["raw_fp"]),
                "fn": int(row["raw_fn"]), "F1": float(row["raw_F1"]),
            }
    if set(quality) != set(images):
        raise RuntimeError("Epoch-3 per-image cache does not cover the frozen split")
    counts = aggregate(quality, annotations)
    for key in ("F1", "precision", "recall"):
        if abs(counts[key] - float(shared[key])) > 1e-9:
            raise RuntimeError("Epoch-3 shared threshold counts do not reproduce cached metrics")

    if not E3_PREDICTIONS.is_file():
        raise RuntimeError("Missing epoch-3 foreground prediction cache")
    all_predictions = json.loads(E3_PREDICTIONS.read_text(encoding="utf-8"))
    if not isinstance(all_predictions, list):
        raise RuntimeError("Epoch-3 foreground prediction cache is malformed")
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in all_predictions:
        image_id = int(item["image_id"])
        x, y, width, height = (float(value) for value in item["bbox"])
        score = float(item["score"])
        valid = (
            image_id in images and int(item["category_id"]) == 1 and math.isfinite(score)
            and width > 0 and height > 0 and all(math.isfinite(value) for value in (x, y, width, height))
        )
        if not valid:
            raise RuntimeError("Epoch-3 foreground cache contains invalid data")
        if score >= SHARED_THRESHOLD:
            predictions[image_id].append({"box": [x, y, x + width, y + height], "score": score})

    provenance = {
        "checkpoint": checkpoint,
        "standalone_result": {"path": str(E3_EVAL.relative_to(PROJECT)), "sha256": digest(E3_EVAL)},
        "threshold_detail": {"path": str(E3_DETAIL.relative_to(PROJECT)), "sha256": digest(E3_DETAIL)},
        "per_image_cache": {"path": str(E3_PER_IMAGE.relative_to(PROJECT)), "sha256": digest(E3_PER_IMAGE)},
        "foreground_prediction_cache": {"path": str(E3_PREDICTIONS.relative_to(PROJECT)), "sha256": digest(E3_PREDICTIONS)},
        "reused": True,
    }
    return {"metrics": metrics, "shared": shared, "best_f1": best, "reproduction": reproduce}, quality, predictions, provenance


def load_lower() -> dict[str, Any]:
    checkpoint = pinned(LOWER_CHECKPOINT, LOWER_SHA, "lower-LR epoch-6 best EMA checkpoint")
    result = load_json(LOWER_RESULT, "lower-LR continuation result")
    best = result.get("selected_best", {})
    if result.get("status") != "passed" or int(best.get("completed_epoch", -1)) != 6:
        raise RuntimeError("The lower-LR run does not identify epoch 6 as best")
    if best.get("checkpoint_sha256") != LOWER_SHA or abs(float(best.get("AP_50_95", math.nan)) - LOWER_AP) > 1e-12:
        raise RuntimeError("Lower-LR epoch-6 candidate does not match its stored selection record")
    return {"checkpoint": checkpoint, "recorded_AP_50_95": LOWER_AP, "run_result": str(LOWER_RESULT.relative_to(PROJECT))}


def rows_for(
    images: dict[int, dict[str, Any]], annotations: dict[int, list[dict[str, Any]]], groups: dict[str, str],
    old: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for image_id in sorted(images):
        e3, e6 = old[image_id], new[image_id]
        result.append({
            "image_id": image_id,
            "source_image_path": str(images[image_id]["file_name"]),
            "video_group": groups[str(images[image_id]["file_name"])],
            "ground_truth_instances": len(annotations.get(image_id, [])),
            "epoch3_tp": e3["tp"], "epoch3_fp": e3["fp"], "epoch3_fn": e3["fn"], "epoch3_F1": e3["F1"],
            "lower_lr_epoch6_tp": e6["tp"], "lower_lr_epoch6_fp": e6["fp"], "lower_lr_epoch6_fn": e6["fn"], "lower_lr_epoch6_F1": e6["F1"],
            "lower_lr_epoch6_minus_epoch3_F1": e6["F1"] - e3["F1"],
            "lower_lr_epoch6_minus_epoch3_tp": e6["tp"] - e3["tp"],
            "lower_lr_epoch6_minus_epoch3_fp": e6["fp"] - e3["fp"],
            "lower_lr_epoch6_minus_epoch3_fn": e6["fn"] - e3["fn"],
        })
    return result


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_group"])].append(row)
    summary: list[dict[str, Any]] = []
    for group, members in grouped.items():
        def totals(prefix: str) -> tuple[int, int, int]:
            return tuple(sum(int(row[f"{prefix}_{key}"]) for row in members) for key in ("tp", "fp", "fn"))
        old_tp, old_fp, old_fn = totals("epoch3")
        new_tp, new_fp, new_fn = totals("lower_lr_epoch6")
        old_f1 = 2 * old_tp / (2 * old_tp + old_fp + old_fn) if 2 * old_tp + old_fp + old_fn else 0.0
        new_f1 = 2 * new_tp / (2 * new_tp + new_fp + new_fn) if 2 * new_tp + new_fp + new_fn else 0.0
        summary.append({
            "video_group": group, "images": len(members),
            "ground_truth_instances": sum(int(row["ground_truth_instances"]) for row in members),
            "epoch3_tp": old_tp, "epoch3_fp": old_fp, "epoch3_fn": old_fn, "epoch3_F1": old_f1,
            "lower_lr_epoch6_tp": new_tp, "lower_lr_epoch6_fp": new_fp, "lower_lr_epoch6_fn": new_fn, "lower_lr_epoch6_F1": new_f1,
            "lower_lr_epoch6_minus_epoch3_F1": new_f1 - old_f1,
        })
    return sorted(summary, key=lambda row: row["video_group"])


def fixed_rows(plan: dict[str, Any], images: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = plan.get("examples")
    if not isinstance(examples, list) or len(examples) != 9 or int(plan.get("distinct_video_groups", -1)) != 9:
        raise RuntimeError("Existing fixed preview plan is incomplete")
    image_for_path = {str(image["file_name"]): image_id for image_id, image in images.items()}
    data = {int(row["image_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    groups: set[str] = set()
    for item in examples:
        image_id = image_for_path.get(item.get("source_image_path"))
        group = item.get("video_group")
        if image_id is None or not isinstance(group, str) or group in groups:
            raise RuntimeError("Fixed preview plan does not map to nine distinct validation groups")
        groups.add(group)
        selected.append({**data[image_id], "selection": f"fixed_predeclared_{item.get('selection_role')}"})
    return selected


def write_preview_sheet(
    target: Path, selected: list[dict[str, Any]], images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]], old_predictions: dict[int, list[dict[str, Any]]],
    new_predictions: dict[int, list[dict[str, Any]]],
) -> None:
    panels: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    for row in selected:
        image_id = int(row["image_id"])
        with Image.open(PROJECT / str(images[image_id]["file_name"])) as opened:
            image = opened.convert("RGB")
        old = draw_variant(image, annotations.get(image_id, []), old_predictions.get(image_id, []), "no-jitter epoch 3")
        new = draw_variant(image, annotations.get(image_id, []), new_predictions[image_id], "lower-LR epoch 6")
        old.thumbnail((600, 420), Image.Resampling.LANCZOS)
        new.thumbnail((600, 420), Image.Resampling.LANCZOS)
        panels.append((row, old, new))
    margin, label_height, width = 16, 36, 1232
    height = margin + sum(label_height + max(old.height, new.height) + margin for _, old, new in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw, font, y = ImageDraw.Draw(canvas), ImageFont.load_default(), margin
    for row, old, new in panels:
        label = (
            f"{row['selection']} | {row['video_group']} | id={row['image_id']} | GT={row['ground_truth_instances']} | "
            f"epoch3 F1={row['epoch3_F1']:.2f} | lower-LR e6 F1={row['lower_lr_epoch6_F1']:.2f}"
        )
        draw.text((margin, y), label, fill="black", font=font)
        y += label_height
        canvas.paste(old, (margin, y))
        canvas.paste(new, (margin * 2 + 600, y))
        y += max(old.height, new.height) + margin
    canvas.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/evaluations/")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    output_dir = ROOT / args.run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse evaluation directory: {output_dir}")
    if Path(os.environ.get("RF_HOME", "")).resolve() != (PROJECT / ".cache/rfdetr").resolve():
        raise SystemExit("RF_HOME must resolve to this project's .cache/rfdetr directory")

    device = require_mps()
    freeze = frozen_entries()
    frozen = {str(path.relative_to(PROJECT)): verify_frozen_file(path, freeze) for path in (VAL_COCO, CLASS_MAPPING)}
    frozen[str(FREEZE_FILE.relative_to(PROJECT))] = digest(FREEZE_FILE)
    payload = load_json(VAL_COCO, "frozen validation annotations")
    manifest_rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, manifest_rows, "valid")
    if len(images) != IMAGE_COUNT or sum(len(value) for value in annotations.values()) != ANNOTATION_COUNT:
        raise RuntimeError("Expected exactly 1,082 validation images and 289 annotations")
    video_groups = {str(row["source_image_path"]): str(row["video_group"]) for row in manifest_rows}
    if set(video_groups) != {str(image["file_name"]) for image in images.values()}:
        raise RuntimeError("Frozen manifest and COCO image map disagree")

    epoch3, e3_quality, e3_predictions, e3_cache = load_epoch3(frozen, images, annotations)
    lower = load_lower()
    plan = load_json(FIXED_PREVIEWS, "fixed validation preview plan")
    output_dir.mkdir(parents=True)
    started, memory_before = time.perf_counter(), mps_memory()
    write_json(output_dir / "preflight.json", {
        "scope": "Validation only. Reuses verified epoch-3 cache; executes only lower-LR epoch-6 inference.",
        "device_policy": "MPS required and CPU fallback disabled. Neural inference/postprocess stay on MPS; CPU handles image I/O, metric reduction, reports, and previews.",
        "frozen_checks": frozen,
        "epoch3_cache": e3_cache,
        "lower_lr_epoch6": lower,
        "shared_threshold": SHARED_THRESHOLD,
        "fixed_preview_plan": {"path": str(FIXED_PREVIEWS.relative_to(PROJECT)), "sha256": digest(FIXED_PREVIEWS), "examples": 9},
    })

    import rfdetr
    from rfdetr import RFDETRSmall

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("rfdetr is not the pinned project-local source")
    model = RFDETRSmall.from_checkpoint(str(LOWER_CHECKPOINT), device="mps")
    core = model.model.model
    if core is None or model.model.device.type != "mps" or model.model_config.resolution != 512 or model.model_config.num_classes != 1:
        raise RuntimeError("Lower-LR epoch-6 did not load as one-class 512px RF-DETR Small on MPS")
    core = core.to(device).eval()
    if any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError("Lower-LR epoch-6 detector is not wholly MPS-resident")
    evaluated, e6_quality_all = evaluate_detector_variant(
        "lower_lr_epoch6", model, core, images, annotations,
        lambda _image_id, image: PROJECT / str(image["file_name"]), device,
    )
    del model, core
    torch.mps.empty_cache()

    e6_quality = {
        image_id: {
            "tp": int(item["tp"]), "fp": int(item["fp"]), "fn": int(item["fn"]),
            "F1": float(item["F1"]), "predictions": item["predictions"],
        }
        for image_id, item in e6_quality_all.items()
    }
    e3_counts, e6_counts = aggregate(e3_quality, annotations), aggregate(e6_quality, annotations)
    for key in ("F1", "precision", "recall"):
        if abs(e6_counts[key] - float(evaluated["precision_recall_iou50_shared_threshold"][key])) > 1e-9:
            raise RuntimeError("Epoch-6 per-image threshold counts do not reproduce the RF-DETR metric")
    e6_ap = finite(evaluated["training_compatible_metrics"]["AP_50_95"], "epoch-6 AP")
    reproduced = {
        "recorded_AP_50_95": LOWER_AP, "evaluated_AP_50_95": e6_ap,
        "absolute_difference": abs(e6_ap - LOWER_AP), "tolerance": TOLERANCE,
        "passed": abs(e6_ap - LOWER_AP) <= TOLERANCE,
    }
    if not reproduced["passed"]:
        raise RuntimeError("Epoch-6 standalone evaluator failed to reproduce recorded AP within tolerance")

    rows = rows_for(images, annotations, video_groups, e3_quality, e6_quality)
    summaries = group_summary(rows)
    fixed = fixed_rows(plan, images, rows)
    per_image = output_dir / "per_image_shared_threshold_043.csv"
    with per_image.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_group = output_dir / "video_group_shared_threshold_043.csv"
    with per_group.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    previews = output_dir / "comparison_previews"
    previews.mkdir()
    fixed_sheet = previews / "fixed_epoch3_vs_lower_lr_epoch6_shared_threshold_043.png"
    write_preview_sheet(
        fixed_sheet, fixed, images, annotations, e3_predictions,
        {image_id: item["predictions"] for image_id, item in e6_quality.items()},
    )
    write_json(previews / "selection.json", {
        "threshold": SHARED_THRESHOLD,
        "fixed_predeclared_examples": fixed,
        "layout": "left=no-jitter epoch 3; right=lower-LR epoch 6; same raw image in both. Green=frozen GT; red=score >= 0.43 prediction.",
        "sheet": str(fixed_sheet.relative_to(PROJECT)),
    })

    e3_metrics, e6_metrics = epoch3["metrics"], evaluated["training_compatible_metrics"]
    delta = {key: finite(e6_metrics[key], key) - finite(e3_metrics[key], key) for key in METRICS}
    gain = sum(float(item["lower_lr_epoch6_minus_epoch3_F1"]) > 0 for item in summaries)
    regression = sum(float(item["lower_lr_epoch6_minus_epoch3_F1"]) < 0 for item in summaries)
    result = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Validation-only comparison on the 1,082-image frozen raw TrashCan split. Test data, checkpoints, enhancement outputs, original data, and website files were not changed.",
        "device": "mps", "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "cpu_fallback_disabled": True,
        "frozen_checks": frozen,
        "ground_truth_object_counts": {"total": ANNOTATION_COUNT, "COCO_area_buckets": coco_area_counts(annotations)},
        "no_jitter_epoch3_reused": {
            **epoch3, "aggregate_iou50_shared_threshold": e3_counts, "cache_provenance": e3_cache,
        },
        "lower_lr_epoch6": {
            **lower, "reproduction": reproduced,
            "training_compatible_ema_metrics": e6_metrics,
            "precision_recall_iou50_shared_threshold": evaluated["precision_recall_iou50_shared_threshold"],
            "precision_recall_iou50_validation_selected_best_f1": evaluated["precision_recall_iou50_validation_selected_best_f1"],
            "aggregate_iou50_shared_threshold": e6_counts,
            "timing_seconds": evaluated["timing_seconds"],
        },
        "lower_lr_epoch6_minus_no_jitter_epoch3": {
            "COCO_metrics": delta,
            "shared_threshold_counts": {key: e6_counts[key] - e3_counts[key] for key in ("tp", "fp", "fn")},
            "shared_threshold_statistics": {
                key: e6_counts[key] - e3_counts[key]
                for key in ("F1", "precision", "recall", "false_positives_per_negative_image", "fraction_negative_images_with_false_detection")
            },
        },
        "per_image_shared_threshold": str(per_image.relative_to(PROJECT)),
        "per_video_group_shared_threshold": str(per_group.relative_to(PROJECT)),
        "video_group_summary": {
            "groups": len(summaries), "groups_with_F1_gain": gain,
            "groups_with_F1_regression": regression, "groups_unchanged": len(summaries) - gain - regression,
        },
        "previews": {
            "fixed_predeclared": str(fixed_sheet.relative_to(PROJECT)),
            "selection": str((previews / "selection.json").relative_to(PROJECT)),
        },
        "timing_seconds": {
            "epoch6_inference_and_metrics": evaluated["timing_seconds"],
            "total": round(time.perf_counter() - started, 3),
        },
        "memory": {
            "before": memory_before, "after": mps_memory(),
            "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    write_json(output_dir / "evaluation_result.json", result)

    best3 = epoch3["best_f1"]
    best6 = evaluated["precision_recall_iou50_validation_selected_best_f1"]
    areas = coco_area_counts(annotations)
    report = f"""# Lower-LR epoch-6 versus no-jitter epoch-3 RF-DETR validation

This is a validation-only comparison on all {IMAGE_COUNT} frozen raw TrashCan
images and {ANNOTATION_COUNT} annotations. The epoch-3 cache was hash- and
provenance-checked before reuse. The lower-LR epoch-6 EMA checkpoint was
evaluated afresh on MPS.

- No-jitter epoch-3 SHA-256: {E3_SHA}
- Lower-LR epoch-6 SHA-256: {LOWER_SHA}
- Epoch-6 AP reproduction: {reproduced['evaluated_AP_50_95']:.6f} versus recorded
  {reproduced['recorded_AP_50_95']:.6f}; passed={reproduced['passed']}.

| Metric | No-jitter epoch 3 | Lower-LR epoch 6 | Difference |
| --- | ---: | ---: | ---: |
| AP@[.50:.95] | {e3_metrics['AP_50_95']:.4f} | {e6_metrics['AP_50_95']:.4f} | {delta['AP_50_95']:+.4f} |
| AP50 | {e3_metrics['AP50']:.4f} | {e6_metrics['AP50']:.4f} | {delta['AP50']:+.4f} |
| AP75 | {e3_metrics['AP75']:.4f} | {e6_metrics['AP75']:.4f} | {delta['AP75']:+.4f} |
| Small AP / AR | {e3_metrics['AP_small']:.4f} / {e3_metrics['AR_small_500']:.4f} | {e6_metrics['AP_small']:.4f} / {e6_metrics['AR_small_500']:.4f} | {delta['AP_small']:+.4f} / {delta['AR_small_500']:+.4f} |
| Medium AP / AR | {e3_metrics['AP_medium']:.4f} / {e3_metrics['AR_medium_500']:.4f} | {e6_metrics['AP_medium']:.4f} / {e6_metrics['AR_medium_500']:.4f} | {delta['AP_medium']:+.4f} / {delta['AR_medium_500']:+.4f} |
| Large AP / AR | {e3_metrics['AP_large']:.4f} / {e3_metrics['AR_large_500']:.4f} | {e6_metrics['AP_large']:.4f} / {e6_metrics['AR_large_500']:.4f} | {delta['AP_large']:+.4f} / {delta['AR_large_500']:+.4f} |
| F1 @ 0.43 | {e3_counts['F1']:.4f} | {e6_counts['F1']:.4f} | {e6_counts['F1'] - e3_counts['F1']:+.4f} |
| Precision @ 0.43 | {e3_counts['precision']:.4f} | {e6_counts['precision']:.4f} | {e6_counts['precision'] - e3_counts['precision']:+.4f} |
| Recall @ 0.43 | {e3_counts['recall']:.4f} | {e6_counts['recall']:.4f} | {e6_counts['recall'] - e3_counts['recall']:+.4f} |

At score 0.43 and IoU 0.5, TP/FP/FN are
{e3_counts['tp']}/{e3_counts['fp']}/{e3_counts['fn']} for epoch 3 and
{e6_counts['tp']}/{e6_counts['fp']}/{e6_counts['fn']} for lower-LR epoch 6.
Across {e3_counts['negative_images']} negative images, false positives per
negative image are {e3_counts['false_positives_per_negative_image']:.4f}
versus {e6_counts['false_positives_per_negative_image']:.4f}; fractions with
at least one false detection are
{e3_counts['fraction_negative_images_with_false_detection']:.4f} versus
{e6_counts['fraction_negative_images_with_false_detection']:.4f}.

Best-F1 thresholds are separately validation-selected: epoch 3
{best3['best_confidence_threshold']:.2f} with F1 {best3['F1']:.4f}; lower-LR
epoch 6 {best6['best_confidence_threshold']:.2f} with F1 {best6['F1']:.4f}.
They are not used for the shared-threshold comparison.

The COCO buckets contain {areas['small']} small, {areas['medium']} medium,
and {areas['large']} large instances. Across {len(summaries)} video groups,
aggregate F1 improves in {gain}, regresses in {regression}, and is unchanged
in {len(summaries) - gain - regression}. The CSV files retain each image and
video-group result.

![Fixed previews](comparison_previews/fixed_epoch3_vs_lower_lr_epoch6_shared_threshold_043.png)

The fixed panel reuses the existing nine distinct-group examples. Both sides
use the same raw image; green boxes are frozen ground truth and red boxes are
detections at 0.43.
"""
    (output_dir / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

