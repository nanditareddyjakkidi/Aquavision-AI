#!/usr/bin/env python3
"""Validation-only analysis of lower-LR RF-DETR epoch-6 missed objects.

The prior comparison saved threshold-0.43 counts but not unthresholded boxes
and scores. This MPS-only script runs one fresh inference pass for the
hash-pinned lower-LR epoch-6 EMA checkpoint, exports all foreground
postprocess predictions, and verifies the 0.43 TP/FP/FN counts from the prior
validation report. It never trains, accesses test data, creates an optimizer,
or writes frozen datasets/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only analysis.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_rfdetr_epoch5_ema_validation import validation_tensor
from evaluate_rfdetr_raw_vs_funie_enhanced import write_json
from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING, FREEZE_FILE, PROJECT, RFDETR_SOURCE, RUNS_ROOT, VAL_COCO,
    check_coco_payload, digest, frozen_entries, load_manifest, mps_memory,
    require_mps, verify_frozen_file,
)

EVALUATIONS_ROOT = RUNS_ROOT / "evaluations"
LOWER_RUN = RUNS_ROOT / "rfdetr_no_jitter_epoch3_lower_lr_e4to8_mps_20261003T054821Z"
CHECKPOINT = LOWER_RUN / "training_output/checkpoint_best_ema.pth"
CHECKPOINT_SHA256 = "7670935cb63af62f4a9471dc8081b62dd34d41cf8ff1fdca442bbe9f79a0e753"
PREVIOUS_EVALUATION = EVALUATIONS_ROOT / "rfdetr_lower_lr_epoch6_vs_epoch3_validation_mps_20261004T064115Z/evaluation_result.json"

IMAGE_COUNT = 1082
ANNOTATION_COUNT = 289
SHARED_THRESHOLD = 0.43
IOU_MATCH = 0.50
IOU_USEFUL_OVERLAP = 0.10
THRESHOLDS = (0.10, 0.20, 0.30, 0.43, 0.47)


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def check_checkpoint() -> dict[str, str]:
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Missing lower-LR epoch-6 checkpoint: {CHECKPOINT}")
    actual = digest(CHECKPOINT)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(f"Lower-LR epoch-6 checkpoint hash mismatch: expected {CHECKPOINT_SHA256}, got {actual}")
    return {"path": str(CHECKPOINT.relative_to(PROJECT)), "sha256": actual}


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def size_bucket(annotation: dict[str, Any]) -> str:
    area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def ground_truth(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "annotation_id": int(annotation["id"]),
            "box": [
                float(annotation["bbox"][0]), float(annotation["bbox"][1]),
                float(annotation["bbox"][0]) + float(annotation["bbox"][2]),
                float(annotation["bbox"][1]) + float(annotation["bbox"][3]),
            ],
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "size": size_bucket(annotation),
        }
        for annotation in annotations
    ]


def match_at_threshold(predictions: list[dict[str, Any]], truths: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Score-descending, one-to-one greedy matching at IoU >= 0.50.

    Each foreground prediction with score >= threshold is visited once in
    descending score order. It can match only one currently unmatched ground
    truth, choosing that truth by largest IoU. A ground truth can match only
    one prediction. This mirrors the shared-threshold accounting used in the
    prior validation comparison.
    """
    selected = sorted((item for item in predictions if float(item["score"]) >= threshold), key=lambda item: float(item["score"]), reverse=True)
    matched_truths: set[int] = set()
    assignment: dict[int, int] = {}
    for prediction in selected:
        choices = [
            (iou(prediction["box"], truth["box"]), truth_index)
            for truth_index, truth in enumerate(truths)
            if truth_index not in matched_truths
        ]
        if choices:
            overlap, truth_index = max(choices)
            if overlap >= IOU_MATCH:
                assignment[int(prediction["prediction_id"])] = truth_index
                matched_truths.add(truth_index)
    matched_prediction_ids = set(assignment)
    return {
        "selected": selected,
        "assignment": assignment,
        "matched_truths": matched_truths,
        "matched_prediction_ids": matched_prediction_ids,
        "tp": len(assignment),
        "fp": len(selected) - len(assignment),
        "fn": len(truths) - len(assignment),
    }


def pairs_for(
    predictions: list[dict[str, Any]], truth_indices: set[int], truths: list[dict[str, Any]],
    lower_iou: float, upper_iou: float | None, score_descending: bool,
) -> list[tuple[float, float, int, int]]:
    """Construct candidate pairs for deterministic one-to-one secondary matching."""
    pairs: list[tuple[float, float, int, int]] = []
    for prediction in predictions:
        for truth_index in truth_indices:
            overlap = iou(prediction["box"], truths[truth_index]["box"])
            if overlap >= lower_iou and (upper_iou is None or overlap < upper_iou):
                pairs.append((float(prediction["score"]), overlap, int(prediction["prediction_id"]), truth_index))
    if score_descending:
        return sorted(pairs, key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return sorted(pairs, key=lambda item: (-item[1], -item[0], item[2], item[3]))


def assign_pairs(pairs: list[tuple[float, float, int, int]]) -> dict[int, tuple[int, float, float]]:
    """Use each prediction and truth at most once; return truth -> (prediction, score, IoU)."""
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    assigned: dict[int, tuple[int, float, float]] = {}
    for score, overlap, prediction_id, truth_index in pairs:
        if prediction_id in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_id)
        used_truths.add(truth_index)
        assigned[truth_index] = (prediction_id, score, overlap)
    return assigned


def analyze_image(
    image_id: int, predictions: list[dict[str, Any]], truths: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Partition base false negatives while retaining duplicate/assignment modifiers.

    Primary classes are exclusive:
    1. below_threshold_match: an available score < 0.43 prediction reaches
       IoU >= 0.50, paired one-to-one with a base false negative.
    2. overlap_below_iou_requirement: remaining false negatives paired
       one-to-one with an otherwise-unassigned prediction at 0.10 <= IoU < 0.50.
    3. no_eligible_useful_overlap: remaining false negatives have no eligible
       unassigned prediction at IoU >= 0.10.

    High-score predictions already assigned to a different object never become
    eligible for a second object. High-score duplicates remain eligible for the
    localization pairing but are flagged. Both effects are reported separately.
    """
    base = match_at_threshold(predictions, truths, SHARED_THRESHOLD)
    fn_truths = set(range(len(truths))) - set(base["matched_truths"])
    by_id = {int(item["prediction_id"]): item for item in predictions}
    high_status: dict[int, str] = {}
    for prediction in base["selected"]:
        prediction_id = int(prediction["prediction_id"])
        if prediction_id in base["matched_prediction_ids"]:
            high_status[prediction_id] = "assigned_to_matched_truth"
            continue
        max_iou = max((iou(prediction["box"], truth["box"]) for truth in truths), default=0.0)
        if max_iou >= IOU_MATCH:
            high_status[prediction_id] = "duplicate_high_score_prediction"
        elif max_iou >= IOU_USEFUL_OVERLAP:
            high_status[prediction_id] = "high_score_localization_prediction"
        else:
            high_status[prediction_id] = "high_score_no_useful_overlap"

    low_predictions = [item for item in predictions if float(item["score"]) < SHARED_THRESHOLD]
    low_pairs = pairs_for(low_predictions, fn_truths, truths, IOU_MATCH, None, score_descending=True)
    below_threshold = assign_pairs(low_pairs)
    remaining = fn_truths - set(below_threshold)

    base_assigned = set(base["matched_prediction_ids"])
    below_used = {prediction_id for prediction_id, _score, _overlap in below_threshold.values()}
    localization_pool = [
        item for item in predictions
        if int(item["prediction_id"]) not in base_assigned and int(item["prediction_id"]) not in below_used
    ]
    localization_pairs = pairs_for(localization_pool, remaining, truths, IOU_USEFUL_OVERLAP, IOU_MATCH, score_descending=False)
    localization = assign_pairs(localization_pairs)
    remaining_after_localization = remaining - set(localization)

    high_assigned_other: dict[int, list[int]] = defaultdict(list)
    high_duplicates: dict[int, list[int]] = defaultdict(list)
    for truth_index in fn_truths:
        for prediction_id, assigned_truth in base["assignment"].items():
            if assigned_truth != truth_index and iou(by_id[prediction_id]["box"], truths[truth_index]["box"]) >= IOU_USEFUL_OVERLAP:
                high_assigned_other[truth_index].append(prediction_id)
        for prediction_id, status in high_status.items():
            if status == "duplicate_high_score_prediction" and iou(by_id[prediction_id]["box"], truths[truth_index]["box"]) >= IOU_USEFUL_OVERLAP:
                high_duplicates[truth_index].append(prediction_id)

    rows: list[dict[str, Any]] = []
    for truth_index in sorted(fn_truths):
        truth = truths[truth_index]
        if truth_index in below_threshold:
            prediction_id, score, overlap = below_threshold[truth_index]
            category = "below_threshold_match_iou_ge_050"
            source = "below_threshold_prediction"
        elif truth_index in localization:
            prediction_id, score, overlap = localization[truth_index]
            category = "overlap_iou_010_to_lt_050"
            source = high_status.get(prediction_id, "below_threshold_prediction")
        else:
            prediction_id, score, overlap = None, None, None
            category = "no_eligible_useful_overlap_iou_ge_010"
            source = None
        rows.append({
            "image_id": image_id,
            "annotation_id": truth["annotation_id"],
            "gt_bbox_xywh": truth["bbox_xywh"],
            "object_size": truth["size"],
            "primary_category": category,
            "paired_prediction_id": prediction_id,
            "paired_prediction_score": score,
            "paired_prediction_iou": overlap,
            "paired_prediction_source": source,
            "has_high_prediction_assigned_to_other_object_overlap": bool(high_assigned_other.get(truth_index)),
            "assigned_to_other_prediction_ids": high_assigned_other.get(truth_index, []),
            "has_high_duplicate_prediction_overlap": bool(high_duplicates.get(truth_index)),
            "duplicate_prediction_ids": high_duplicates.get(truth_index, []),
        })
    audit = Counter(high_status.values())
    audit["high_score_foreground_predictions"] = len(base["selected"])
    audit["high_score_true_positive_assignments"] = base["tp"]
    audit["high_score_false_positive_predictions"] = base["fp"]
    return rows, dict(audit)


def threshold_summary(
    image_predictions: dict[int, list[dict[str, Any]]], image_truths: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for image_id in sorted(image_truths):
            matched = match_at_threshold(image_predictions[image_id], image_truths[image_id], threshold)
            tp += int(matched["tp"])
            fp += int(matched["fp"])
            fn += int(matched["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result.append({
            "threshold": threshold, "iou_threshold": IOU_MATCH,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "F1": f1,
            "false_positives_per_validation_image": fp / IMAGE_COUNT,
        })
    return result


def select_contact_rows(rows: list[dict[str, Any]], limit_per_category: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    category_order = (
        "below_threshold_match_iou_ge_050",
        "overlap_iou_010_to_lt_050",
        "no_eligible_useful_overlap_iou_ge_010",
    )
    for category in category_order:
        candidates = [row for row in rows if row["primary_category"] == category]
        if category == "below_threshold_match_iou_ge_050":
            candidates.sort(key=lambda row: (-(row["paired_prediction_score"] or 0.0), -(row["paired_prediction_iou"] or 0.0), row["image_id"]))
        elif category == "overlap_iou_010_to_lt_050":
            candidates.sort(key=lambda row: (-(row["paired_prediction_iou"] or 0.0), -(row["paired_prediction_score"] or 0.0), row["image_id"]))
        else:
            candidates.sort(key=lambda row: (row["object_size"], row["image_id"]))
        chosen = 0
        for row in candidates:
            if row["video_group"] in used_groups:
                continue
            selected.append(row)
            used_groups.add(row["video_group"])
            chosen += 1
            if chosen == limit_per_category:
                break
    return selected


def contact_sheet(
    target: Path, rows: list[dict[str, Any]], images: dict[int, dict[str, Any]],
    image_truths: dict[int, list[dict[str, Any]]], image_predictions: dict[int, list[dict[str, Any]]],
) -> None:
    """Draw one panel per selected miss, with all GT, selected GT, and paired prediction."""
    panels: list[tuple[dict[str, Any], Image.Image]] = []
    for row in rows:
        image_id = int(row["image_id"])
        with Image.open(PROJECT / str(images[image_id]["file_name"])) as opened:
            panel = opened.convert("RGB").copy()
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        selected_truth = next(item for item in image_truths[image_id] if int(item["annotation_id"]) == int(row["annotation_id"]))
        for truth in image_truths[image_id]:
            draw.rectangle(tuple(truth["box"]), outline="lime", width=2)
        draw.rectangle(tuple(selected_truth["box"]), outline="yellow", width=4)
        if row["paired_prediction_id"] is not None:
            prediction = next(item for item in image_predictions[image_id] if int(item["prediction_id"]) == int(row["paired_prediction_id"]))
            draw.rectangle(tuple(prediction["box"]), outline="orange", width=3)
            draw.text((prediction["box"][0] + 2, prediction["box"][1] + 2), f"{prediction['score']:.2f}, IoU {row['paired_prediction_iou']:.2f}", fill="orange", font=font)
        panel.thumbnail((560, 360), Image.Resampling.LANCZOS)
        panels.append((row, panel))
    columns, margin, title_height, cell_width = 2, 14, 42, 580
    rows_needed = math.ceil(len(panels) / columns)
    cell_heights = [
        max(panel.height for _row, panel in panels[index:index + columns]) if panels[index:index + columns] else 0
        for index in range(0, len(panels), columns)
    ]
    height = margin + sum(title_height + value + margin for value in cell_heights)
    canvas = Image.new("RGB", (columns * cell_width + (columns + 1) * margin, height), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    y, cursor = margin, 0
    for row_height in cell_heights:
        for column in range(columns):
            if cursor >= len(panels):
                break
            row, panel = panels[cursor]
            x = margin + column * (cell_width + margin)
            title = f"{row['primary_category']} | {row['video_group']} | id={row['image_id']} | {row['object_size']}"
            draw.text((x, y), title, fill="black", font=font)
            canvas.paste(panel, (x, y + title_height))
            cursor += 1
        y += title_height + row_height + margin
    canvas.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/evaluations/")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    output_dir = EVALUATIONS_ROOT / args.run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse analysis directory: {output_dir}")
    if Path(os.environ.get("RF_HOME", "")).resolve() != (PROJECT / ".cache/rfdetr").resolve():
        raise SystemExit("RF_HOME must resolve to this project's .cache/rfdetr directory")

    device = require_mps()
    checkpoint = check_checkpoint()
    freeze = frozen_entries()
    frozen = {str(path.relative_to(PROJECT)): verify_frozen_file(path, freeze) for path in (VAL_COCO, CLASS_MAPPING)}
    frozen[str(FREEZE_FILE.relative_to(PROJECT))] = digest(FREEZE_FILE)
    previous = load_json(PREVIOUS_EVALUATION, "prior lower-LR epoch-6 comparison")
    expected = previous.get("lower_lr_epoch6", {})
    if expected.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256 or previous.get("frozen_checks") != frozen:
        raise RuntimeError("The prior lower-LR evaluation has different checkpoint or frozen-data provenance")
    expected_counts = expected.get("aggregate_iou50_shared_threshold", {})
    if (int(expected_counts.get("tp", -1)), int(expected_counts.get("fp", -1)), int(expected_counts.get("fn", -1))) != (93, 81, 196):
        raise RuntimeError("The prior lower-LR evaluation does not contain the expected 0.43 counts")

    payload = load_json(VAL_COCO, "frozen validation annotation file")
    manifest_rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, manifest_rows, "valid")
    if len(images) != IMAGE_COUNT or sum(len(value) for value in annotations.values()) != ANNOTATION_COUNT:
        raise RuntimeError("Expected exactly the 1,082-image, 289-annotation frozen validation split")
    video_groups = {str(row["source_image_path"]): str(row["video_group"]) for row in manifest_rows}
    if set(video_groups) != {str(image["file_name"]) for image in images.values()}:
        raise RuntimeError("Frozen manifest and COCO image metadata disagree")

    output_dir.mkdir(parents=True)
    started, memory_before = time.perf_counter(), mps_memory()
    write_json(output_dir / "preflight.json", {
        "scope": "Validation-only lower-LR epoch-6 false-negative analysis. One fresh MPS inference pass is necessary because the prior result retained threshold-0.43 counts but not all unthresholded boxes and scores.",
        "checkpoint": checkpoint,
        "frozen_checks": frozen,
        "prior_threshold_043_counts": expected_counts,
        "matching_policy": {
            "base": "Foreground predictions with score >= 0.43 are visited descending score. Each matches at most one unmatched GT at maximum IoU when IoU >= 0.50.",
            "below_threshold": "For base false negatives, score < 0.43 foreground predictions with IoU >= 0.50 are paired one-to-one by score descending, then IoU descending.",
            "localization": "Remaining false negatives are paired one-to-one with otherwise-unassigned predictions at 0.10 <= IoU < 0.50, by IoU descending then score descending.",
            "none": "No eligible otherwise-unassigned prediction overlaps at IoU >= 0.10. High-score predictions already assigned to another GT remain unavailable and are separately flagged.",
        },
    })

    import rfdetr
    from rfdetr import RFDETRSmall

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("rfdetr is not loaded from the pinned project-local source")
    model = RFDETRSmall.from_checkpoint(str(CHECKPOINT), device="mps")
    core = model.model.model
    if core is None or model.model.device.type != "mps" or model.model_config.resolution != 512 or model.model_config.num_classes != 1:
        raise RuntimeError("Checkpoint did not load as one-class, 512px RF-DETR Small on MPS")
    core = core.to(device).eval()
    if any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError("Detector parameters are not wholly MPS-resident")
    declared_cap = getattr(model.model.postprocess, "num_select", None)

    image_predictions: dict[int, list[dict[str, Any]]] = {}
    image_truths: dict[int, list[dict[str, Any]]] = {}
    foreground_export: list[dict[str, Any]] = []
    all_slots = background_slots = degenerate_foreground = 0
    max_slots = 0
    scores: list[float] = []
    with torch.inference_mode():
        for index, image_id in enumerate(sorted(images), start=1):
            image = images[image_id]
            path = PROJECT / str(image["file_name"])
            if not path.is_file() or not path.is_relative_to(PROJECT / "datasets/trashcan/material_version"):
                raise RuntimeError(f"Validation input is not a frozen raw material_version image: {path}")
            tensor, (height, width) = validation_tensor(path, model.means, model.stds)
            if (height, width) != (int(image["height"]), int(image["width"])):
                raise RuntimeError(f"Frozen image dimensions changed: {path}")
            outputs = core(tensor.unsqueeze(0).to(device))
            if isinstance(outputs, tuple):
                outputs = {"pred_logits": outputs[1], "pred_boxes": outputs[0]}
            result = model.model.postprocess(
                outputs, target_sizes=torch.tensor([[height, width]], dtype=torch.int64, device=device)
            )[0]
            if model.model.device.type != "mps" or any(parameter.device.type != "mps" for parameter in core.parameters()):
                raise RuntimeError("Inference left MPS; CPU fallback is forbidden")
            boxes = result["boxes"].detach().cpu().to(dtype=torch.float32)
            pred_scores = result["scores"].detach().cpu().to(dtype=torch.float32)
            labels = result["labels"].detach().cpu().to(dtype=torch.int64)
            if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != pred_scores.numel() or pred_scores.numel() != labels.numel():
                raise RuntimeError(f"Invalid postprocess output for image {image_id}")
            if not torch.isfinite(boxes).all() or not torch.isfinite(pred_scores).all() or not set(labels.tolist()).issubset({0, 1}):
                raise RuntimeError(f"Invalid boxes, scores, or labels for image {image_id}")
            all_slots += int(labels.numel())
            max_slots = max(max_slots, int(labels.numel()))
            background_slots += int((labels == 1).sum().item())
            current: list[dict[str, Any]] = []
            for prediction_id, (box, score, label) in enumerate(zip(boxes.tolist(), pred_scores.tolist(), labels.tolist(), strict=True)):
                if label != 0:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                degenerate = not (x2 > x1 and y2 > y1)
                degenerate_foreground += int(degenerate)
                item = {"prediction_id": prediction_id, "box": [x1, y1, x2, y2], "score": float(score), "degenerate": degenerate}
                current.append(item)
                scores.append(float(score))
                foreground_export.append({
                    "image_id": image_id, "prediction_id": prediction_id, "category_id": 1,
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1], "score": float(score), "degenerate": degenerate,
                })
            image_predictions[image_id] = current
            image_truths[image_id] = ground_truth(annotations.get(image_id, []))
            if index % 50 == 0:
                print(f"exported {index}/{IMAGE_COUNT} frozen validation images on MPS", flush=True)
    torch.mps.synchronize()
    del model, core
    torch.mps.empty_cache()

    threshold_rows = threshold_summary(image_predictions, image_truths)
    at_shared = next(row for row in threshold_rows if row["threshold"] == SHARED_THRESHOLD)
    if (at_shared["tp"], at_shared["fp"], at_shared["fn"]) != (93, 81, 196):
        raise RuntimeError(f"Fresh threshold-0.43 counts do not reproduce prior validation: {at_shared}")
    if abs(at_shared["F1"] - float(expected_counts["F1"])) > 1e-9:
        raise RuntimeError("Fresh threshold-0.43 F1 does not reproduce prior validation")

    miss_rows: list[dict[str, Any]] = []
    assignment_audit: Counter[str] = Counter()
    for image_id in sorted(images):
        rows, audit = analyze_image(image_id, image_predictions[image_id], image_truths[image_id])
        assignment_audit.update(audit)
        for row in rows:
            row["source_image_path"] = str(images[image_id]["file_name"])
            row["video_group"] = video_groups[row["source_image_path"]]
            miss_rows.append(row)
    if len(miss_rows) != 196:
        raise RuntimeError(f"False-negative analysis did not classify 196 misses: found {len(miss_rows)}")

    category_counts = Counter(row["primary_category"] for row in miss_rows)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in miss_rows:
        by_size[str(row["object_size"])]["total_false_negatives"] += 1
        by_size[str(row["object_size"])][str(row["primary_category"])] += 1
        by_size[str(row["object_size"])]["assigned_to_other_object_overlap"] += int(row["has_high_prediction_assigned_to_other_object_overlap"])
        by_size[str(row["object_size"])]["duplicate_prediction_overlap"] += int(row["has_high_duplicate_prediction_overlap"])
        by_group[str(row["video_group"])]["total_false_negatives"] += 1
        by_group[str(row["video_group"])][str(row["primary_category"])] += 1
        by_group[str(row["video_group"])]["assigned_to_other_object_overlap"] += int(row["has_high_prediction_assigned_to_other_object_overlap"])
        by_group[str(row["video_group"])]["duplicate_prediction_overlap"] += int(row["has_high_duplicate_prediction_overlap"])

    misses_csv = output_dir / "false_negative_analysis_iou50_threshold043.csv"
    with misses_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(miss_rows[0]))
        writer.writeheader()
        writer.writerows(miss_rows)
    size_rows = [{"object_size": size, **dict(counter)} for size, counter in sorted(by_size.items())]
    size_csv = output_dir / "misses_by_object_size.csv"
    with size_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["object_size", "total_false_negatives", "below_threshold_match_iou_ge_050", "overlap_iou_010_to_lt_050", "no_eligible_useful_overlap_iou_ge_010", "assigned_to_other_object_overlap", "duplicate_prediction_overlap"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(size_rows)
    group_rows = [{"video_group": group, **dict(counter)} for group, counter in sorted(by_group.items())]
    group_csv = output_dir / "misses_by_video_group.csv"
    with group_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["video_group", "total_false_negatives", "below_threshold_match_iou_ge_050", "overlap_iou_010_to_lt_050", "no_eligible_useful_overlap_iou_ge_010", "assigned_to_other_object_overlap", "duplicate_prediction_overlap"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(group_rows)
    threshold_csv = output_dir / "threshold_operating_points_iou50.csv"
    with threshold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    prediction_json = output_dir / "foreground_predictions_all_scores.json"
    prediction_json.write_text(json.dumps(foreground_export) + "\n", encoding="utf-8")

    selected = select_contact_rows(miss_rows)
    preview_dir = output_dir / "representative_contact_sheet"
    preview_dir.mkdir()
    sheet = preview_dir / "miss_categories_distinct_video_groups.png"
    contact_sheet(sheet, selected, images, image_truths, image_predictions)
    write_json(preview_dir / "selection.json", {
        "selection_policy": "Up to two examples per primary category, with no repeated video group across the sheet. Selected after analysis; it is illustrative rather than a validation metric.",
        "items": selected,
        "layout": "lime=all frozen GT, yellow=selected missed GT, orange=paired below-threshold/localization prediction when one exists.",
        "sheet": str(sheet.relative_to(PROJECT)),
    })

    result = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MPS-only validation analysis on frozen raw TrashCan validation images. The fresh inference pass exists solely because prior saved artifacts omitted unthresholded boxes and scores. No training, test access, or source/checkpoint modification occurred.",
        "checkpoint": checkpoint,
        "device": "mps", "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "cpu_fallback_disabled": True,
        "frozen_checks": frozen,
        "ground_truth_objects": ANNOTATION_COUNT,
        "prediction_export": {
            "path": str(prediction_json.relative_to(PROJECT)),
            "sha256": digest(prediction_json),
            "saved_foreground_predictions": len(foreground_export),
            "degenerate_foreground_predictions": degenerate_foreground,
            "minimum_foreground_score": min(scores),
            "maximum_foreground_score": max(scores),
            "postprocess_detection_cap_declared": declared_cap,
            "maximum_total_postprocess_slots_observed": max_slots,
            "total_postprocess_slots": all_slots,
            "background_label_1_slots_not_exported": background_slots,
            "note": "The cap applies to all postprocess slots. The export preserves only foreground label 0 predictions, including scores below 0.43.",
        },
        "threshold_operating_points_iou50": threshold_rows,
        "threshold_043_reproduction": {
            "prior": {"tp": 93, "fp": 81, "fn": 196, "F1": float(expected_counts["F1"])},
            "fresh": at_shared,
            "passed": True,
        },
        "false_negative_primary_classification": {
            "total": len(miss_rows),
            "definitions": {
                "below_threshold_match_iou_ge_050": "Score < 0.43, IoU >= 0.50, and one-to-one paired to a base false negative.",
                "overlap_iou_010_to_lt_050": "After the below-threshold pairing, one-to-one paired to an otherwise-unassigned prediction with 0.10 <= IoU < 0.50.",
                "no_eligible_useful_overlap_iou_ge_010": "No otherwise-unassigned prediction remains with IoU >= 0.10 after the preceding matchings.",
            },
            "one_to_one_matching": "Base threshold matching is score-descending and uses each GT/prediction at most once. Below-threshold recovery pairs sort by score then IoU; localization pairs sort by IoU then score.",
            "counts": dict(category_counts),
            "high_score_prediction_assignment_audit": dict(assignment_audit),
            "assigned_to_other_object_overlap_note": "Nonexclusive modifier: a high-score prediction was already assigned to a different GT and overlaps this false negative at IoU >= 0.10, so it was unavailable for a second match.",
            "duplicate_prediction_overlap_note": "Nonexclusive modifier: an unmatched high-score prediction already has IoU >= 0.50 to another GT and overlaps this false negative at IoU >= 0.10.",
        },
        "breakdowns": {
            "by_size": size_rows,
            "by_video_group": group_rows,
            "per_false_negative": str(misses_csv.relative_to(PROJECT)),
            "by_size_csv": str(size_csv.relative_to(PROJECT)),
            "by_video_group_csv": str(group_csv.relative_to(PROJECT)),
        },
        "contact_sheet": {"path": str(sheet.relative_to(PROJECT)), "sha256": digest(sheet), "selection": str((preview_dir / "selection.json").relative_to(PROJECT))},
        "threshold_table_csv": str(threshold_csv.relative_to(PROJECT)),
        "timing_seconds": round(time.perf_counter() - started, 3),
        "memory": {"before": memory_before, "after": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(output_dir / "analysis_result.json", result)

    dominant = max(category_counts, key=category_counts.get)
    report = f"""# Lower-LR epoch-6 missed-object analysis

This validation-only analysis re-ran the preserved epoch-6 EMA checkpoint on
the frozen 1,082-image raw validation split because the prior comparison kept
only threshold-0.43 per-image counts, not the unthresholded prediction boxes
and scores needed for miss attribution.

The checkpoint hash is {CHECKPOINT_SHA256}. The fresh 0.43, IoU-0.50 counts
reproduced the prior validation result exactly: 93 TP, 81 FP, and 196 FN.

## False-negative classification

Primary classes are mutually exclusive and sum to 196:

- below-threshold match: score below 0.43 and IoU at least 0.50, one-to-one
  paired to a base false negative.
- localization overlap: after that pairing, an otherwise-unassigned prediction
  at IoU 0.10 through less than 0.50, one-to-one paired.
- no eligible useful overlap: no otherwise-unassigned prediction remains at
  IoU at least 0.10.

The high-score assignment audit and the false-negative CSV separately identify
prediction duplicates and predictions already assigned to another object. Those
two modifiers are nonexclusive and do not change a false negative's primary
class.

| Primary class | Count |
| --- | ---: |
| Below-threshold match, IoU at least 0.50 | {category_counts['below_threshold_match_iou_ge_050']} |
| Overlap, IoU 0.10 to less than 0.50 | {category_counts['overlap_iou_010_to_lt_050']} |
| No eligible useful overlap | {category_counts['no_eligible_useful_overlap_iou_ge_010']} |

The saved foreground prediction export contains {len(foreground_export)}
unthresholded label-0 predictions. Minimum saved score:
{min(scores):.8f}; postprocess detection cap declared:
{declared_cap}; maximum total slots observed per image: {max_slots}. Degenerate
foreground boxes retained in the export and counted as non-matching predictions:
{degenerate_foreground}.

## Threshold operating points

These threshold values are precision/recall operating points at IoU 0.50.
They do not alter or improve AP, which is threshold-integrated.

| Threshold | Precision | Recall | F1 | FP per validation image |
| --- | ---: | ---: | ---: | ---: |
"""
    for row in threshold_rows:
        report += f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} | {row['false_positives_per_validation_image']:.4f} |\n"
    report += f"""
The dominant primary error class is {dominant}. The next justified experiment
should address that class while holding the frozen split, raw inputs,
checkpoint initialization, seed, and evaluation protocol fixed. Do not infer
that a threshold change affects AP.

![Representative misses](representative_contact_sheet/miss_categories_distinct_video_groups.png)
"""
    (output_dir / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
