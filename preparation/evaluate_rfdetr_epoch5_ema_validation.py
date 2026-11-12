#!/usr/bin/env python3
"""Reproduce hash-pinned RF-DETR EMA validation on the frozen validation split.

This is MPS inference plus CPU metric reduction only. It loads the preserved
epoch-5 EMA inference checkpoint, evaluates exactly the frozen 1,082 raw
validation images, and writes a new evaluation directory. It never trains,
changes weights, creates an optimizer, mounts a test split, or changes data.

The implementation intentionally mirrors the pinned training validation path:
PIL RGB decode; 512x512 bilinear, antialiased resize; [0,1] conversion;
checkpoint normalization; the model's unthresholded PostProcess; and RF-DETR's
configured ``vernier`` COCO metric backend. That is necessary before the
result can be compared with the selected checkpoint's recorded EMA AP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only evaluation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f

from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING,
    PROJECT,
    RUNS_ROOT,
    VAL_COCO,
    check_coco_payload,
    digest,
    frozen_entries,
    load_manifest,
    mps_memory,
    require_mps,
    verify_frozen_file,
)


BEST_EPOCH5_EMA = RUNS_ROOT / "rfdetr_raw_v2_e1to5_fixed512_mps_20261001T091117Z/checkpoints/best_validation_ap_50_95_ema.pth"
SCALE_JITTER_OFF_EPOCH3_EMA = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_best_ema.pth"
EVALUATION_TARGETS = {
    "original_epoch5_baseline": {
        "checkpoint": BEST_EPOCH5_EMA,
        "sha256": "9e269f264845c638e65da5e2b77cdc54b19764fd85c2f00b0a3d846866b1f5a2",
        "recorded_ap_50_95": 0.12120693176984787,
        "description": "preserved original raw-image epoch-5 best EMA inference checkpoint",
    },
    "scale_jitter_off_epoch3_best": {
        "checkpoint": SCALE_JITTER_OFF_EPOCH3_EMA,
        "sha256": "6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440",
        "recorded_ap_50_95": 0.17765460908412933,
        "description": "preserved scale_jitter=False epoch-3 best EMA inference checkpoint",
    },
}
REPRODUCTION_TOLERANCE = 0.001
EVALUATIONS_ROOT = RUNS_ROOT / "evaluations"
SHARED_THRESHOLD = 0.43


def native(value: Any) -> Any:
    """Convert tensor and NumPy values to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def finite_scalar(metrics: dict[str, torch.Tensor], key: str) -> float:
    """Read one scalar from RF-DETR's TorchMetrics-compatible result."""
    if key not in metrics:
        raise RuntimeError(f"Training-compatible COCO metric is missing {key!r}")
    value = float(metrics[key].detach().cpu().reshape(-1)[0].item())
    if not math.isfinite(value):
        raise RuntimeError(f"Training-compatible COCO metric {key!r} is non-finite")
    return value


def summarize_training_metric(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Give RF-DETR's vernier metric keys explicit report names."""
    keys = {
        "AP_50_95": "map", "AP50": "map_50", "AP75": "map_75",
        "AP_small": "map_small", "AP_medium": "map_medium", "AP_large": "map_large",
        "AR_1": "mar_1", "AR_10": "mar_10", "AR_500": "mar_500",
        "AR_small_500": "mar_small", "AR_medium_500": "mar_medium", "AR_large_500": "mar_large",
    }
    return {name: finite_scalar(metrics, key) for name, key in keys.items()}


def coco_area_counts(annotations_by_image: dict[int, list[dict[str, Any]]]) -> dict[str, int]:
    """Count frozen validation instances in standard COCO small/medium/large buckets."""
    counts = {"small": 0, "medium": 0, "large": 0}
    for annotations in annotations_by_image.values():
        for annotation in annotations:
            area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
            if area < 32**2:
                counts["small"] += 1
            elif area < 96**2:
                counts["medium"] += 1
            else:
                counts["large"] += 1
    if sum(counts.values()) != 289:
        raise RuntimeError("Frozen validation object-size counts do not sum to the verified 289 annotations")
    return counts


def training_style_f1(matching: dict[int, dict[str, Any]]) -> dict[str, float]:
    """Recreate RF-DETR callback's confidence-sweep F1/precision/recall summary."""
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds

    if not matching:
        raise RuntimeError("No matching data was accumulated for frozen validation")
    ids = sorted(matching)
    per_class = [matching[class_id] for class_id in ids]
    classes_with_gt = [index for index, class_id in enumerate(ids) if matching[class_id]["total_gt"] > 0]
    if classes_with_gt != [ids.index(0)]:
        raise RuntimeError("The frozen one-class validation set did not retain foreground label 0 as its only GT class")
    best = max(sweep_confidence_thresholds(per_class, np.linspace(0, 1, 101), classes_with_gt), key=lambda item: item["macro_f1"])
    return {
        "F1": float(best["macro_f1"]),
        "precision": float(best["macro_precision"]),
        "recall": float(best["macro_recall"]),
        "best_confidence_threshold": float(best["confidence_threshold"]),
    }


def shared_threshold_f1(matching: dict[int, dict[str, Any]]) -> dict[str, float]:
    """Report precision/recall/F1 at the predeclared common threshold only."""
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds

    ids = sorted(matching)
    per_class = [matching[class_id] for class_id in ids]
    classes_with_gt = [index for index, class_id in enumerate(ids) if matching[class_id]["total_gt"] > 0]
    if classes_with_gt != [ids.index(0)]:
        raise RuntimeError("The frozen one-class validation set did not retain foreground label 0 as its only GT class")
    result = sweep_confidence_thresholds(per_class, [SHARED_THRESHOLD], classes_with_gt)[0]
    return {
        "threshold": SHARED_THRESHOLD,
        "F1": float(result["macro_f1"]),
        "precision": float(result["macro_precision"]),
        "recall": float(result["macro_recall"]),
    }


def validation_tensor(source_path: Path, means: Any, stds: Any) -> tuple[torch.Tensor, tuple[int, int]]:
    """Apply the pinned validation preprocessing before moving the image to MPS."""
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        resized = vision_f.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    tensor = vision_f.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
    return vision_f.normalize(tensor, mean=means, std=stds), (height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/evaluations/")
    parser.add_argument("--target", choices=sorted(EVALUATION_TARGETS), default="original_epoch5_baseline", help="Hash-pinned preserved EMA checkpoint to evaluate")
    parser.add_argument("--checkpoint", type=Path, help="Optional redundant path; it must equal the hash-pinned target checkpoint")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    output_dir = EVALUATIONS_ROOT / args.run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse an evaluation directory: {output_dir}")
    target = EVALUATION_TARGETS[args.target]
    checkpoint = Path(target["checkpoint"]).resolve()
    if args.checkpoint is not None and args.checkpoint.resolve() != checkpoint:
        raise SystemExit("--checkpoint must equal the selected hash-pinned target checkpoint")
    if not checkpoint.is_file():
        raise RuntimeError(f"The selected preserved checkpoint is missing: {checkpoint}")
    if digest(checkpoint) != target["sha256"]:
        raise RuntimeError("The selected preserved EMA checkpoint differs from its recorded SHA-256")

    device = require_mps()
    frozen = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, frozen) for path in (VAL_COCO, CLASS_MAPPING)}
    validation_payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    validation_rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations_by_image = check_coco_payload(validation_payload, validation_rows, "valid")
    if len(images) != 1082:
        raise RuntimeError("This evaluator must receive exactly the frozen 1,082-image validation split")

    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    memory_before = mps_memory()
    from rfdetr import RFDETRSmall
    from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    model = RFDETRSmall.from_checkpoint(str(checkpoint), device="mps")
    if model.model.device.type != "mps" or model.model_config.resolution != 512 or model.model_config.num_classes != 1:
        raise RuntimeError("The epoch-5 inference checkpoint did not load as one-class, 512px MPS RF-DETR Small")
    core_model = model.model.model
    if core_model is None:
        raise RuntimeError("The preserved checkpoint did not provide a live RF-DETR model")
    # ``RFDETR.predict()`` normally performs this placement lazily through its
    # device decorator. This evaluator deliberately calls the model directly to
    # reproduce validation preprocessing, so it must perform the same placement
    # explicitly before entering inference mode.
    core_model = core_model.to(device)
    if any(parameter.device.type != "mps" for parameter in core_model.parameters()):
        raise RuntimeError("The preserved RF-DETR module could not be placed fully on MPS")
    core_model.eval()
    metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 500], backend="vernier", sync_on_compute=False)

    coco_predictions: list[dict[str, Any]] = []
    matching: dict[int, dict[str, Any]] = {}
    foreground_prediction_counts: list[int] = []
    background_slot_predictions = 0
    all_slot_predictions = 0
    clipped_foreground_boxes = 0
    discarded_degenerate_foreground_boxes = 0
    allowed_labels = {0, 1}  # 0=visible_plastic_debris; 1=the detector's no-object slot.

    with torch.inference_mode():
        for index, image_id in enumerate(sorted(images), start=1):
            image = images[image_id]
            source_path = PROJECT / str(image["file_name"])
            if not source_path.is_file() or not source_path.is_relative_to(PROJECT / "datasets/trashcan/material_version"):
                raise RuntimeError(f"Validation source is not a frozen raw material_version image: {source_path}")
            input_tensor, (height, width) = validation_tensor(source_path, model.means, model.stds)
            if (height, width) != (int(image["height"]), int(image["width"])):
                raise RuntimeError(f"Frozen validation image dimensions changed: {source_path}")
            batch = input_tensor.unsqueeze(0).to(device)
            raw_outputs = core_model(batch)
            if isinstance(raw_outputs, tuple):
                raw_outputs = {"pred_logits": raw_outputs[1], "pred_boxes": raw_outputs[0]}
            target_sizes = torch.tensor([[height, width]], dtype=torch.int64, device=device)
            result = model.model.postprocess(raw_outputs, target_sizes=target_sizes)[0]
            if model.model.device.type != "mps" or next(core_model.parameters()).device.type != "mps":
                raise RuntimeError("RF-DETR validation left MPS; refusing any CPU model fallback")

            boxes = result["boxes"].detach().cpu().to(dtype=torch.float32)
            scores = result["scores"].detach().cpu().to(dtype=torch.float32)
            labels = result["labels"].detach().cpu().to(dtype=torch.int64)
            if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.numel() or scores.numel() != labels.numel():
                raise RuntimeError(f"Invalid postprocess output for validation image {image_id}")
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not set(labels.tolist()).issubset(allowed_labels):
                raise RuntimeError(f"Invalid labels, boxes, or scores for validation image {image_id}")
            all_slot_predictions += int(labels.numel())
            background_slot_predictions += int((labels == 1).sum().item())

            gt_annotations = annotations_by_image.get(image_id, [])
            if gt_annotations:
                gt_boxes = torch.tensor([[float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][0]) + float(a["bbox"][2]), float(a["bbox"][1]) + float(a["bbox"][3])] for a in gt_annotations], dtype=torch.float32)
            else:
                gt_boxes = torch.empty((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((len(gt_annotations),), dtype=torch.int64)
            prediction_for_metric = {"boxes": boxes, "scores": scores, "labels": labels}
            target_for_metric = {"boxes": gt_boxes, "labels": gt_labels}
            metric.update([prediction_for_metric], [target_for_metric])
            merge_matching_data(matching, build_matching_data([prediction_for_metric], [target_for_metric], iou_threshold=0.5, iou_type="bbox"))

            foreground = labels == 0
            foreground_boxes = boxes[foreground].numpy().copy()
            foreground_scores = scores[foreground].numpy()
            unclipped = foreground_boxes.copy()
            foreground_boxes[:, 0::2] = np.clip(foreground_boxes[:, 0::2], 0.0, float(width))
            foreground_boxes[:, 1::2] = np.clip(foreground_boxes[:, 1::2], 0.0, float(height))
            clipped_foreground_boxes += int(np.any(foreground_boxes != unclipped, axis=1).sum())
            keep = (foreground_boxes[:, 2] > foreground_boxes[:, 0]) & (foreground_boxes[:, 3] > foreground_boxes[:, 1])
            discarded_degenerate_foreground_boxes += int((~keep).sum())
            foreground_boxes, foreground_scores = foreground_boxes[keep], foreground_scores[keep]
            foreground_prediction_counts.append(int(len(foreground_boxes)))
            for box, score in zip(foreground_boxes, foreground_scores, strict=True):
                x1, y1, x2, y2 = map(float, box)
                coco_predictions.append({"image_id": int(image_id), "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
            if index % 50 == 0:
                print(f"evaluated {index}/{len(images)} frozen validation images on MPS", flush=True)

    torch.mps.synchronize()
    training_metrics = summarize_training_metric(metric.compute())
    pr_metrics = training_style_f1(matching)
    shared_pr_metrics = shared_threshold_f1(matching)
    ap_delta = training_metrics["AP_50_95"] - float(target["recorded_ap_50_95"])
    reproduction_passed = abs(ap_delta) <= REPRODUCTION_TOLERANCE
    memory_after = mps_memory()
    result = {
        "status": "passed" if reproduction_passed else "review_required",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen raw TrashCan validation split only; 1,082 images. No training, test data, enhanced images, or dataset/checkpoint modifications.",
        "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "checkpoint": {"path": str(checkpoint.relative_to(PROJECT)), "sha256": digest(checkpoint), "type": str(target["description"]), "target": args.target},
        "validation_path": {
            "preprocessing": "PIL RGB decode -> Resize(512,512, BILINEAR, antialias=True) -> ToImage/ToDtype(scale=True) -> checkpoint Normalize",
            "postprocess": "unthresholded official PostProcess with num_select=300; labels 0 and 1 retained for training-compatible metrics",
            "metrics": "official pinned RF-DETR OnePassCocoMeanAveragePrecision backend=vernier, max_detection_thresholds=[1,10,500]",
            "cpu_work": "Only metric aggregation and JSON serialization use CPU; all neural-network inference and postprocess execute on MPS.",
        },
        "frozen_checks": frozen_checks,
        "validation_images": len(images),
        "ground_truth_object_counts": {"total": 289, "COCO_area_buckets": coco_area_counts(annotations_by_image)},
        "training_compatible_ema_metrics": training_metrics,
        "precision_recall_iou50_shared_threshold": shared_pr_metrics,
        "precision_recall_iou50_validation_selected_best_f1": pr_metrics,
        "reproduction_check": {
            "recorded_AP_50_95": float(target["recorded_ap_50_95"]),
            "evaluated_AP_50_95": training_metrics["AP_50_95"],
            "absolute_difference": abs(ap_delta), "tolerance": REPRODUCTION_TOLERANCE, "passed": reproduction_passed,
            "interpretation": "Per-size results are suitable for comparison only when passed is true; otherwise resolve the mismatch before interpreting them.",
        },
        "predictions": {
            "all_postprocess_slots": all_slot_predictions, "background_label_1_slots": background_slot_predictions,
            "foreground_label_0_slots": all_slot_predictions - background_slot_predictions,
            "saved_foreground_COCO_records": len(coco_predictions), "foreground_per_image_mean": float(np.mean(foreground_prediction_counts)),
            "foreground_per_image_max": int(max(foreground_prediction_counts, default=0)),
            "clipped_foreground_boxes_for_JSON": clipped_foreground_boxes,
            "discarded_degenerate_foreground_boxes_for_JSON": discarded_degenerate_foreground_boxes,
            "background_note": "Label 1 is RF-DETR's no-object logit slot, not a second TrashCan category. It is retained for the training-compatible metric and omitted from the one-class COCO prediction export.",
        },
        "timing_seconds": round(time.perf_counter() - started, 3),
        "memory": {"before": memory_before, "after": memory_after, "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    (output_dir / "foreground_coco_predictions.json").write_text(json.dumps(coco_predictions) + "\n", encoding="utf-8")
    (output_dir / "evaluation_result.json").write_text(json.dumps(result, indent=2, default=native) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=native))


if __name__ == "__main__":
    main()
