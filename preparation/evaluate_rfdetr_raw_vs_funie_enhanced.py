#!/usr/bin/env python3
"""Compare raw and inference-time FUnIE-enhanced detector inputs on validation.

This MPS-only, validation-only runner compares the same preserved raw-trained
RF-DETR Small checkpoint on frozen raw TrashCan validation images and on
separately written, geometry-preserving FUnIE-GAN enhancement derivatives.
It never trains, accesses test data, rewrites frozen annotations/images, or
alters either model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only evaluation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_rfdetr_epoch5_ema_validation import (
    SHARED_THRESHOLD,
    coco_area_counts,
    shared_threshold_f1,
    summarize_training_metric,
    training_style_f1,
    validation_tensor,
)
from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING,
    FREEZE_FILE,
    PROJECT,
    RFDETR_SOURCE,
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


DETECTOR_CHECKPOINT = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_best_ema.pth"
DETECTOR_SHA256 = "6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440"
DETECTOR_RECORDED_AP = 0.17765460908412933
FUNIE_RUN = PROJECT / "training_runs/funie/funie_batchnorm_frozen_e2to4_mps_20260930"
FUNIE_CHECKPOINT = FUNIE_RUN / "checkpoints/best_validation_psnr.pt"
FUNIE_LATEST = FUNIE_RUN / "checkpoints/latest.pt"
FUNIE_CANDIDATE = PROJECT / "model_candidates/funie_gan_frozen_batchnorm_epoch20_mps.json"
FUNIE_SHA256 = "c9259bc5257ef28c58c09090ee165c8c990eaab49f455bdc86c147363c5b77d5"
FUNIE_SOURCE = PROJECT / "third_party/FUnIE-GAN"
FUNIE_PYTORCH = FUNIE_SOURCE / "PyTorch"
EVALUATIONS_ROOT = RUNS_ROOT / "evaluations"
FUNIE_CANVAS = 256
PREVIEW_COUNT_PER_KIND = 4
NEGATIVE_PREVIEW_COUNT = 3


def git_revision(path: Path) -> str:
    """Read the pinned source revision without changing repository state."""
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def finite(value: float | torch.Tensor, name: str) -> float:
    result = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is non-finite")
    return result


def assert_frozen_detector() -> dict[str, Any]:
    """Require the preserved no-jitter epoch-3 inference checkpoint exactly."""
    if not DETECTOR_CHECKPOINT.is_file() or digest(DETECTOR_CHECKPOINT) != DETECTOR_SHA256:
        raise RuntimeError("The preserved no-jitter epoch-3 RF-DETR EMA checkpoint hash does not match")
    return {"path": str(DETECTOR_CHECKPOINT.relative_to(PROJECT)), "sha256": DETECTOR_SHA256, "recorded_AP_50_95": DETECTOR_RECORDED_AP, "description": "raw-trained RF-DETR Small, scale_jitter=False epoch-3 EMA"}


def assert_frozen_funie() -> dict[str, Any]:
    """Require the candidate epoch-20 frozen-BatchNorm FUnIE-GAN checkpoint."""
    if not all(path.is_file() for path in (FUNIE_CHECKPOINT, FUNIE_LATEST, FUNIE_CANDIDATE, FREEZE_FILE)):
        raise RuntimeError("The preserved frozen-BatchNorm FUnIE candidate artifacts are incomplete")
    candidate = json.loads(FUNIE_CANDIDATE.read_text(encoding="utf-8"))
    checkpoint = candidate.get("checkpoint", {})
    if checkpoint.get("sha256") != FUNIE_SHA256 or digest(FUNIE_CHECKPOINT) != FUNIE_SHA256 or digest(FUNIE_LATEST) != FUNIE_SHA256:
        raise RuntimeError("The preserved FUnIE epoch-20 best/latest checkpoint hash does not match")
    payload = torch.load(FUNIE_CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "epoch": 19,
        "format": "aqua_vision_funie_batchnorm_frozen_v1",
        "batchnorm_statistics_frozen": True,
        "frozen_manifest_sha256": digest(FREEZE_FILE),
        "source_revision": git_revision(FUNIE_SOURCE),
    }
    mismatches = {key: {"expected": expected, "actual": payload.get(key)} for key, expected in required.items() if payload.get(key) != expected}
    if mismatches or not isinstance(payload.get("generator"), dict) or not isinstance(payload.get("batchnorm_epoch1_generator_state"), dict):
        raise RuntimeError(f"The FUnIE candidate checkpoint provenance is invalid: {mismatches}")
    return {
        "path": str(FUNIE_CHECKPOINT.relative_to(PROJECT)),
        "sha256": FUNIE_SHA256,
        "latest_path": str(FUNIE_LATEST.relative_to(PROJECT)),
        "epoch": int(payload["epoch"]) + 1,
        "global_step": int(payload["global_step"]),
        "format": payload["format"],
        "batchnorm_statistics_frozen": payload["batchnorm_statistics_frozen"],
        "source_revision": payload["source_revision"],
        "frozen_manifest_sha256": payload["frozen_manifest_sha256"],
    }


def batchnorm_integrity(generator: torch.nn.Module, recorded: dict[str, Any]) -> dict[str, Any]:
    """Verify epoch-1 BatchNorm buffers remain the frozen checkpoint values."""
    state = generator.state_dict()
    checked = 0
    for name, buffers in recorded.items():
        if not isinstance(buffers, dict):
            raise RuntimeError(f"Malformed saved BatchNorm buffers for {name}")
        module = dict(generator.named_modules()).get(name)
        if not isinstance(module, torch.nn.BatchNorm2d) or module.training:
            raise RuntimeError(f"FUnIE generator BatchNorm {name} is not frozen in eval mode")
        if module.weight is None or module.bias is None or not module.weight.requires_grad or not module.bias.requires_grad:
            raise RuntimeError(f"FUnIE generator BatchNorm {name} does not retain trainable affine parameters")
        for field, expected in buffers.items():
            actual = state.get(f"{name}.{field}")
            if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor) or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                raise RuntimeError(f"FUnIE generator BatchNorm buffer changed: {name}.{field}")
        checked += 1
    return {"layers_verified": checked, "all_eval_mode": True, "affine_parameters_recorded_trainable": True, "running_buffers_match_epoch1_snapshot": True}


def load_funie_generator(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only the preserved generator and make its inference MPS-resident."""
    payload = torch.load(FUNIE_CHECKPOINT, map_location="cpu", weights_only=False)
    sys.path.insert(0, str(FUNIE_PYTORCH))
    from nets.funiegan import GeneratorFunieGAN

    generator = GeneratorFunieGAN().to(device)
    generator.load_state_dict(payload["generator"], strict=True)
    generator.eval()
    if any(parameter.device.type != "mps" for parameter in generator.parameters()):
        raise RuntimeError("FUnIE generator is not wholly MPS-resident")
    return generator, batchnorm_integrity(generator, payload["batchnorm_epoch1_generator_state"])


def aspect_preserving_funie_canvas(image: Image.Image) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resize longest side to 256, reflect-pad to a square, and normalize [-1,1]."""
    source = image.convert("RGB")
    width, height = source.size
    scale = FUNIE_CANVAS / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = source.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    pad_left = (FUNIE_CANVAS - resized_width) // 2
    pad_right = FUNIE_CANVAS - resized_width - pad_left
    pad_top = (FUNIE_CANVAS - resized_height) // 2
    pad_bottom = FUNIE_CANVAS - resized_height - pad_top
    array = np.asarray(resized, dtype=np.uint8)
    pad_mode = "reflect" if min(resized_width, resized_height) > 1 else "edge"
    canvas = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=pad_mode)
    if canvas.shape != (FUNIE_CANVAS, FUNIE_CANVAS, 3):
        raise RuntimeError(f"FUnIE canvas has unexpected geometry: {canvas.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).to(dtype=torch.float32).div_(127.5).sub_(1.0)
    metadata = {
        "source_dimensions": {"width": width, "height": height},
        "funie_canvas": {"width": FUNIE_CANVAS, "height": FUNIE_CANVAS},
        "resized_dimensions": {"width": resized_width, "height": resized_height},
        "uniform_resize_scale": scale,
        "padding_pixels": {"left": pad_left, "right": pad_right, "top": pad_top, "bottom": pad_bottom, "mode": pad_mode},
        "preprocessing": "RGB -> aspect-preserving bicubic resize with longest side 256 -> reflection pad to 256x256 -> [-1,1] tensor",
    }
    return tensor, metadata


def restore_enhanced_geometry(output: torch.Tensor, metadata: dict[str, Any]) -> Image.Image:
    """Remove canvas padding and resize the enhanced crop exactly back to source size."""
    if list(output.shape) != [3, FUNIE_CANVAS, FUNIE_CANVAS] or not torch.isfinite(output).all():
        raise RuntimeError("FUnIE output is non-finite or not a 3x256x256 tensor")
    pixels = ((output.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(dtype=torch.uint8)
    array = pixels.permute(1, 2, 0).numpy()
    padding = metadata["padding_pixels"]
    resized = metadata["resized_dimensions"]
    crop = array[int(padding["top"]):int(padding["top"]) + int(resized["height"]), int(padding["left"]):int(padding["left"]) + int(resized["width"])]
    if crop.shape[:2] != (int(resized["height"]), int(resized["width"])):
        raise RuntimeError("FUnIE output crop does not match the letterboxed input geometry")
    source = metadata["source_dimensions"]
    return Image.fromarray(crop, mode="RGB").resize((int(source["width"]), int(source["height"])), Image.Resampling.BICUBIC)


def valid_original_bbox(annotation: dict[str, Any], width: int, height: int) -> None:
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    if not (box_width > 0 and box_height > 0 and x >= 0 and y >= 0 and x + box_width <= width and y + box_height <= height):
        raise RuntimeError(f"Frozen annotation is invalid in the original coordinate space: {annotation}")


def enhance_validation_images(
    generator: torch.nn.Module,
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Create one same-size enhanced PNG derivative for each frozen val image."""
    derivatives: dict[int, Path] = {}
    manifest: list[dict[str, Any]] = []
    enhanced_root = output_dir / "enhanced_validation_images"
    enhanced_root.mkdir(parents=True)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, image_id in enumerate(sorted(images), start=1):
            image = images[image_id]
            source_path = PROJECT / str(image["file_name"])
            if not source_path.is_file() or not source_path.is_relative_to(PROJECT / "datasets/trashcan/material_version"):
                raise RuntimeError(f"Source is not a frozen raw TrashCan material_version validation image: {source_path}")
            with Image.open(source_path) as opened:
                raw = opened.convert("RGB")
            width, height = raw.size
            if (width, height) != (int(image["width"]), int(image["height"])):
                raise RuntimeError(f"Frozen image dimensions changed: {source_path}")
            for annotation in annotations.get(image_id, []):
                valid_original_bbox(annotation, width, height)
            funie_input, geometry = aspect_preserving_funie_canvas(raw)
            enhanced = restore_enhanced_geometry(generator(funie_input.unsqueeze(0).to(device))[0], geometry)
            if enhanced.size != (width, height):
                raise RuntimeError("Enhanced derivative did not retain exact original dimensions")
            target = enhanced_root / f"{image_id:06d}_{Path(str(image['file_name'])).stem}.png"
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite an enhanced derivative: {target}")
            enhanced.save(target)
            with Image.open(target) as decoded:
                if decoded.convert("RGB").size != (width, height):
                    raise RuntimeError(f"Enhanced derivative did not decode with original dimensions: {target}")
            derivatives[image_id] = target
            manifest.append({
                "image_id": image_id,
                "source_image_path": str(image["file_name"]),
                "enhanced_derivative_path": str(target.relative_to(PROJECT)),
                "source_dimensions": {"width": width, "height": height},
                "enhanced_dimensions": {"width": enhanced.width, "height": enhanced.height},
                "bbox_coordinate_alignment": "unchanged_original_pixel_coordinates; annotations are not copied, transformed, or rewritten",
                "bbox_count": len(annotations.get(image_id, [])),
                "enhancement_geometry": geometry,
                "enhanced_sha256": digest(target),
            })
            if index % 50 == 0:
                print(f"enhanced {index}/{len(images)} frozen validation images on MPS", flush=True)
    torch.mps.synchronize()
    duration = time.perf_counter() - started
    if len(derivatives) != len(images):
        raise RuntimeError("Enhancement derivative count does not match the frozen validation split")
    for item in manifest:
        if item["source_dimensions"] != item["enhanced_dimensions"]:
            raise RuntimeError("At least one enhanced derivative changed the image coordinate system")
    write_json(output_dir / "enhancement_derivative_manifest.json", {"status": "passed", "images": manifest, "summary": {"count": len(manifest), "duration_seconds": round(duration, 3), "geometry": "All enhanced PNG derivatives decode at the original width/height. COCO boxes remain in original pixel coordinates and are reused unchanged."}})
    return derivatives, manifest


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def shared_threshold_quality(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-image TP/FP/FN at the shared detector threshold and IoU 0.5."""
    predictions = [
        {"box": [float(value) for value in box], "score": float(score)}
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True)
        if label == 0 and score >= SHARED_THRESHOLD
    ]
    predictions.sort(key=lambda prediction: prediction["score"], reverse=True)
    ground_truth = [[float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][0]) + float(a["bbox"][2]), float(a["bbox"][1]) + float(a["bbox"][3])] for a in annotations]
    matched: set[int] = set()
    true_positive = 0
    for prediction in predictions:
        choices = [(iou(prediction["box"], truth), index) for index, truth in enumerate(ground_truth) if index not in matched]
        if choices:
            score, best_index = max(choices)
            if score >= 0.5:
                matched.add(best_index)
                true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative, "F1": (2 * true_positive / denominator) if denominator else 0.0, "predictions": predictions}


def evaluate_detector_variant(
    name: str,
    model: Any,
    core: torch.nn.Module,
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    path_for_image: Callable[[int, dict[str, Any]], Path],
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Run the exact training-compatible detector preprocessing/metric path."""
    from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 500], backend="vernier", sync_on_compute=False)
    matching: dict[int, dict[str, Any]] = {}
    per_image: dict[int, dict[str, Any]] = {}
    all_slots = background_slots = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, image_id in enumerate(sorted(images), start=1):
            image = images[image_id]
            source_path = path_for_image(image_id, image)
            if not source_path.is_file():
                raise RuntimeError(f"Missing {name} detector input: {source_path}")
            input_tensor, (height, width) = validation_tensor(source_path, model.means, model.stds)
            if (height, width) != (int(image["height"]), int(image["width"])):
                raise RuntimeError(f"{name} detector input changed geometry for image {image_id}")
            outputs = core(input_tensor.unsqueeze(0).to(device))
            if isinstance(outputs, tuple):
                outputs = {"pred_logits": outputs[1], "pred_boxes": outputs[0]}
            result = model.model.postprocess(outputs, target_sizes=torch.tensor([[height, width]], dtype=torch.int64, device=device))[0]
            if model.model.device.type != "mps" or any(parameter.device.type != "mps" for parameter in core.parameters()):
                raise RuntimeError(f"{name} detector inference left MPS; CPU fallback is forbidden")
            boxes = result["boxes"].detach().cpu().to(dtype=torch.float32)
            scores = result["scores"].detach().cpu().to(dtype=torch.float32)
            labels = result["labels"].detach().cpu().to(dtype=torch.int64)
            if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.numel() or scores.numel() != labels.numel() or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not set(labels.tolist()).issubset({0, 1}):
                raise RuntimeError(f"Invalid {name} detector postprocess output for image {image_id}")
            image_annotations = annotations.get(image_id, [])
            gt_boxes = torch.tensor([[float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][0]) + float(a["bbox"][2]), float(a["bbox"][1]) + float(a["bbox"][3])] for a in image_annotations], dtype=torch.float32) if image_annotations else torch.empty((0, 4), dtype=torch.float32)
            target = {"boxes": gt_boxes, "labels": torch.zeros((len(image_annotations),), dtype=torch.int64)}
            prediction = {"boxes": boxes, "scores": scores, "labels": labels}
            metric.update([prediction], [target])
            merge_matching_data(matching, build_matching_data([prediction], [target], iou_threshold=0.5, iou_type="bbox"))
            per_image[image_id] = shared_threshold_quality(boxes, scores, labels, image_annotations)
            all_slots += int(labels.numel())
            background_slots += int((labels == 1).sum().item())
            if index % 50 == 0:
                print(f"evaluated {name} {index}/{len(images)} frozen validation images on MPS", flush=True)
    torch.mps.synchronize()
    return {
        "variant": name,
        "validation_images": len(images),
        "training_compatible_metrics": summarize_training_metric(metric.compute()),
        "precision_recall_iou50_shared_threshold": shared_threshold_f1(matching),
        "precision_recall_iou50_validation_selected_best_f1": training_style_f1(matching),
        "prediction_slots": {"all": all_slots, "background_label_1": background_slots, "foreground_label_0": all_slots - background_slots},
        "detector_preprocessing": "PIL RGB -> Resize(512,512,BILINEAR,antialias=True) -> scale [0,1] -> checkpoint Normalize -> unthresholded official PostProcess -> RF-DETR vernier metric backend",
        "timing_seconds": round(time.perf_counter() - started, 3),
        "memory_after": mps_memory(),
    }, per_image


def draw_variant(image: Image.Image, annotations: list[dict[str, Any]], predictions: list[dict[str, Any]], label: str) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for annotation in annotations:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        draw.rectangle((x, y, x + width, y + height), outline="lime", width=3)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 2, y1 + 2), f"{prediction['score']:.2f}", fill="red", font=font)
    draw.rectangle((0, 0, panel.width, 18), fill="black")
    draw.text((3, 3), f"{label}; green=GT, red=prediction @ {SHARED_THRESHOLD:.2f}", fill="white", font=font)
    return panel


def preview_selection(
    rows: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
    video_groups: dict[str, str],
) -> list[dict[str, Any]]:
    """Preselect largest shared-threshold gains/losses and three distinct negatives."""
    positives = [row for row in rows if row["ground_truth_instances"] > 0]
    positives.sort(key=lambda row: (row["enhanced_F1_minus_raw"], row["enhanced_tp"] - row["raw_tp"], -row["image_id"]), reverse=True)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for row in positives[:PREVIEW_COUNT_PER_KIND]:
        selected.append({**row, "selection": "largest_shared_threshold_improvement"})
        used.add(row["image_id"])
    for row in sorted((row for row in positives if row["image_id"] not in used), key=lambda row: (row["enhanced_F1_minus_raw"], row["enhanced_tp"] - row["raw_tp"], row["image_id"]))[:PREVIEW_COUNT_PER_KIND]:
        selected.append({**row, "selection": "largest_shared_threshold_regression"})
        used.add(row["image_id"])
    negative_groups: set[str] = set()
    for row in sorted((row for row in rows if row["ground_truth_instances"] == 0), key=lambda row: (video_groups[row["source_image_path"]], row["source_image_path"])):
        group = video_groups[row["source_image_path"]]
        if group in negative_groups:
            continue
        selected.append({**row, "selection": "negative_image"})
        negative_groups.add(group)
        if len(negative_groups) == NEGATIVE_PREVIEW_COUNT:
            break
    if len(selected) < 2 * PREVIEW_COUNT_PER_KIND + NEGATIVE_PREVIEW_COUNT:
        raise RuntimeError("Could not select the required positive improvement/regression and negative preview groups")
    return selected


def create_paired_previews(
    output_dir: Path,
    selected: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    derivatives: dict[int, Path],
    raw_quality: dict[int, dict[str, Any]],
    enhanced_quality: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Write raw/enhanced detector panels after metrics determine selection."""
    preview_dir = output_dir / "paired_previews"
    preview_dir.mkdir(parents=True)
    font = ImageFont.load_default()
    item_panels: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    for row in selected:
        image_id = int(row["image_id"])
        image = images[image_id]
        with Image.open(PROJECT / str(image["file_name"])) as opened:
            raw_image = opened.convert("RGB")
        with Image.open(derivatives[image_id]) as opened:
            enhanced_image = opened.convert("RGB")
        raw_panel = draw_variant(raw_image, annotations.get(image_id, []), raw_quality[image_id]["predictions"], "raw")
        enhanced_panel = draw_variant(enhanced_image, annotations.get(image_id, []), enhanced_quality[image_id]["predictions"], "enhanced")
        raw_panel.thumbnail((600, 420), Image.Resampling.LANCZOS)
        enhanced_panel.thumbnail((600, 420), Image.Resampling.LANCZOS)
        item_panels.append((row, raw_panel, enhanced_panel))
    margin, label_height, width = 16, 34, 1232
    height = margin + sum(label_height + max(raw.height, enhanced.height) + margin for _, raw, enhanced in item_panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    for row, raw_panel, enhanced_panel in item_panels:
        title = f"{row['selection']} | id={row['image_id']} | GT={row['ground_truth_instances']} | raw F1={row['raw_F1']:.2f} | enhanced F1={row['enhanced_F1']:.2f}"
        draw.text((margin, y), title, fill="black", font=font)
        y += label_height
        canvas.paste(raw_panel, (margin, y))
        canvas.paste(enhanced_panel, (margin * 2 + 600, y))
        y += max(raw_panel.height, enhanced_panel.height) + margin
    sheet = preview_dir / "raw_vs_enhanced_shared_threshold_043.png"
    canvas.save(sheet)
    write_json(preview_dir / "selection.json", {"threshold": SHARED_THRESHOLD, "selection": selected, "sheet": str(sheet.relative_to(PROJECT)), "layout": "left=raw detector input; right=FUnIE-enhanced detector input; green=unchanged frozen GT; red=detector predictions"})
    return {"sheet": str(sheet.relative_to(PROJECT)), "sheet_sha256": digest(sheet), "selection": str((preview_dir / "selection.json").relative_to(PROJECT)), "items": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/evaluations/")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    output_dir = EVALUATIONS_ROOT / args.run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse an evaluation directory: {output_dir}")
    expected_cache = (PROJECT / ".cache/rfdetr").resolve()
    if Path(os.environ.get("RF_HOME", "")).resolve() != expected_cache:
        raise SystemExit("RF_HOME must resolve to this project's .cache/rfdetr directory")
    detector_record = assert_frozen_detector()
    funie_record = assert_frozen_funie()
    device = require_mps()
    frozen = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, frozen) for path in (VAL_COCO, CLASS_MAPPING)}
    frozen_checks[str(FREEZE_FILE.relative_to(PROJECT))] = digest(FREEZE_FILE)
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    validation_rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, validation_rows, "valid")
    if len(images) != 1082 or sum(len(value) for value in annotations.values()) != 289:
        raise RuntimeError("This comparison requires exactly 1,082 frozen validation images and 289 annotations")
    video_groups = {row["source_image_path"]: row["video_group"] for row in validation_rows}
    if set(str(image["file_name"]) for image in images.values()) != set(video_groups):
        raise RuntimeError("Frozen validation manifest/image map mismatch")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    memory_before = mps_memory()
    write_json(output_dir / "preflight.json", {
        "scope": "Validation-only inference-time enhancement comparison. The detector was trained on raw images; enhanced images are never used for training here.",
        "detector": detector_record,
        "enhancer": funie_record,
        "frozen_checks": frozen_checks,
        "validation_images": len(images),
        "annotations": sum(len(value) for value in annotations.values()),
        "geometry_policy": "Source image -> aspect-preserving bicubic resize (longest side 256) -> reflect-pad 256x256 -> FUnIE -> remove identical padding -> bicubic resize to exact original width/height. Frozen COCO annotations are reused unchanged in original pixel coordinates.",
        "detector_policy": "Identical raw/enhanced preprocessing and official unthresholded RF-DETR postprocess/metric settings. Neural inference remains on MPS; CPU is used only for image encoding and metric aggregation.",
    })

    generator, batchnorm_audit = load_funie_generator(device)
    derivatives, derivative_manifest = enhance_validation_images(generator, images, annotations, output_dir, device)
    del generator
    torch.mps.empty_cache()

    import rfdetr
    from rfdetr import RFDETRSmall

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The detector package is not the pinned project-local RF-DETR source")
    detector = RFDETRSmall.from_checkpoint(str(DETECTOR_CHECKPOINT), device="mps")
    core = detector.model.model
    if core is None or detector.model_config.resolution != 512 or detector.model_config.num_classes != 1:
        raise RuntimeError("The preserved detector did not load as one-class, 512px RF-DETR Small")
    core.to(device).eval()
    if any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError("The detector is not wholly MPS-resident")
    raw_result, raw_quality = evaluate_detector_variant("raw", detector, core, images, annotations, lambda _id, image: PROJECT / str(image["file_name"]), device)
    raw_ap = finite(raw_result["training_compatible_metrics"]["AP_50_95"], "raw AP")
    raw_reproduction = {"recorded_AP_50_95": DETECTOR_RECORDED_AP, "evaluated_raw_AP_50_95": raw_ap, "absolute_difference": abs(raw_ap - DETECTOR_RECORDED_AP), "tolerance": 0.001, "passed": abs(raw_ap - DETECTOR_RECORDED_AP) <= 0.001}
    if not raw_reproduction["passed"]:
        raise RuntimeError("Raw branch did not reproduce the preserved detector's standalone epoch-3 validation AP")
    enhanced_result, enhanced_quality = evaluate_detector_variant("funie_enhanced", detector, core, images, annotations, lambda image_id, _image: derivatives[image_id], device)
    del detector, core
    torch.mps.empty_cache()

    per_image_rows: list[dict[str, Any]] = []
    for image_id in sorted(images):
        image = images[image_id]
        raw, enhanced = raw_quality[image_id], enhanced_quality[image_id]
        per_image_rows.append({
            "image_id": image_id,
            "source_image_path": str(image["file_name"]),
            "video_group": video_groups[str(image["file_name"])],
            "ground_truth_instances": len(annotations.get(image_id, [])),
            "raw_tp": raw["tp"], "raw_fp": raw["fp"], "raw_fn": raw["fn"], "raw_F1": raw["F1"],
            "enhanced_tp": enhanced["tp"], "enhanced_fp": enhanced["fp"], "enhanced_fn": enhanced["fn"], "enhanced_F1": enhanced["F1"],
            "enhanced_F1_minus_raw": enhanced["F1"] - raw["F1"],
        })
    with (output_dir / "per_image_shared_threshold_043.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    selected = preview_selection(per_image_rows, images, video_groups)
    preview = create_paired_previews(output_dir, selected, images, annotations, derivatives, raw_quality, enhanced_quality)
    metric_deltas = {key: finite(enhanced_result["training_compatible_metrics"][key], key) - finite(raw_result["training_compatible_metrics"][key], key) for key in raw_result["training_compatible_metrics"]}
    shared_deltas = {key: finite(enhanced_result["precision_recall_iou50_shared_threshold"][key], key) - finite(raw_result["precision_recall_iou50_shared_threshold"][key], key) for key in ("F1", "precision", "recall")}
    result = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Inference-time FUnIE enhancement evaluated with a detector trained on raw inputs. This result does not test training the detector on enhanced images; a negative result cannot establish that enhanced-image training would not help.",
        "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "cpu_fallback_disabled": True,
        "detector": detector_record,
        "enhancer": {**funie_record, "batchnorm_integrity": batchnorm_audit},
        "source_revisions": {"rfdetr": git_revision(RFDETR_SOURCE), "funie_gan": git_revision(FUNIE_SOURCE)},
        "frozen_checks": frozen_checks,
        "ground_truth_object_counts": {"total": 289, "COCO_area_buckets": coco_area_counts(annotations)},
        "raw_reproduction": raw_reproduction,
        "raw": raw_result,
        "funie_enhanced": enhanced_result,
        "enhanced_minus_raw": {"training_compatible_metrics": metric_deltas, "precision_recall_iou50_shared_threshold": shared_deltas},
        "enhancement_derivatives": {"directory": str((output_dir / "enhanced_validation_images").relative_to(PROJECT)), "manifest": str((output_dir / "enhancement_derivative_manifest.json").relative_to(PROJECT)), "count": len(derivative_manifest), "geometry_alignment": "verified exact original width/height for every derivative; frozen annotations remain unchanged in the original coordinate system"},
        "per_image_shared_threshold_043": str((output_dir / "per_image_shared_threshold_043.csv").relative_to(PROJECT)),
        "paired_previews": preview,
        "timing_seconds": {"total": round(time.perf_counter() - started, 3), "raw_detector": raw_result["timing_seconds"], "enhancement_plus_write": json.loads((output_dir / "enhancement_derivative_manifest.json").read_text(encoding="utf-8"))["summary"]["duration_seconds"], "enhanced_detector": enhanced_result["timing_seconds"]},
        "memory": {"before": memory_before, "after": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(output_dir / "raw_vs_funie_enhanced_result.json", result)
    raw_metrics = raw_result["training_compatible_metrics"]
    enhanced_metrics = enhanced_result["training_compatible_metrics"]
    raw_shared = raw_result["precision_recall_iou50_shared_threshold"]
    enhanced_shared = enhanced_result["precision_recall_iou50_shared_threshold"]
    report = f"""# Raw versus FUnIE-enhanced validation comparison

This is an **inference-time enhancement** comparison using the same
raw-trained no-jitter RF-DETR Small epoch-3 EMA detector for both branches.
It uses the frozen 1,082-image raw TrashCan validation split only; no test
data or training was used. A negative result would not establish that training
or fine-tuning the detector on enhanced images cannot help.

- Detector checkpoint SHA-256: `{DETECTOR_SHA256}`
- FUnIE-GAN epoch-20 frozen-BatchNorm checkpoint SHA-256: `{FUNIE_SHA256}`
- Raw-branch AP reproduction: {raw_reproduction['evaluated_raw_AP_50_95']:.6f}
  versus {raw_reproduction['recorded_AP_50_95']:.6f}; passed={raw_reproduction['passed']}.
- Enhanced derivatives: {len(derivative_manifest)} PNGs, each verified at its
  original width and height. Frozen COCO annotations remained in the original
  pixel coordinate system.

| Metric | Raw | FUnIE-enhanced | Enhanced − raw |
| --- | ---: | ---: | ---: |
| AP@[.50:.95] | {raw_metrics['AP_50_95']:.4f} | {enhanced_metrics['AP_50_95']:.4f} | {metric_deltas['AP_50_95']:+.4f} |
| AP50 | {raw_metrics['AP50']:.4f} | {enhanced_metrics['AP50']:.4f} | {metric_deltas['AP50']:+.4f} |
| AP75 | {raw_metrics['AP75']:.4f} | {enhanced_metrics['AP75']:.4f} | {metric_deltas['AP75']:+.4f} |
| Small AP / AR | {raw_metrics['AP_small']:.4f} / {raw_metrics['AR_small_500']:.4f} | {enhanced_metrics['AP_small']:.4f} / {enhanced_metrics['AR_small_500']:.4f} | {metric_deltas['AP_small']:+.4f} / {metric_deltas['AR_small_500']:+.4f} |
| Medium AP / AR | {raw_metrics['AP_medium']:.4f} / {raw_metrics['AR_medium_500']:.4f} | {enhanced_metrics['AP_medium']:.4f} / {enhanced_metrics['AR_medium_500']:.4f} | {metric_deltas['AP_medium']:+.4f} / {metric_deltas['AR_medium_500']:+.4f} |
| Large AP / AR | {raw_metrics['AP_large']:.4f} / {raw_metrics['AR_large_500']:.4f} | {enhanced_metrics['AP_large']:.4f} / {enhanced_metrics['AR_large_500']:.4f} | {metric_deltas['AP_large']:+.4f} / {metric_deltas['AR_large_500']:+.4f} |
| F1 @ 0.43 | {raw_shared['F1']:.4f} | {enhanced_shared['F1']:.4f} | {shared_deltas['F1']:+.4f} |
| Precision @ 0.43 | {raw_shared['precision']:.4f} | {enhanced_shared['precision']:.4f} | {shared_deltas['precision']:+.4f} |
| Recall @ 0.43 | {raw_shared['recall']:.4f} | {enhanced_shared['recall']:.4f} | {shared_deltas['recall']:+.4f} |

Object counts: small {coco_area_counts(annotations)['small']}, medium
{coco_area_counts(annotations)['medium']}, large {coco_area_counts(annotations)['large']}.

![Paired previews]({preview['sheet']})

The preview panel is selected after shared-threshold evaluation for four
largest positive-image gains, four largest positive-image regressions, and
three distinct negative-image groups. It is qualitative; the full validation
metrics above use every frozen validation image.

Timing: enhancement/write {result['timing_seconds']['enhancement_plus_write']:.1f}s;
raw detector {result['timing_seconds']['raw_detector']:.1f}s; enhanced detector
{result['timing_seconds']['enhanced_detector']:.1f}s; total
{result['timing_seconds']['total']:.1f}s.
"""
    (output_dir / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, default=native))


if __name__ == "__main__":
    main()
