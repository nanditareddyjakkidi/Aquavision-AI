#!/usr/bin/env python3
"""Evaluate the resize-only control for the raw-versus-FUnIE validation study.

This MPS-only, validation-only runner recreates the previously evaluated
FUnIE branch's geometry pipeline exactly, but sends the 256x256 FUnIE canvas
straight to its geometry-restoration step instead of running FUnIE-GAN. It
creates new same-size PNG derivatives, evaluates them using the same pinned
raw-trained RF-DETR checkpoint and COCO settings, and reuses only verified
raw/FUnIE metrics from the completed prior comparison. It never trains,
accesses a test split, writes a frozen input, or changes a model checkpoint.
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

import torch
from PIL import Image

from evaluate_rfdetr_raw_vs_funie_enhanced import (
    DETECTOR_RECORDED_AP,
    DETECTOR_SHA256,
    EVALUATIONS_ROOT,
    FUNIE_SHA256,
    aspect_preserving_funie_canvas,
    assert_frozen_detector,
    assert_frozen_funie,
    evaluate_detector_variant,
    finite,
    restore_enhanced_geometry,
    valid_original_bbox,
    write_json,
)
from evaluate_rfdetr_epoch5_ema_validation import coco_area_counts
from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING,
    FREEZE_FILE,
    PROJECT,
    RFDETR_SOURCE,
    VAL_COCO,
    check_coco_payload,
    digest,
    frozen_entries,
    load_manifest,
    mps_memory,
    require_mps,
    verify_frozen_file,
)


PREVIOUS_EVALUATION_ID = "rfdetr_raw_vs_funie_enhanced_val_mps_20261002T163013Z"
PREVIOUS_EVALUATION_DIR = EVALUATIONS_ROOT / PREVIOUS_EVALUATION_ID
PREVIOUS_RESULT = PREVIOUS_EVALUATION_DIR / "raw_vs_funie_enhanced_result.json"
PREVIOUS_DERIVATIVE_MANIFEST = PREVIOUS_EVALUATION_DIR / "enhancement_derivative_manifest.json"
EXPECTED_VALIDATION_IMAGES = 1082
EXPECTED_ANNOTATIONS = 289
REPRODUCTION_TOLERANCE = 0.001


def read_required_metrics(result: dict[str, Any], name: str) -> None:
    """Require the metric shapes needed for a three-way comparison."""
    required_metrics = {
        "AP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR_small_500", "AR_medium_500", "AR_large_500",
    }
    required_shared = {"threshold", "F1", "precision", "recall"}
    metrics = result.get("training_compatible_metrics")
    shared = result.get("precision_recall_iou50_shared_threshold")
    if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
        raise RuntimeError(f"Reusable {name} metrics are incomplete")
    if not isinstance(shared, dict) or not required_shared.issubset(shared):
        raise RuntimeError(f"Reusable {name} shared-threshold metrics are incomplete")
    if int(result.get("validation_images", -1)) != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(f"Reusable {name} results did not use the frozen 1,082-image validation split")
    for key in required_metrics:
        finite(metrics[key], f"reusable {name} {key}")
    if float(shared["threshold"]) != 0.43:
        raise RuntimeError(f"Reusable {name} results did not use the shared threshold 0.43")
    for key in ("F1", "precision", "recall"):
        finite(shared[key], f"reusable {name} {key}")


def load_verified_previous_results(frozen_checks: dict[str, str]) -> dict[str, Any]:
    """Reuse only the prior comparison with matching model and frozen-data provenance."""
    if not PREVIOUS_RESULT.is_file() or not PREVIOUS_DERIVATIVE_MANIFEST.is_file():
        raise RuntimeError("The completed raw-versus-FUnIE validation artifacts are missing")
    previous = json.loads(PREVIOUS_RESULT.read_text(encoding="utf-8"))
    if previous.get("status") != "passed":
        raise RuntimeError("The completed raw-versus-FUnIE validation result was not successful")
    if previous.get("detector", {}).get("sha256") != DETECTOR_SHA256:
        raise RuntimeError("The reusable result did not use the preserved no-jitter epoch-3 detector")
    if previous.get("enhancer", {}).get("sha256") != FUNIE_SHA256:
        raise RuntimeError("The reusable result did not use the preserved epoch-20 FUnIE candidate")
    if previous.get("frozen_checks") != frozen_checks:
        raise RuntimeError("The reusable result's frozen dataset checks do not match this evaluation")
    if previous.get("enhancement_derivatives", {}).get("count") != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError("The reusable FUnIE result did not create exactly 1,082 derivatives")
    reproduction = previous.get("raw_reproduction", {})
    if not reproduction.get("passed") or abs(float(reproduction.get("recorded_AP_50_95", math.nan)) - DETECTOR_RECORDED_AP) > 1e-12:
        raise RuntimeError("The reusable raw branch has no valid checkpoint-reproduction audit")
    if abs(float(reproduction.get("evaluated_raw_AP_50_95", math.nan)) - DETECTOR_RECORDED_AP) > REPRODUCTION_TOLERANCE:
        raise RuntimeError("The reusable raw branch did not reproduce the detector AP within tolerance")
    read_required_metrics(previous.get("raw", {}), "raw")
    read_required_metrics(previous.get("funie_enhanced", {}), "FUnIE-enhanced")
    derivative_manifest = json.loads(PREVIOUS_DERIVATIVE_MANIFEST.read_text(encoding="utf-8"))
    if derivative_manifest.get("status") != "passed" or derivative_manifest.get("summary", {}).get("count") != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError("The reusable FUnIE derivative manifest is incomplete")
    timings = previous.get("timing_seconds")
    timing_keys = {"raw_detector", "enhancement_plus_write", "enhanced_detector"}
    if not isinstance(timings, dict) or not timing_keys.issubset(timings):
        raise RuntimeError("The reusable raw-versus-FUnIE timing record is incomplete")
    return {
        "result_path": str(PREVIOUS_RESULT.relative_to(PROJECT)),
        "result_sha256": digest(PREVIOUS_RESULT),
        "derivative_manifest_path": str(PREVIOUS_DERIVATIVE_MANIFEST.relative_to(PROJECT)),
        "derivative_manifest_sha256": digest(PREVIOUS_DERIVATIVE_MANIFEST),
        "raw": previous["raw"],
        "funie_enhanced": previous["funie_enhanced"],
        "raw_reproduction": reproduction,
        "previous_timing_seconds": {key: float(timings[key]) for key in timing_keys},
    }


def create_resize_only_derivatives(
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> tuple[dict[int, Path], list[dict[str, Any]], float]:
    """Create PNGs using the FUnIE geometry path, with no FUnIE model call."""
    derivative_root = output_dir / "resize_only_validation_images"
    derivative_root.mkdir(parents=True)
    derivatives: dict[int, Path] = {}
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, image_id in enumerate(sorted(images), start=1):
        image = images[image_id]
        source_path = PROJECT / str(image["file_name"])
        material_root = PROJECT / "datasets/trashcan/material_version"
        if not source_path.is_file() or not source_path.is_relative_to(material_root):
            raise RuntimeError(f"Source is not a frozen raw TrashCan material_version validation image: {source_path}")
        with Image.open(source_path) as opened:
            raw = opened.convert("RGB")
        width, height = raw.size
        if (width, height) != (int(image["width"]), int(image["height"])):
            raise RuntimeError(f"Frozen image dimensions changed: {source_path}")
        for annotation in annotations.get(image_id, []):
            valid_original_bbox(annotation, width, height)

        # This is intentionally the exact FUnIE input/output geometry path.
        # Passing its normalized canvas straight through is the only difference:
        # no neural model is loaded or invoked for this control.
        funie_input, geometry = aspect_preserving_funie_canvas(raw)
        resize_only = restore_enhanced_geometry(funie_input, geometry)
        if resize_only.size != (width, height):
            raise RuntimeError("Resize-only derivative did not retain exact original dimensions")
        target = derivative_root / f"{image_id:06d}_{Path(str(image['file_name'])).stem}.png"
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite a resize-only derivative: {target}")
        resize_only.save(target)
        with Image.open(target) as decoded:
            if decoded.convert("RGB").size != (width, height):
                raise RuntimeError(f"Resize-only derivative did not decode with original dimensions: {target}")
        derivatives[image_id] = target
        manifest.append({
            "image_id": image_id,
            "source_image_path": str(image["file_name"]),
            "resize_only_derivative_path": str(target.relative_to(PROJECT)),
            "source_dimensions": {"width": width, "height": height},
            "resize_only_dimensions": {"width": resize_only.width, "height": resize_only.height},
            "bbox_coordinate_alignment": "unchanged_original_pixel_coordinates; annotations are not copied, transformed, or rewritten",
            "bbox_count": len(annotations.get(image_id, [])),
            "geometry": geometry,
            "processing": "FUnIE-GAN bypassed: normalized 256x256 input canvas was passed directly to the identical crop, bicubic restore, and PNG-save path",
            "resize_only_sha256": digest(target),
        })
        if index % 50 == 0:
            print(f"wrote resize-only {index}/{len(images)} frozen validation images", flush=True)
    duration = round(time.perf_counter() - started, 3)
    if len(derivatives) != len(images):
        raise RuntimeError("Resize-only derivative count does not match the frozen validation split")
    if any(item["source_dimensions"] != item["resize_only_dimensions"] for item in manifest):
        raise RuntimeError("At least one resize-only derivative changed the image coordinate system")
    write_json(output_dir / "resize_only_derivative_manifest.json", {
        "status": "passed",
        "images": manifest,
        "summary": {
            "count": len(manifest),
            "duration_seconds": duration,
            "geometry": "Source -> aspect-preserving bicubic resize (longest side 256) -> reflection/edge pad to 256x256 -> identical padding removal -> bicubic resize to original dimensions -> PNG save. FUnIE-GAN was not invoked.",
            "bbox_alignment": "All resize-only PNG derivatives decode at original width/height. Frozen COCO boxes remain unchanged in original pixel coordinates.",
        },
    })
    return derivatives, manifest, duration


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return right-minus-left deltas for every requested metric."""
    metric_keys = ("AP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_small_500", "AR_medium_500", "AR_large_500")
    shared_keys = ("F1", "precision", "recall")
    return {
        "training_compatible_metrics": {
            key: finite(right["training_compatible_metrics"][key], key) - finite(left["training_compatible_metrics"][key], key)
            for key in metric_keys
        },
        "precision_recall_iou50_shared_threshold": {
            key: finite(right["precision_recall_iou50_shared_threshold"][key], key) - finite(left["precision_recall_iou50_shared_threshold"][key], key)
            for key in shared_keys
        },
    }


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
    previous = load_verified_previous_results(frozen_checks)
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    validation_rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, validation_rows, "valid")
    if len(images) != EXPECTED_VALIDATION_IMAGES or sum(len(value) for value in annotations.values()) != EXPECTED_ANNOTATIONS:
        raise RuntimeError("This control requires exactly 1,082 frozen validation images and 289 annotations")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    memory_before = mps_memory()
    write_json(output_dir / "preflight.json", {
        "scope": "Validation-only resize control. It uses the raw-trained detector and frozen raw validation data; no FUnIE inference, training, test access, or source-data writes occur.",
        "detector": detector_record,
        "funie_candidate_verified_for_reused_result": funie_record,
        "frozen_checks": frozen_checks,
        "validation_images": len(images),
        "annotations": sum(len(value) for value in annotations.values()),
        "resize_only_geometry_policy": "Source -> aspect-preserving bicubic resize (longest side 256) -> reflection/edge pad to 256x256 -> bypass FUnIE -> remove identical padding -> bicubic resize to exact original width/height -> PNG save.",
        "detector_policy": "Identical pinned RF-DETR validation preprocessing, unthresholded postprocess, and vernier COCO metric backend. Neural detector inference must remain on MPS; CPU is used only for image I/O, resampling, and metric aggregation.",
        "reused_raw_and_funie_result": {key: value for key, value in previous.items() if key not in {"raw", "funie_enhanced"}},
    })

    derivatives, derivative_manifest, derivative_time = create_resize_only_derivatives(images, annotations, output_dir)

    import rfdetr
    from rfdetr import RFDETRSmall

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The detector package is not the pinned project-local RF-DETR source")
    detector = RFDETRSmall.from_checkpoint(str(PROJECT / detector_record["path"]), device="mps")
    core = detector.model.model
    if core is None or detector.model_config.resolution != 512 or detector.model_config.num_classes != 1:
        raise RuntimeError("The preserved detector did not load as one-class, 512px RF-DETR Small")
    core.to(device).eval()
    if any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError("The detector is not wholly MPS-resident")
    resize_only_result, _ = evaluate_detector_variant(
        "resize_only", detector, core, images, annotations, lambda image_id, _image: derivatives[image_id], device
    )
    del detector, core
    torch.mps.empty_cache()

    raw_result = previous["raw"]
    enhanced_result = previous["funie_enhanced"]
    comparisons = {
        "resize_only_minus_raw": metric_delta(raw_result, resize_only_result),
        "funie_enhanced_minus_resize_only": metric_delta(resize_only_result, enhanced_result),
        "funie_enhanced_minus_raw": metric_delta(raw_result, enhanced_result),
    }
    result = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Three-way validation-only comparison: raw inputs, exact resize-only control, and prior FUnIE-enhanced inputs. Detector trained only on raw inputs. A negative enhanced result cannot establish that detector training on enhanced images would not help.",
        "device": "mps",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_fallback_disabled": True,
        "detector": detector_record,
        "frozen_checks": frozen_checks,
        "ground_truth_object_counts": {"total": EXPECTED_ANNOTATIONS, "COCO_area_buckets": coco_area_counts(annotations)},
        "reused_previous_evaluation": {key: value for key, value in previous.items() if key not in {"raw", "funie_enhanced"}},
        "raw": raw_result,
        "resize_only": resize_only_result,
        "funie_enhanced": enhanced_result,
        "comparisons": comparisons,
        "resize_only_derivatives": {
            "directory": str((output_dir / "resize_only_validation_images").relative_to(PROJECT)),
            "manifest": str((output_dir / "resize_only_derivative_manifest.json").relative_to(PROJECT)),
            "count": len(derivative_manifest),
            "geometry_alignment": "verified exact original width/height for every derivative; frozen annotations remain unchanged in the original coordinate system",
        },
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 3),
            "resize_only_derivative_write": derivative_time,
            "resize_only_detector": resize_only_result["timing_seconds"],
            "reused_raw_detector": previous["previous_timing_seconds"]["raw_detector"],
            "reused_funie_enhancement_plus_write": previous["previous_timing_seconds"]["enhancement_plus_write"],
            "reused_funie_detector": previous["previous_timing_seconds"]["enhanced_detector"],
        },
        "memory": {"before": memory_before, "after": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(output_dir / "raw_resize_only_funie_result.json", result)

    metric_names = ("AP_50_95", "AP50", "AP75")
    size_rows = (("Small", "AP_small", "AR_small_500"), ("Medium", "AP_medium", "AR_medium_500"), ("Large", "AP_large", "AR_large_500"))
    def values(variant: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
        return [f"{finite(variant['training_compatible_metrics'][key], key):.4f}" for key in keys]
    raw_shared = raw_result["precision_recall_iou50_shared_threshold"]
    resize_shared = resize_only_result["precision_recall_iou50_shared_threshold"]
    enhanced_shared = enhanced_result["precision_recall_iou50_shared_threshold"]
    lines = [
        "# Raw, resize-only, and FUnIE-enhanced validation comparison",
        "",
        "This validation-only study uses the same preserved no-jitter epoch-3 EMA",
        "RF-DETR Small detector and the frozen 1,082-image TrashCan validation split.",
        "The detector was trained on raw images. Raw and FUnIE-enhanced metrics are",
        "reused from the prior hash-verified comparison; only the resize-only branch",
        "was newly evaluated here.",
        "",
        f"- Detector SHA-256: `{DETECTOR_SHA256}`",
        f"- FUnIE candidate SHA-256, verified for the reused branch: `{FUNIE_SHA256}`",
        f"- Reused prior-result SHA-256: `{previous['result_sha256']}`",
        f"- Resize-only derivatives: {len(derivative_manifest)}, each decoded at its original width and height; frozen COCO boxes were reused unchanged.",
        "",
        "## What the control measures",
        "",
        "Resize-only reproduces the FUnIE branch's source-to-256-pixel-longest-side",
        "bicubic resize, reflection padding to 256x256, padding removal, bicubic",
        "restore to the original dimensions, and PNG-save path—but bypasses FUnIE-GAN.",
        "Therefore raw versus resize-only measures the resampling pipeline's effect,",
        "and resize-only versus FUnIE-enhanced measures FUnIE's additional effect",
        "within this particular pipeline. Neither comparison tests a detector trained",
        "on enhanced images.",
        "",
        "| Metric | Raw | Resize-only | FUnIE-enhanced | Resize − raw | FUnIE − resize |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in zip(metric_names, ("AP@[.50:.95]", "AP50", "AP75"), strict=True):
        lines.append(
            f"| {label} | {finite(raw_result['training_compatible_metrics'][key], key):.4f} | {finite(resize_only_result['training_compatible_metrics'][key], key):.4f} | {finite(enhanced_result['training_compatible_metrics'][key], key):.4f} | {comparisons['resize_only_minus_raw']['training_compatible_metrics'][key]:+.4f} | {comparisons['funie_enhanced_minus_resize_only']['training_compatible_metrics'][key]:+.4f} |"
        )
    for label, ap_key, ar_key in size_rows:
        lines.append(
            f"| {label} AP / AR | {finite(raw_result['training_compatible_metrics'][ap_key], ap_key):.4f} / {finite(raw_result['training_compatible_metrics'][ar_key], ar_key):.4f} | {finite(resize_only_result['training_compatible_metrics'][ap_key], ap_key):.4f} / {finite(resize_only_result['training_compatible_metrics'][ar_key], ar_key):.4f} | {finite(enhanced_result['training_compatible_metrics'][ap_key], ap_key):.4f} / {finite(enhanced_result['training_compatible_metrics'][ar_key], ar_key):.4f} | {comparisons['resize_only_minus_raw']['training_compatible_metrics'][ap_key]:+.4f} / {comparisons['resize_only_minus_raw']['training_compatible_metrics'][ar_key]:+.4f} | {comparisons['funie_enhanced_minus_resize_only']['training_compatible_metrics'][ap_key]:+.4f} / {comparisons['funie_enhanced_minus_resize_only']['training_compatible_metrics'][ar_key]:+.4f} |"
        )
    for key, label in (("F1", "F1 @ 0.43"), ("precision", "Precision @ 0.43"), ("recall", "Recall @ 0.43")):
        lines.append(
            f"| {label} | {finite(raw_shared[key], key):.4f} | {finite(resize_shared[key], key):.4f} | {finite(enhanced_shared[key], key):.4f} | {comparisons['resize_only_minus_raw']['precision_recall_iou50_shared_threshold'][key]:+.4f} | {comparisons['funie_enhanced_minus_resize_only']['precision_recall_iou50_shared_threshold'][key]:+.4f} |"
        )
    area = coco_area_counts(annotations)
    lines.extend([
        "",
        f"COCO area-bucket counts: small {area['small']}, medium {area['medium']}, large {area['large']}.",
        "",
        f"Timing: resize-only derivative/write {derivative_time:.1f}s; resize-only detector {resize_only_result['timing_seconds']:.1f}s. The raw and FUnIE timings in the JSON are retained from the reused prior evaluation, not rerun here.",
    ])
    (output_dir / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
