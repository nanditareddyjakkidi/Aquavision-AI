#!/usr/bin/env python3
"""Run the strictly scoped cls_loss_coef=2 RF-DETR continuation, epochs 4--8.

This MPS-only exploratory ablation restores the exact full Lightning state
behind the no-jitter epoch-3 candidate, retains the existing one-fifth LR
procedure, and changes *only* ``cls_loss_coef`` from 1.0 to 2.0.  It uses the
frozen raw TrashCan train/validation data, including negatives.  It neither
reads test data nor uses enhanced inputs.

It deliberately cannot claim an exact stochastic replay: the source full
checkpoint has no serialised Python/NumPy/Torch/MPS RNG state nor sampler
permutation/data-loader generator state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.metadata as metadata
import json
import math
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only ablation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from pytorch_lightning import Callback

import continue_rfdetr_no_jitter_epoch3_lower_lr as base
from evaluate_rfdetr_raw_vs_funie_enhanced import evaluate_detector_variant
from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING,
    FREEZE_FILE,
    PROJECT,
    RFDETR_SOURCE,
    RUNS_ROOT,
    TRAIN_COCO,
    VAL_COCO,
    build_adapter,
    check_coco_payload,
    digest,
    frozen_entries,
    load_manifest,
    mps_memory,
    require_mps,
    verify_frozen_file,
)
from run_rfdetr_raw_v2_scale_jitter_off import (
    EXPECTED_PRETRAIN_SHA256,
    PREVIEW_COUNT,
    SEED,
    create_alias,
    load_ema_model,
    validate_preview_plan,
    write_json,
)


TARGET_CLS_LOSS_COEF = 2.0
SOURCE_CLS_LOSS_COEF = 1.0
EXISTING_LOWER_LR_RUN = RUNS_ROOT / "rfdetr_no_jitter_epoch3_lower_lr_e4to8_mps_20261003T054821Z"
EXISTING_LOWER_LR_RESULT = EXISTING_LOWER_LR_RUN / "lower_lr_result.json"
EXISTING_LOWER_LR_HISTORY = EXISTING_LOWER_LR_RUN / "metric_history_ema.json"


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is non-finite")
    return result


def assert_equal(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {left} does not equal {right}")


def intended_config(source: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    candidate = copy.deepcopy(source)
    train = candidate["train_config"]
    train.update({
        "epochs": base.TARGET_TOTAL_EPOCHS,
        "dataset_dir": str(run_dir / "raw_coco_adapter"),
        "output_dir": str(run_dir / "training_output"),
        "resume": str(base.SOURCE_FULL_CHECKPOINT.relative_to(PROJECT)),
        "lr": base.MAX_LR_AFTER,
        "lr_encoder": base.ENCODER_LR_AFTER,
        "cls_loss_coef": TARGET_CLS_LOSS_COEF,
        "notes": {
            "purpose": "Controlled RF-DETR Small no-jitter lower-LR classification-loss ablation from epoch-3 full state",
            "only_semantic_change_from_existing_lower_lr_branch": "cls_loss_coef 1.0 -> 2.0",
            "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent",
            "enhancement_input": "not used",
            "comparison": "Existing lower-LR no-jitter epochs 4-8, with the same epoch-3 source state and LR /5 procedure",
        },
    })
    return candidate


def configuration_audit(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    source_train, candidate_train = source["train_config"], candidate["train_config"]
    source_model, candidate_model = source["model_config"], candidate["model_config"]
    train_diffs = {
        key: {"source": source_train.get(key), "candidate": candidate_train.get(key)}
        for key in sorted(set(source_train).union(candidate_train))
        if source_train.get(key) != candidate_train.get(key)
    }
    model_diffs = {
        key: {"source": source_model.get(key), "candidate": candidate_model.get(key)}
        for key in sorted(set(source_model).union(candidate_model))
        if source_model.get(key) != candidate_model.get(key)
    }
    allowed = {"epochs", "dataset_dir", "output_dir", "resume", "lr", "lr_encoder", "cls_loss_coef", "notes"}
    unexpected = set(train_diffs).difference(allowed)
    if unexpected or model_diffs:
        raise RuntimeError(f"Ablation would change settings beyond the approved set: train={sorted(unexpected)}, model={model_diffs}")
    required = {
        "epochs": {"source": 10, "candidate": 8},
        "lr": {"source": base.MAX_LR_BEFORE, "candidate": base.MAX_LR_AFTER},
        "lr_encoder": {"source": base.ENCODER_LR_BEFORE, "candidate": base.ENCODER_LR_AFTER},
        "cls_loss_coef": {"source": SOURCE_CLS_LOSS_COEF, "candidate": TARGET_CLS_LOSS_COEF},
    }
    if {key: train_diffs.get(key) for key in required} != required:
        raise RuntimeError("Ablation does not retain the specified epoch budget, lower LRs, and cls_loss_coef=2.0")
    return {
        "approved_train_config_differences": train_diffs,
        "model_config_differences": model_diffs,
        "controlled_change": {
            "classification_loss_coefficient": {"before": SOURCE_CLS_LOSS_COEF, "after": TARGET_CLS_LOSS_COEF},
            "learning_rate_divisor": base.LR_DIVISOR,
            "maximum_group_learning_rate": {"before": base.MAX_LR_BEFORE, "after": base.MAX_LR_AFTER},
            "method": "After Lightning restores the source optimizer/scheduler state, the inherited lower-LR callback divides every optimizer group LR, initial_lr, scheduler base_lr, and scheduler last_lr by five. The only additional training-setting difference from that lower-LR branch is cls_loss_coef=2.0.",
        },
        "loss_scope": "Pinned lwdetr.py builds loss_ce from cls_loss_coef then duplicates it to auxiliary decoder and encoder losses. Matcher costs, IA-BCE settings, bbox/GIoU weights, augmentation, resolution, batch size, data, EMA, and scheduler settings are checked separately.",
        "epoch_budget_note": "The max epoch value is 8 solely to execute completed epochs 4-8. The pinned step scheduler first drops after epoch 100, so only the inherited LR /5 scaling is active here.",
        "reproducibility_note": "Because the source checkpoint has no RNG or sampler-permutation snapshot, this is a controlled exploratory comparison, not an exact stochastic replay.",
    }


class ClassificationLossAudit(Callback):
    """Fail if the actual pinned criterion resolves anything other than cls=2 only."""

    def __init__(self) -> None:
        super().__init__()
        self.record: dict[str, Any] | None = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        train = getattr(pl_module, "train_config", None)
        criterion = getattr(pl_module, "criterion", None)
        matcher = getattr(criterion, "matcher", None)
        if train is None or criterion is None or matcher is None:
            raise RuntimeError("RF-DETR training module did not expose train_config, criterion, and matcher")
        assert_equal(finite(getattr(train, "cls_loss_coef", math.nan), "runtime cls_loss_coef"), TARGET_CLS_LOSS_COEF, "runtime cls_loss_coef")
        weights = {str(key): finite(value, f"criterion weight {key}") for key, value in criterion.weight_dict.items()}
        expected_classification = {"loss_ce", "loss_ce_0", "loss_ce_1", "loss_ce_enc"}
        actual_classification = {key: value for key, value in weights.items() if key == "loss_ce" or key.startswith("loss_ce_")}
        if set(actual_classification) != expected_classification:
            raise RuntimeError(f"Pinned criterion resolved unexpected classification loss terms: {sorted(actual_classification)}")
        for name, value in actual_classification.items():
            assert_equal(value, TARGET_CLS_LOSS_COEF, f"{name} weight")
        expected_box = {"loss_bbox", "loss_bbox_0", "loss_bbox_1", "loss_bbox_enc"}
        expected_giou = {"loss_giou", "loss_giou_0", "loss_giou_1", "loss_giou_enc"}
        for name in expected_box:
            assert_equal(finite(weights.get(name, math.nan), name), 5.0, f"{name} weight")
        for name in expected_giou:
            assert_equal(finite(weights.get(name, math.nan), name), 2.0, f"{name} weight")
        if list(getattr(criterion, "losses", [])) != ["labels", "boxes", "cardinality"]:
            raise RuntimeError("The criterion loss set differs from the pinned one-class detection recipe")
        assert_equal(finite(getattr(criterion, "focal_alpha", math.nan), "criterion focal_alpha"), 0.25, "criterion focal_alpha")
        if not getattr(criterion, "ia_bce_loss", False) or getattr(criterion, "use_varifocal_loss", True) or getattr(criterion, "use_position_supervised_loss", True):
            raise RuntimeError("The criterion no longer uses the pinned IA-BCE classification-loss configuration")
        assert_equal(finite(getattr(matcher, "cost_class", math.nan), "matcher class cost"), 2.0, "matcher class cost")
        assert_equal(finite(getattr(matcher, "cost_bbox", math.nan), "matcher bbox cost"), 5.0, "matcher bbox cost")
        assert_equal(finite(getattr(matcher, "cost_giou", math.nan), "matcher GIoU cost"), 2.0, "matcher GIoU cost")
        assert_equal(finite(getattr(matcher, "focal_alpha", math.nan), "matcher focal_alpha"), 0.25, "matcher focal_alpha")
        self.record = {
            "passed": True,
            "train_config_cls_loss_coef": TARGET_CLS_LOSS_COEF,
            "loss_assembly": "The pinned module sums loss_dict[k] * criterion.weight_dict[k] for weighted keys. cardinality_error remains an unweighted diagnostic.",
            "classification_loss_weights": {
                "main": {"loss_ce": actual_classification["loss_ce"]},
                "auxiliary_decoder": {key: actual_classification[key] for key in ("loss_ce_0", "loss_ce_1")},
                "auxiliary_encoder": {"loss_ce_enc": actual_classification["loss_ce_enc"]},
            },
            "other_resolved_loss_weights": {
                "bbox": {key: weights[key] for key in sorted(expected_box)},
                "giou": {key: weights[key] for key in sorted(expected_giou)},
            },
            "matcher_costs": {"class": float(matcher.cost_class), "bbox": float(matcher.cost_bbox), "giou": float(matcher.cost_giou), "focal_alpha": float(matcher.focal_alpha)},
            "classification_recipe": {
                "ia_bce_loss": True,
                "use_varifocal_loss": False,
                "use_position_supervised_loss": False,
                "focal_alpha": float(criterion.focal_alpha),
                "pinned_implementation_note": "With IA-BCE enabled, the label target for matched positives is max(sigmoid(logit)^0.25 * IoU^0.75, 0.01); the implementation uses p^2 for ordinary negative weighting. cls_loss_coef multiplies the main and auxiliary loss_ce values after the criterion calculates them.",
            },
            "all_weight_dict_entries": weights,
        }


def load_existing_lower_lr_history() -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if not EXISTING_LOWER_LR_RESULT.is_file() or not EXISTING_LOWER_LR_HISTORY.is_file():
        raise RuntimeError("The existing lower-LR branch artifacts are unavailable for comparison")
    result = json.loads(EXISTING_LOWER_LR_RESULT.read_text(encoding="utf-8"))
    history = json.loads(EXISTING_LOWER_LR_HISTORY.read_text(encoding="utf-8"))
    if result.get("status") != "passed" or not isinstance(history, list):
        raise RuntimeError("Existing lower-LR branch is not a complete successful reference")
    source = result.get("source_checkpoint", {}).get("full_state_checkpoint", {})
    if source.get("sha256") != base.SOURCE_FULL_SHA256 or int(source.get("completed_epoch", -1)) != base.SOURCE_COMPLETED_EPOCH:
        raise RuntimeError("Existing lower-LR branch did not use the same verified epoch-3 full checkpoint")
    by_epoch = {int(item["completed_epoch"]): item for item in history}
    if set(by_epoch) != set(base.CONTINUATION_EPOCHS):
        raise RuntimeError("Existing lower-LR branch does not contain exactly epochs 4-8")
    return by_epoch, {
        "run_result": str(EXISTING_LOWER_LR_RESULT.relative_to(PROJECT)),
        "run_result_sha256": digest(EXISTING_LOWER_LR_RESULT),
        "history": str(EXISTING_LOWER_LR_HISTORY.relative_to(PROJECT)),
        "history_sha256": digest(EXISTING_LOWER_LR_HISTORY),
        "source_full_checkpoint_sha256": base.SOURCE_FULL_SHA256,
    }


def compare_histories(candidate: list[dict[str, Any]], lower_lr: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("AP_50_95", "AP50", "AP75", "mAR", "F1_optimized_threshold", "precision_optimized_threshold", "recall_optimized_threshold")
    result: list[dict[str, Any]] = []
    for entry in candidate:
        epoch = int(entry["completed_epoch"])
        reference = lower_lr[epoch]
        result.append({
            "completed_epoch": epoch,
            "cls_loss_coef_2": entry,
            "existing_lower_lr_cls_loss_coef_1": reference,
            "EMA_metric_deltas_cls2_minus_existing_lower_lr": {
                key: float(entry["EMA_validation"][key]) - float(reference["EMA_validation"][key]) for key in keys
            },
        })
    return result


def verify_inference_checkpoint_matches_full_state(inference: Path, full: Path, selected_epoch: int) -> dict[str, Any]:
    if not inference.is_file():
        raise RuntimeError("RF-DETR did not write its selected best EMA inference checkpoint")
    full_payload = torch.load(full, map_location="cpu", weights_only=False)
    inference_payload = torch.load(inference, map_location="cpu", weights_only=False)
    average = full_payload.get("callbacks", {}).get("RFDETREMACallback", {}).get("average_model_state_dict") if isinstance(full_payload, dict) else None
    inference_model = inference_payload.get("model") if isinstance(inference_payload, dict) else None
    if not isinstance(average, dict) or not isinstance(inference_model, dict):
        raise RuntimeError("Cannot verify the selected best EMA inference checkpoint against its full state")
    full_hash = base.state_fingerprint(average, strip_ema_prefix=True)
    inference_hash = base.state_fingerprint(inference_model)
    if full_hash != inference_hash:
        raise RuntimeError("checkpoint_best_ema.pth does not match the selected full-state EMA weights")
    return {
        "path": str(inference.relative_to(PROJECT)),
        "sha256": digest(inference),
        "completed_epoch": selected_epoch,
        "EMA_model_state_sha256": inference_hash,
        "matches_selected_full_state": True,
    }


def negative_summary(quality: dict[int, dict[str, Any]], annotations: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    negatives = [image_id for image_id in sorted(quality) if not annotations.get(image_id, [])]
    false_positives = sum(int(quality[image_id]["fp"]) for image_id in negatives)
    images_with_fp = sum(int(quality[image_id]["fp"]) > 0 for image_id in negatives)
    return {
        "shared_threshold": 0.43,
        "iou_threshold": 0.5,
        "negative_images": len(negatives),
        "false_positives_on_negative_images": false_positives,
        "false_positives_per_negative_image": false_positives / len(negatives) if negatives else 0.0,
        "negative_images_with_false_detection": images_with_fp,
        "fraction_negative_images_with_false_detection": images_with_fp / len(negatives) if negatives else 0.0,
    }


def evaluate_selected_checkpoint(run_dir: Path, checkpoint: Path, epoch: int, history_record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Evaluate the selected full-state EMA with the established validation-only path."""
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, rows, "valid")
    if len(images) != 1082 or sum(len(value) for value in annotations.values()) != 289:
        raise RuntimeError("Selected-checkpoint evaluation requires the 1,082-image, 289-instance frozen validation split")
    groups = {str(row["source_image_path"]): str(row["video_group"]) for row in rows}
    if set(groups) != {str(image["file_name"]) for image in images.values()}:
        raise RuntimeError("Frozen validation manifest and COCO image map disagree")
    model = load_ema_model(checkpoint, device)
    core = model.model.model
    if core is None:
        raise RuntimeError("Selected full checkpoint did not yield a live RF-DETR module")
    metrics, quality = evaluate_detector_variant(
        f"cls_loss_coef_2_epoch_{epoch}", model, core, images, annotations,
        lambda _image_id, image: PROJECT / str(image["file_name"]), device,
    )
    del model, core
    gc.collect()
    torch.mps.empty_cache()
    recorded = finite(history_record["EMA_validation"]["AP_50_95"], "recorded selected AP")
    measured = finite(metrics["training_compatible_metrics"]["AP_50_95"], "measured selected AP")
    reproduction = {"recorded_AP_50_95": recorded, "evaluated_AP_50_95": measured, "absolute_difference": abs(measured - recorded), "tolerance": 0.001}
    reproduction["passed"] = reproduction["absolute_difference"] <= reproduction["tolerance"]
    if not reproduction["passed"]:
        raise RuntimeError("Selected-checkpoint evaluator did not reproduce its recorded EMA AP within 0.001")
    per_image_rows = []
    for image_id in sorted(images):
        item = quality[image_id]
        per_image_rows.append({
            "image_id": image_id,
            "source_image_path": str(images[image_id]["file_name"]),
            "video_group": groups[str(images[image_id]["file_name"])],
            "ground_truth_instances": len(annotations.get(image_id, [])),
            "tp": int(item["tp"]), "fp": int(item["fp"]), "fn": int(item["fn"]), "F1": float(item["F1"]),
        })
    evaluation_dir = run_dir / "selected_checkpoint_validation"
    evaluation_dir.mkdir(parents=True)
    per_image_path = evaluation_dir / "per_image_shared_threshold_043.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    area_counts = {"small": 0, "medium": 0, "large": 0}
    for values in annotations.values():
        for annotation in values:
            area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
            area_counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    result = {
        "status": "passed",
        "checkpoint": {"path": str(checkpoint.relative_to(PROJECT)), "sha256": digest(checkpoint), "completed_epoch": epoch, "validation_weights": "EMA"},
        "validation_images": len(images),
        "ground_truth_object_counts": {"total": 289, "COCO_area_buckets": area_counts},
        "training_compatible_ema_metrics": metrics["training_compatible_metrics"],
        "precision_recall_iou50_shared_threshold": metrics["precision_recall_iou50_shared_threshold"],
        "precision_recall_iou50_validation_selected_best_f1": metrics["precision_recall_iou50_validation_selected_best_f1"],
        "negative_image_false_positives_shared_threshold": negative_summary(quality, annotations),
        "reproduction": reproduction,
        "per_image_shared_threshold_043": str(per_image_path.relative_to(PROJECT)),
        "detector_preprocessing": metrics["detector_preprocessing"],
        "timing_seconds": metrics["timing_seconds"],
        "memory_after": metrics["memory_after"],
    }
    write_json(evaluation_dir / "selected_checkpoint_evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New ablation directory name under training_runs/rfdetr/")
    parser.add_argument("--epochs", type=int, default=base.TARGET_TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    parser.add_argument("--cls-loss-coef", type=float, default=TARGET_CLS_LOSS_COEF)
    args = parser.parse_args()
    if args.epochs != base.TARGET_TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.seed != SEED or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This ablation requires --epochs 8 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9")
    if args.cls_loss_coef != TARGET_CLS_LOSS_COEF:
        raise SystemExit("This prepared ablation requires --cls-loss-coef 2.0 exactly")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple new directory name")
    source = base.inspect_source_checkpoint()
    source_config, _original_no_jitter, source_preview = base.source_config_and_reference_history()
    if finite(source_config["train_config"].get("cls_loss_coef", math.nan), "source cls_loss_coef") != SOURCE_CLS_LOSS_COEF:
        raise RuntimeError("The no-jitter epoch-3 source configuration did not use cls_loss_coef=1.0")
    lower_lr_history, lower_lr_provenance = load_existing_lower_lr_history()
    preview_plan = validate_preview_plan(source_preview)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse an existing ablation directory: {run_dir}")
    required = (FREEZE_FILE, TRAIN_COCO, VAL_COCO, CLASS_MAPPING, RFDETR_SOURCE)
    if not all(path.exists() for path in required):
        missing = [str(path.relative_to(PROJECT)) for path in required if not path.exists()]
        raise SystemExit(f"Required frozen/source artifact is missing: {missing}")
    expected_cache = (PROJECT / ".cache/rfdetr").resolve()
    if Path(os.environ.get("RF_HOME", "")).resolve() != expected_cache:
        raise SystemExit("RF_HOME must resolve to this project's .cache/rfdetr directory")
    device = require_mps()
    intended = intended_config(source_config, run_dir)
    config_audit = configuration_audit(source_config, intended)
    frozen = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, frozen) for path in (TRAIN_COCO, VAL_COCO, CLASS_MAPPING)}

    import rfdetr
    import rfdetr.training as training_stack
    from rfdetr import RFDETRSmall
    from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir, validate_pretrain_weights

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The imported RF-DETR package is not the pinned project-local source")
    if Path(get_model_cache_dir()).resolve() != expected_cache:
        raise RuntimeError("RF-DETR pretrained-weight cache is not project-local")
    asset = ModelWeights.from_filename("rf-detr-small.pth")
    pretrained = expected_cache / "rf-detr-small.pth"
    if asset is None or asset.md5_hash is None or not pretrained.is_file() or digest(pretrained) != EXPECTED_PRETRAIN_SHA256 or not validate_pretrain_weights(str(pretrained), strict=True):
        raise RuntimeError("The original verified official RF-DETR Small pretrained checkpoint is unavailable or changed")

    run_dir.mkdir(parents=True)
    adapter = build_adapter(run_dir, PREVIEW_COUNT)
    adapter["fixed_validation_previews"] = preview_plan["examples"]
    adapter["description"] = "Read-only adapter over frozen raw TrashCan material_version train/validation images. Negatives retained; test and enhanced images are absent."
    write_json(run_dir / "preview_plan.json", preview_plan)
    seed_record = base.seed_available_generators(args.seed)
    write_json(run_dir / "configuration_audit_before_training.json", {
        "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_audit": source,
        "configuration_difference_audit": config_audit,
        "intended_resolved_config": intended,
        "frozen_checks": frozen_checks,
        "adapter": adapter,
        "RNG_and_data_order": {"seed_action": seed_record, "checkpoint_rng_state_paths": source["full_state_checkpoint"]["rng_state_paths"], "data_order_state": source["full_state_checkpoint"]["data_order_state"]},
        "existing_lower_lr_reference": lower_lr_provenance,
        "pinned_source": {"rfdetr_version": metadata.version("rfdetr"), "source_revision": base.git_revision(RFDETR_SOURCE), "pretrained_sha256": digest(pretrained)},
        "scope": "Prepared MPS-only exploratory continuation from completed epoch 3 through completed epoch 8. It does not train until the user runs this script in macOS Terminal.",
    })

    model = RFDETRSmall(resolution=512, device="mps")
    if Path(model.model_config.pretrain_weights).resolve() != pretrained or not validate_pretrain_weights(str(model.model_config.pretrain_weights), strict=True):
        raise RuntimeError("RF-DETR did not initialize from the verified project-local pretrained asset before full-state restore")
    telemetry = base.EpochTelemetry()
    mps_audit = base.MPSExecutionAudit()
    lr_audit = base.RestoreAndLowerLRAudit(source)
    cls_audit = ClassificationLossAudit()
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_audits(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.extend((telemetry, mps_audit, lr_audit, cls_audit))
        return trainer

    output_dir = run_dir / "training_output"
    started, memory_before = time.perf_counter(), mps_memory()
    training_stack.build_trainer = build_trainer_with_audits
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=base.TARGET_TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512, resume=str(base.SOURCE_FULL_CHECKPOINT),
            optimizer="adamw", lr=base.MAX_LR_AFTER, lr_encoder=base.ENCODER_LR_AFTER, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step", lr_scheduler_kwargs={}, warmup_epochs=0.0,
            grad_accum_steps=1, amp_dtype=None, augmentation_backend="cpu", multi_scale="off", expanded_scales=False,
            scale_jitter=False, square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False, compute_val_loss="auto",
            log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False, wandb=False, early_stopping=False, run_test=False,
            cls_loss_coef=TARGET_CLS_LOSS_COEF, seed=args.seed, notes=intended["train_config"]["notes"],
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    if lr_audit.restore_before_change is None or lr_audit.after_change is None or cls_audit.record is None:
        raise RuntimeError("Required resume, LR, or classification-loss audit did not run")
    if mps_audit.epochs_checked != list(base.CONTINUATION_EPOCHS):
        raise RuntimeError(f"MPS was not verified for exactly epochs 4-8: {mps_audit.epochs_checked}")
    if digest(base.SOURCE_FULL_CHECKPOINT) != base.SOURCE_FULL_SHA256 or digest(base.SOURCE_EMA_INFERENCE) != base.SOURCE_EMA_SHA256:
        raise RuntimeError("A preserved epoch-3 source checkpoint changed while the ablation ran")

    resolved_path = output_dir / "training_config.json"
    if not resolved_path.is_file():
        raise RuntimeError("RF-DETR did not write its resolved training configuration")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    after_config_audit = configuration_audit(source_config, resolved)
    if resolved.get("train_config", {}).get("resume") != str(base.SOURCE_FULL_CHECKPOINT) or resolved.get("train_config", {}).get("scale_jitter") is not False or finite(resolved.get("train_config", {}).get("cls_loss_coef", math.nan), "resolved cls_loss_coef") != TARGET_CLS_LOSS_COEF:
        raise RuntimeError("Resolved configuration did not retain full-state resume, no-jitter, and cls_loss_coef=2.0")
    write_json(run_dir / "configuration_audit_after_training.json", {
        "status": "passed", "resolved_config": str(resolved_path.relative_to(PROJECT)),
        "configuration_difference_audit": after_config_audit,
        "restore_before_lr_change": lr_audit.restore_before_change, "lr_change": lr_audit.after_change,
        "actual_learning_rates": lr_audit.epoch_learning_rates, "resolved_classification_loss": cls_audit.record,
    })

    history = base.parse_new_history(output_dir / "metrics.csv", lr_audit)
    checkpoints = {epoch: output_dir / f"checkpoint_{epoch - 1}.ckpt" for epoch in base.CONTINUATION_EPOCHS}
    checkpoint_audits = [base.audit_scaled_checkpoint(checkpoint, epoch, source["full_state_checkpoint"]) for epoch, checkpoint in checkpoints.items()]
    best_record = max(history, key=lambda record: float(record["EMA_validation"]["AP_50_95"]))
    best_epoch = int(best_record["completed_epoch"])
    best_full = checkpoints[best_epoch]
    best_inference = output_dir / "checkpoint_best_ema.pth"
    best_inference_audit = verify_inference_checkpoint_matches_full_state(best_inference, best_full, best_epoch)
    latest_full = checkpoints[base.TARGET_TOTAL_EPOCHS]
    aliases = {
        "best_validation_AP_50_95_ema": create_alias(run_dir / "checkpoints/best_validation_ap_50_95_ema.pth", best_inference),
        "latest_full_state": create_alias(run_dir / "checkpoints/latest_full_state.ckpt", latest_full),
    }

    del model
    gc.collect()
    torch.mps.empty_cache()
    previews = base.generate_previews_for_epochs(run_dir, preview_plan, checkpoints, device)
    comparisons = compare_histories(history, lower_lr_history)
    selected_evaluation = evaluate_selected_checkpoint(run_dir, best_full, best_epoch, best_record, device)
    result = {
        "status": "passed", "run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MPS-only controlled exploratory RF-DETR Small continuation from the no-jitter epoch-3 full state through epochs 4-8. It retains the established lower-LR procedure and changes only cls_loss_coef from 1.0 to 2.0. Frozen raw train/validation inputs include negative images; test, enhanced, source-dataset, and website files are not used or changed.",
        "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "cpu_fallback_disabled": True,
        "source_checkpoint": source, "restore_before_learning_rate_change": lr_audit.restore_before_change,
        "learning_rate_change": lr_audit.after_change, "actual_learning_rate_records": lr_audit.epoch_learning_rates,
        "resolved_classification_loss": cls_audit.record,
        "RNG_and_data_order_limitations": {"seed_action": seed_record, "checkpoint_rng_state_paths": source["full_state_checkpoint"]["rng_state_paths"], "data_order_state": source["full_state_checkpoint"]["data_order_state"]},
        "configuration_audits": {"before": str((run_dir / "configuration_audit_before_training.json").relative_to(PROJECT)), "after": str((run_dir / "configuration_audit_after_training.json").relative_to(PROJECT))},
        "frozen_checks": frozen_checks, "EMA_metric_history": history,
        "pinned_source": {"rfdetr_version": metadata.version("rfdetr"), "source_revision": base.git_revision(RFDETR_SOURCE), "pretrained_sha256": digest(pretrained)},
        "comparison_against_existing_lower_lr_epochs_4_to_8": comparisons,
        "existing_lower_lr_reference": lower_lr_provenance, "MPS_execution_audit": {"epochs_checked": mps_audit.epochs_checked, "cpu_fallback_disabled": True},
        "epoch_telemetry": telemetry.records, "full_checkpoint_audits": checkpoint_audits, "checkpoint_aliases": aliases,
        "selected_best_ema_epochs_4_to_8": {"completed_epoch": best_epoch, "AP_50_95": float(best_record["EMA_validation"]["AP_50_95"]), "full_state_checkpoint": str(best_full.relative_to(PROJECT)), "full_state_sha256": digest(best_full), "inference_checkpoint": best_inference_audit, "selection_scope": "maximum EMA AP@[.50:.95] among newly completed epochs 4-8 only"},
        "final_epoch_8_ema": history[-1], "selected_checkpoint_validation": selected_evaluation,
        "fixed_EMA_validation_previews": previews,
        "timing_seconds": {"training_and_postprocessing": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_postprocessing": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(run_dir / "classification_loss_ablation_result.json", result)
    write_json(run_dir / "metric_history_ema.json", history)
    report = [
        "# RF-DETR no-jitter lower-LR classification-loss ablation: epochs 4-8",
        "",
        "This is a controlled exploratory comparison, not an exact stochastic replay: the source full checkpoint restores model, optimizer, scheduler, EMA, epoch, global step, and Lightning loop state, but it has no serialised Python/NumPy/Torch/MPS RNG or sampler/data-order state.",
        "",
        "- Source: preserved no-jitter full epoch-3 state, resumed at global step 15,144.",
        "- Relative to the existing lower-LR branch, the only semantic change is `cls_loss_coef: 1.0 -> 2.0`.",
        "- All main and auxiliary classification loss weights resolved to 2.0; box/GIoU weights and matcher costs were verified unchanged.",
        "- Data remain frozen raw TrashCan train/validation with negatives retained. Test and enhanced images are excluded.",
        "",
        "| Epoch | cls=2 EMA AP | existing lower-LR cls=1 EMA AP | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        epoch = item["completed_epoch"]
        new = item["cls_loss_coef_2"]["EMA_validation"]["AP_50_95"]
        old = item["existing_lower_lr_cls_loss_coef_1"]["EMA_validation"]["AP_50_95"]
        delta = item["EMA_metric_deltas_cls2_minus_existing_lower_lr"]["AP_50_95"]
        report.append(f"| {epoch} | {new:.6f} | {old:.6f} | {delta:+.6f} |")
    report.extend([
        "",
        f"Best newly completed EMA checkpoint: epoch {best_epoch}, AP@[.50:.95] {result['selected_best_ema_epochs_4_to_8']['AP_50_95']:.6f}.",
        f"Epoch-8 EMA AP@[.50:.95]: {history[-1]['EMA_validation']['AP_50_95']:.6f}.",
        f"Selected-checkpoint AP reproduction passed: {selected_evaluation['reproduction']['passed']}.",
        f"Selected-checkpoint negative-image FP summary at threshold 0.43: {selected_evaluation['negative_image_false_positives_shared_threshold']}.",
    ])
    (run_dir / "FINDINGS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
