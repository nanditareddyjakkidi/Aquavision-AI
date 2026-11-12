#!/usr/bin/env python3
"""Run a fresh, ten-epoch MPS-only RF-DETR Small scale-jitter ablation.

This runner starts from the same MD5-verified official RF-DETR Small
pretrained checkpoint as the original raw-image experiment. It deliberately
does not resume from any AquaVision training checkpoint. The only training
behaviour changed from the fixed-512 raw-image recipe is ``scale_jitter=False``.

It uses frozen raw TrashCan v2 train/validation data only, preserves negative
images, never mounts a test split or enhanced image, and stops after epoch 10.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
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
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only experiment.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from pytorch_lightning import Callback

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
    create_labeled_previews,
    digest,
    frozen_entries,
    git_revision,
    load_manifest,
    mps_memory,
    require_mps,
    verify_frozen_file,
)
from evaluate_rfdetr_epoch5_ema_validation import (
    coco_area_counts,
    summarize_training_metric,
    training_style_f1,
    validation_tensor,
)


TOTAL_EPOCHS = 10
STEPS_PER_EPOCH = 5048
PREVIEW_COUNT = 9
SEED = 20260930
SHARED_PR_THRESHOLD = 0.43
ORIGINAL_RUN_ID = "rfdetr_raw_v2_e5to10_fixed512_mps_20261001T115124Z"
ORIGINAL_EPOCH5_RUN_ID = "rfdetr_raw_v2_e1to5_fixed512_mps_20261001T091117Z"
ORIGINAL_RUN_DIR = RUNS_ROOT / ORIGINAL_RUN_ID
ORIGINAL_CONFIG = ORIGINAL_RUN_DIR / "training_output/training_config.json"
ORIGINAL_HISTORY = ORIGINAL_RUN_DIR / "metric_history_ema.json"
ORIGINAL_PREVIEW_PLAN = ORIGINAL_RUN_DIR / "preview_plan.json"
ORIGINAL_EPOCH5_BEST_EMA = RUNS_ROOT / ORIGINAL_EPOCH5_RUN_ID / "training_output/checkpoint_best_ema.pth"
ORIGINAL_EPOCH10_FULL = ORIGINAL_RUN_DIR / "training_output/checkpoint_9.ckpt"
EXPECTED_PRETRAIN_SHA256 = "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33"
EXPECTED_EPOCH5_BEST_SHA256 = "9e269f264845c638e65da5e2b77cdc54b19764fd85c2f00b0a3d846866b1f5a2"
EXPECTED_EPOCH10_FULL_SHA256 = "9d8a29244dc3bba751fefc217141dd2507e51538235abe038c2751c1009efa18"
EXPECTED_ORIGINAL_BEST_AP = 0.12120693176984787
EXPECTED_ORIGINAL_EPOCH10_AP = 0.11057670414447784


def numeric(row: dict[str, str], key: str) -> float:
    """Read one finite scalar from RF-DETR's CSV logger."""
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"metrics.csv lacks {key}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"metrics.csv records non-finite {key}")
    return parsed


def write_json(path: Path, payload: Any) -> None:
    """Write a deterministic, human-readable project artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def original_reference() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load and verify the preserved ten-epoch raw-image reference artifacts."""
    required = (ORIGINAL_CONFIG, ORIGINAL_HISTORY, ORIGINAL_PREVIEW_PLAN, ORIGINAL_EPOCH5_BEST_EMA, ORIGINAL_EPOCH10_FULL)
    missing = [str(path.relative_to(PROJECT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Preserved original RF-DETR reference artifacts are missing: {missing}")
    if digest(ORIGINAL_EPOCH5_BEST_EMA) != EXPECTED_EPOCH5_BEST_SHA256:
        raise RuntimeError("The preserved original epoch-5 best EMA checkpoint SHA-256 changed")
    if digest(ORIGINAL_EPOCH10_FULL) != EXPECTED_EPOCH10_FULL_SHA256:
        raise RuntimeError("The preserved original epoch-10 full-state checkpoint SHA-256 changed")
    config = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    history = json.loads(ORIGINAL_HISTORY.read_text(encoding="utf-8"))
    preview_plan = json.loads(ORIGINAL_PREVIEW_PLAN.read_text(encoding="utf-8"))
    train = config.get("train_config")
    model = config.get("model_config")
    if not isinstance(train, dict) or not isinstance(model, dict) or not isinstance(history, list) or not isinstance(preview_plan, dict):
        raise RuntimeError("Original RF-DETR reference artifacts have an unexpected structure")
    required_train = {
        "epochs": TOTAL_EPOCHS,
        "batch_size": 1,
        "seed": SEED,
        "multi_scale": "off",
        "expanded_scales": False,
        "scale_jitter": True,
        "optimizer": "adamw",
        "lr": 0.0001,
        "lr_encoder": 0.00015,
        "lr_scheduler": "step",
        "lr_scheduler_interval": "step",
        "lr_scheduler_kwargs": {},
        "warmup_epochs": 0.0,
        "use_ema": True,
        "ema_decay": 0.993,
        "ema_tau": 100,
        "ema_update_interval": 1,
        "eval_base_model": False,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "eval_backend": "vernier",
        "eval_max_dets": 500,
        "augmentation_backend": "cpu",
        "run_test": False,
        "num_workers": 0,
        "amp_dtype": None,
    }
    discrepancies = {key: {"expected": value, "actual": train.get(key)} for key, value in required_train.items() if train.get(key) != value}
    if model.get("model_name") != "RFDETRSmall" or model.get("resolution") != 512 or model.get("device") != "mps" or model.get("num_classes") != 1 or discrepancies:
        raise RuntimeError(f"Original RF-DETR reference configuration is not the verified fixed-512 raw recipe: {discrepancies}")
    if [int(entry.get("completed_epoch", -1)) for entry in history] != list(range(1, TOTAL_EPOCHS + 1)):
        raise RuntimeError("Original RF-DETR EMA metric history does not cover completed epochs 1 through 10")
    if float(history[4]["EMA_validation"]["AP_50_95"]) != EXPECTED_ORIGINAL_BEST_AP or float(history[9]["EMA_validation"]["AP_50_95"]) != EXPECTED_ORIGINAL_EPOCH10_AP:
        raise RuntimeError("Original RF-DETR best or epoch-10 AP reference does not match its recorded value")
    return config, history, preview_plan


def validate_preview_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Reuse the exact original nine-image qualitative panel, never test data."""
    examples = plan.get("examples")
    if not isinstance(examples, list) or len(examples) != PREVIEW_COUNT:
        raise RuntimeError("The original expanded preview plan does not contain exactly nine examples")
    all_rows = [row for row in load_manifest() if row["derived_split"] in {"train", "val"}]
    rows_by_path = {row["source_image_path"]: row for row in all_rows}
    groups: set[str] = set()
    split_counts = {"train": 0, "val": 0}
    for example in examples:
        if not isinstance(example, dict):
            raise RuntimeError("The original preview plan contains a non-object example")
        source = example.get("source_image_path")
        if not isinstance(source, str) or source not in rows_by_path:
            raise RuntimeError("The original preview plan refers to a non-frozen-train/validation source")
        row = rows_by_path[source]
        split_counts[row["derived_split"]] += 1
        group = row["video_group"]
        if group in groups:
            raise RuntimeError("The original expanded preview plan does not keep its video groups distinct")
        groups.add(group)
    if len(groups) != PREVIEW_COUNT:
        raise RuntimeError("The original expanded preview panel has fewer than nine distinct video groups")
    return {
        "exact_original_plan_reused": True,
        "source": str(ORIGINAL_PREVIEW_PLAN.relative_to(PROJECT)),
        "source_sha256": digest(ORIGINAL_PREVIEW_PLAN),
        "examples": examples,
        "distinct_video_groups": len(groups),
        "source_split_counts": split_counts,
        "scope_note": "This is the exact existing expanded qualitative panel. It contains frozen train/validation images only and is separate from validation metrics, which always use all 1,082 frozen validation images.",
    }


def expected_candidate_config(reference: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Create a pre-training comparison artifact with only approved differences."""
    candidate = copy.deepcopy(reference)
    train = candidate["train_config"]
    train["dataset_dir"] = str(run_dir / "raw_coco_adapter")
    train["output_dir"] = str(run_dir / "training_output")
    train["resume"] = None
    train["scale_jitter"] = False
    train["notes"] = {
        "purpose": "Fresh RF-DETR Small 10-epoch raw-image scale_jitter=False ablation",
        "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent",
        "enhancement_input": "not used",
    }
    return candidate


def config_differences(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Return all reference/candidate train and model differences."""
    ref_train, new_train = reference.get("train_config", {}), candidate.get("train_config", {})
    ref_model, new_model = reference.get("model_config", {}), candidate.get("model_config", {})
    train_diffs = {
        key: {"reference": ref_train.get(key), "candidate": new_train.get(key)}
        for key in sorted(set(ref_train).union(new_train))
        if ref_train.get(key) != new_train.get(key)
    }
    model_diffs = {
        key: {"reference": ref_model.get(key), "candidate": new_model.get(key)}
        for key in sorted(set(ref_model).union(new_model))
        if ref_model.get(key) != new_model.get(key)
    }
    return {"train_config": train_diffs, "model_config": model_diffs}


def audit_expected_differences(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail before training unless the prospective difference set is exactly approved."""
    differences = config_differences(reference, candidate)
    expected = {"dataset_dir", "output_dir", "resume", "notes", "scale_jitter"}
    unexpected = set(differences["train_config"]).difference(expected)
    if unexpected or differences["model_config"]:
        raise RuntimeError(f"Prospective ablation changes settings beyond the approved set: train={sorted(unexpected)}, model={differences['model_config']}")
    if differences["train_config"].get("scale_jitter") != {"reference": True, "candidate": False}:
        raise RuntimeError("The prospective ablation does not set scale_jitter from True to False")
    return {
        "reference_config": str(ORIGINAL_CONFIG.relative_to(PROJECT)),
        "reference_config_sha256": digest(ORIGINAL_CONFIG),
        "approved_train_config_differences": differences["train_config"],
        "model_config_differences": differences["model_config"],
        "behavioural_change": {
            "field": "train_config.scale_jitter",
            "reference": True,
            "candidate": False,
            "meaning": "This tests the combined effect of removing the stochastic resize-crop-resize branch and its associated scale changes. It does not isolate crop removal from scale changes or prove that either mechanism alone is harmful.",
        },
        "fresh_start_note": "resume=None is intentional and non-comparative: the experiment starts from the same official pretrained checkpoint rather than a trained epoch-5/10 checkpoint.",
        "path_note": "dataset_dir and output_dir differ only because this is a separate read-only adapter and output directory.",
        "notes_note": "notes differ only to identify this ablation.",
    }


class EpochTelemetry(Callback):
    """Record timing and MPS allocator observations for every completed epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.starts: dict[int, tuple[float, dict[str, int | None]]] = {}
        self.records: list[dict[str, Any]] = []

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        torch.mps.synchronize()
        self.starts[int(trainer.current_epoch)] = (time.perf_counter(), mps_memory())

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        torch.mps.synchronize()
        epoch = int(trainer.current_epoch)
        started, before = self.starts.pop(epoch)
        self.records.append(
            {
                "completed_epoch": epoch + 1,
                "global_step": int(trainer.global_step),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "mps_memory_before_epoch": before,
                "mps_memory_after_epoch": mps_memory(),
            }
        )


class MPSExecutionAudit(Callback):
    """Fail clearly if the training module is not entirely on MPS."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs_checked: list[int] = []

    @staticmethod
    def verify(pl_module: Any) -> None:
        devices = {parameter.device.type for parameter in pl_module.parameters()}
        if devices != {"mps"}:
            raise RuntimeError(f"RF-DETR training module is not entirely on MPS: {sorted(devices)}")

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)
        self.epochs_checked.append(int(trainer.current_epoch) + 1)


def audit_full_checkpoint(path: Path, expected_epoch: int, expected_step: int) -> dict[str, Any]:
    """Verify one saved checkpoint preserves model, optimizer, scheduler, and EMA state."""
    if not path.is_file():
        raise RuntimeError(f"Expected full-state checkpoint was not created: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} is not a Lightning checkpoint dictionary")
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    missing = sorted(required.difference(payload))
    if missing or int(payload["epoch"]) != expected_epoch or int(payload["global_step"]) != expected_step:
        raise RuntimeError(f"{path.name} is not the expected completed-epoch full state: missing={missing}, epoch={payload.get('epoch')}, step={payload.get('global_step')}")
    optimizer_states = payload["optimizer_states"]
    scheduler_states = payload["lr_schedulers"]
    callbacks = payload["callbacks"]
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1 or not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise RuntimeError(f"{path.name} lacks exactly one optimizer and LR scheduler state")
    optimizer = optimizer_states[0]
    slots = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(slots, dict) or not slots:
        raise RuntimeError(f"{path.name} lacks populated optimizer state")
    steps = {int(value["step"].item()) if torch.is_tensor(value["step"]) else int(value["step"]) for value in slots.values() if isinstance(value, dict) and "step" in value}
    scheduler = scheduler_states[0]
    if not isinstance(scheduler, dict) or steps != {expected_step} or int(scheduler.get("last_epoch", -1)) != expected_step:
        raise RuntimeError(f"{path.name} optimizer or scheduler state does not match step {expected_step}")
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict) or not average or int(ema.get("latest_update_step", -1)) != expected_step:
        raise RuntimeError(f"{path.name} lacks EMA state at step {expected_step}")
    return {
        "path": str(path.relative_to(PROJECT)),
        "sha256": digest(path),
        "completed_epoch": expected_epoch + 1,
        "global_step": expected_step,
        "optimizer_state_slots": len(slots),
        "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler.get("_step_count", -1))},
        "ema_state_keys": len(average),
    }


def parse_ema_metric_history(metrics_path: Path) -> list[dict[str, Any]]:
    """Require one EMA validation record, loss record, and LR record per epoch."""
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    loss_rows = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    expected_epochs = set(range(TOTAL_EPOCHS))
    if set(validation_rows) != expected_epochs or set(loss_rows) != expected_epochs:
        raise RuntimeError(f"Expected EMA validation and training-loss rows for epochs 0-9; got validation={sorted(validation_rows)}, losses={sorted(loss_rows)}")
    history: list[dict[str, Any]] = []
    validation_keys = {
        "AP_50_95": "val/mAP_50_95", "AP50": "val/mAP_50", "AP75": "val/mAP_75", "mAR": "val/mAR",
        "F1_optimized_threshold": "val/F1", "precision_optimized_threshold": "val/precision", "recall_optimized_threshold": "val/recall",
    }
    for epoch in range(TOTAL_EPOCHS):
        validation, loss = validation_rows[epoch], loss_rows[epoch]
        lr_rows = [row for row in rows if int(row["epoch"]) == epoch and row.get("train/lr") not in (None, "")]
        if not lr_rows:
            raise RuntimeError(f"metrics.csv lacks a learning-rate record for epoch {epoch + 1}")
        lr = lr_rows[-1]
        history.append(
            {
                "completed_epoch": epoch + 1,
                "global_step": int(validation["step"]) + 1,
                "validation_weights": "EMA",
                "EMA_validation": {name: numeric(validation, key) for name, key in validation_keys.items()},
                "learning_rates": {name.removeprefix("train/"): numeric(lr, name) for name in ("train/lr", "train/lr_min", "train/lr_max")},
                "training_losses": {key.removeprefix("train/"): numeric(loss, key) for key in loss if key.startswith("train/") and loss[key] not in (None, "")},
                "threshold_note": "The F1/precision/recall values are RF-DETR's separately optimized validation confidence sweep for this epoch, not the shared 0.43 comparison threshold.",
            }
        )
    return history


def create_alias(alias: Path, target: Path) -> dict[str, str]:
    """Create an unambiguous, run-local link to a preserved output checkpoint."""
    if alias.exists() or alias.is_symlink():
        raise RuntimeError(f"Refusing to overwrite checkpoint alias: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(os.path.relpath(target, alias.parent))
    if alias.resolve() != target.resolve():
        raise RuntimeError(f"Checkpoint alias does not resolve to its selected target: {alias}")
    return {"alias": str(alias.relative_to(PROJECT)), "target": str(target.relative_to(PROJECT)), "target_sha256": digest(target)}


def load_ema_model(full_checkpoint: Path, device: torch.device) -> Any:
    """Load EMA weights from a full Lightning checkpoint for evaluation/previews."""
    from rfdetr import RFDETRSmall

    payload = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
    callbacks = payload.get("callbacks") if isinstance(payload, dict) else None
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict):
        raise RuntimeError(f"{full_checkpoint.name} has no EMA state for evaluation")
    ema_state = {key.removeprefix("module.model."): value for key, value in average.items() if key != "n_averaged"}
    model = RFDETRSmall.from_checkpoint(str(full_checkpoint), device="mps")
    core = model.model.model
    if core is None:
        raise RuntimeError(f"{full_checkpoint.name} did not construct an RF-DETR model")
    incompatible = core.load_state_dict(ema_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Could not load EMA state from {full_checkpoint.name}")
    core.to(device).eval()
    if any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError(f"EMA model from {full_checkpoint.name} is not entirely on MPS")
    return model


def load_inference_model(checkpoint: Path, device: torch.device) -> Any:
    """Load an RF-DETR inference checkpoint and make its neural module MPS-resident."""
    from rfdetr import RFDETRSmall

    model = RFDETRSmall.from_checkpoint(str(checkpoint), device="mps")
    core = model.model.model
    if core is None:
        raise RuntimeError(f"{checkpoint.name} did not construct an RF-DETR model")
    core.to(device).eval()
    if model.model_config.resolution != 512 or model.model_config.num_classes != 1 or any(parameter.device.type != "mps" for parameter in core.parameters()):
        raise RuntimeError(f"Inference model {checkpoint.name} is not a one-class 512px MPS RF-DETR Small")
    return model


def shared_threshold_metrics(matching: dict[int, dict[str, Any]]) -> dict[str, float]:
    """Compute F1/precision/recall at the predeclared shared threshold only."""
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds

    ids = sorted(matching)
    if 0 not in ids:
        raise RuntimeError("One-class foreground label 0 is missing from matching data")
    classes_with_gt = [index for index, class_id in enumerate(ids) if matching[class_id]["total_gt"] > 0]
    if classes_with_gt != [ids.index(0)]:
        raise RuntimeError("Only foreground label 0 may have validation ground truth")
    result = sweep_confidence_thresholds([matching[class_id] for class_id in ids], [SHARED_PR_THRESHOLD], classes_with_gt)[0]
    return {
        "threshold": SHARED_PR_THRESHOLD,
        "F1": float(result["macro_f1"]),
        "precision": float(result["macro_precision"]),
        "recall": float(result["macro_recall"]),
    }


def evaluate_ema_checkpoint(checkpoint: Path, checkpoint_type: str, label: str, device: torch.device, output_dir: Path) -> dict[str, Any]:
    """Use the validated epoch-5 evaluator path for any selected EMA checkpoint."""
    from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    images, annotations_by_image = check_coco_payload(payload, rows, "valid")
    if len(images) != 1082:
        raise RuntimeError("Per-size evaluation must use exactly 1,082 frozen validation images")
    model = load_ema_model(checkpoint, device) if checkpoint_type == "full_state_ema" else load_inference_model(checkpoint, device)
    core = model.model.model
    assert core is not None
    metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 500], backend="vernier", sync_on_compute=False)
    matching: dict[int, dict[str, Any]] = {}
    slot_count = 0
    background_slots = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for image_id in sorted(images):
            image = images[image_id]
            source_path = PROJECT / str(image["file_name"])
            if not source_path.is_file() or not source_path.is_relative_to(PROJECT / "datasets/trashcan/material_version"):
                raise RuntimeError(f"Per-size evaluation source is not a frozen raw material_version image: {source_path}")
            tensor, (height, width) = validation_tensor(source_path, model.means, model.stds)
            if (height, width) != (int(image["height"]), int(image["width"])):
                raise RuntimeError(f"Frozen validation dimensions changed: {source_path}")
            raw_outputs = core(tensor.unsqueeze(0).to(device))
            if isinstance(raw_outputs, tuple):
                raw_outputs = {"pred_logits": raw_outputs[1], "pred_boxes": raw_outputs[0]}
            result = model.model.postprocess(raw_outputs, target_sizes=torch.tensor([[height, width]], dtype=torch.int64, device=device))[0]
            if any(parameter.device.type != "mps" for parameter in core.parameters()):
                raise RuntimeError("Per-size evaluation left MPS; CPU model fallback is forbidden")
            boxes = result["boxes"].detach().cpu().to(dtype=torch.float32)
            scores = result["scores"].detach().cpu().to(dtype=torch.float32)
            labels = result["labels"].detach().cpu().to(dtype=torch.int64)
            if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.numel() or scores.numel() != labels.numel() or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not set(labels.tolist()).issubset({0, 1}):
                raise RuntimeError(f"Invalid MPS postprocess result for validation image {image_id}")
            annotations = annotations_by_image.get(image_id, [])
            gt_boxes = torch.tensor([[float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][0]) + float(a["bbox"][2]), float(a["bbox"][1]) + float(a["bbox"][3])] for a in annotations], dtype=torch.float32) if annotations else torch.empty((0, 4), dtype=torch.float32)
            target = {"boxes": gt_boxes, "labels": torch.zeros((len(annotations),), dtype=torch.int64)}
            prediction = {"boxes": boxes, "scores": scores, "labels": labels}
            metric.update([prediction], [target])
            merge_matching_data(matching, build_matching_data([prediction], [target], iou_threshold=0.5, iou_type="bbox"))
            slot_count += int(labels.numel())
            background_slots += int((labels == 1).sum().item())
    torch.mps.synchronize()
    result = {
        "checkpoint": {"path": str(checkpoint.relative_to(PROJECT)), "sha256": digest(checkpoint), "type": checkpoint_type},
        "validation_weights": "EMA",
        "device": "mps",
        "validation_images": len(images),
        "ground_truth_object_counts": {"total": 289, "COCO_area_buckets": coco_area_counts(annotations_by_image)},
        "training_compatible_ema_metrics": summarize_training_metric(metric.compute()),
        "precision_recall_iou50_shared_threshold": shared_threshold_metrics(matching),
        "precision_recall_iou50_separately_optimized_threshold": training_style_f1(matching),
        "prediction_slots": {"all": slot_count, "background_label_1": background_slots, "foreground_label_0": slot_count - background_slots},
        "validation_path": "PIL RGB -> Resize(512,512,BILINEAR,antialias=True) -> scale [0,1] -> checkpoint Normalize -> unthresholded official PostProcess -> pinned RF-DETR vernier metric backend",
        "timing_seconds": round(time.perf_counter() - started, 3),
        "memory": mps_memory(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{label}.json", result)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return result


def generate_previews(run_dir: Path, plan: dict[str, Any], checkpoints: dict[int, Path], device: torch.device) -> list[dict[str, Any]]:
    """Generate the exact original nine-image panel for every newly trained EMA epoch."""
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, rows, "valid")
    images_by_path = {str(image["file_name"]): image for image in images.values()}
    adapter_info = {"fixed_validation_previews": plan["examples"]}
    reports: list[dict[str, Any]] = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        checkpoint = checkpoints[epoch]
        model = load_ema_model(checkpoint, device)
        preview_root = run_dir / "fixed_validation_previews" / f"epoch_{epoch:03d}_ema"
        previews = create_labeled_previews(model, adapter_info, annotations, images_by_path, preview_root, device)
        reports.append({"completed_epoch": epoch, "validation_weights": "EMA", "checkpoint": str(checkpoint.relative_to(PROJECT)), "checkpoint_sha256": digest(checkpoint), "previews": previews})
        del model
        gc.collect()
        torch.mps.empty_cache()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    args = parser.parse_args()
    if args.epochs != TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.seed != SEED or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This ablation requires --epochs 10 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    required = (FREEZE_FILE, TRAIN_COCO, VAL_COCO, CLASS_MAPPING, RFDETR_SOURCE)
    if not all(path.exists() for path in required):
        raise SystemExit("Frozen TrashCan v2 artifacts and pinned RF-DETR source are required")
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse an existing experiment directory: {run_dir}")
    expected_cache = (PROJECT / ".cache/rfdetr").resolve()
    if Path(os.environ.get("RF_HOME", "")).resolve() != expected_cache:
        raise SystemExit("RF_HOME must be set to this project's .cache/rfdetr directory")
    device = require_mps()
    reference_config, original_history, original_plan = original_reference()
    preview_plan = validate_preview_plan(original_plan)
    intended_config = expected_candidate_config(reference_config, run_dir)
    preflight_audit = audit_expected_differences(reference_config, intended_config)
    entries = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, entries) for path in (TRAIN_COCO, VAL_COCO, CLASS_MAPPING)}

    import rfdetr
    import rfdetr.training as training_stack
    from rfdetr import RFDETRSmall
    from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir, validate_pretrain_weights

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The installed RF-DETR package is not the pinned project-local official source")
    if Path(get_model_cache_dir()).resolve() != expected_cache:
        raise RuntimeError("RF-DETR did not resolve pretrained weights inside the project cache")
    official_asset = ModelWeights.from_filename("rf-detr-small.pth")
    if official_asset is None or official_asset.md5_hash is None:
        raise RuntimeError("Pinned RF-DETR source does not expose the official Small pretrained-weight checksum")
    pretrained_path = expected_cache / "rf-detr-small.pth"
    if not pretrained_path.is_file() or digest(pretrained_path) != EXPECTED_PRETRAIN_SHA256 or not validate_pretrain_weights(str(pretrained_path), strict=True):
        raise RuntimeError("The original MD5-verified official RF-DETR Small pretrained checkpoint is unavailable or changed")

    run_dir.mkdir(parents=True)
    write_json(run_dir / "configuration_audit_before_training.json", {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(), "configuration_difference_audit": preflight_audit,
        "intended_resolved_config": intended_config,
        "original_baselines": {"best_validation_AP_50_95": EXPECTED_ORIGINAL_BEST_AP, "epoch_10_AP_50_95": EXPECTED_ORIGINAL_EPOCH10_AP},
        "original_pretrained_checkpoint": {"path": str(pretrained_path.relative_to(PROJECT)), "sha256": digest(pretrained_path), "md5": digest(pretrained_path, "md5"), "official_expected_md5": official_asset.md5_hash, "official_url": official_asset.url},
        "frozen_checks": frozen_checks, "preview_plan": preview_plan,
    })
    adapter_info = build_adapter(run_dir, PREVIEW_COUNT)
    adapter_info["fixed_validation_previews"] = preview_plan["examples"]
    adapter_info["description"] = "Read-only symlink adapter over frozen raw TrashCan material_version train/validation images. Test and enhanced images are absent."
    write_json(run_dir / "preview_plan.json", preview_plan)

    # Match the original pretrained initialization sequence before model construction.
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    model = RFDETRSmall(resolution=512, device="mps")
    configured_pretrained = Path(model.model_config.pretrain_weights)
    if configured_pretrained.resolve() != pretrained_path or not validate_pretrain_weights(str(configured_pretrained), strict=True):
        raise RuntimeError("Fresh RF-DETR initialization did not use the original official pretrained checkpoint")
    telemetry = EpochTelemetry()
    mps_audit = MPSExecutionAudit()
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_audits(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.extend((telemetry, mps_audit))
        return trainer

    output_dir = run_dir / "training_output"
    launched = time.perf_counter()
    memory_before = mps_memory()
    training_stack.build_trainer = build_trainer_with_audits
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512,
            optimizer="adamw", lr=0.0001, lr_encoder=0.00015, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step", lr_scheduler_kwargs={}, warmup_epochs=0.0,
            grad_accum_steps=1, amp_dtype=None, augmentation_backend="cpu", multi_scale="off", expanded_scales=False,
            scale_jitter=False, square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False, compute_val_loss="auto",
            log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False, wandb=False, early_stopping=False,
            run_test=False, seed=args.seed,
            notes={"purpose": "Fresh RF-DETR Small raw-image scale_jitter=False ablation", "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used"},
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    if mps_audit.epochs_checked != list(range(1, TOTAL_EPOCHS + 1)):
        raise RuntimeError(f"MPS execution was not verified for every epoch: {mps_audit.epochs_checked}")
    if digest(pretrained_path) != EXPECTED_PRETRAIN_SHA256:
        raise RuntimeError("The original pretrained checkpoint changed during training")
    resolved_config_path = output_dir / "training_config.json"
    if not resolved_config_path.is_file():
        raise RuntimeError("RF-DETR did not write a resolved training configuration")
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    resolved_differences = config_differences(reference_config, resolved_config)
    expected_keys = {"dataset_dir", "output_dir", "resume", "notes", "scale_jitter"}
    unexpected = set(resolved_differences["train_config"]).difference(expected_keys)
    if unexpected or resolved_differences["model_config"] or resolved_config.get("train_config", {}).get("scale_jitter") is not False:
        raise RuntimeError(f"Resolved ablation configuration changed settings beyond the approved set: train={sorted(unexpected)}, model={resolved_differences['model_config']}")
    write_json(run_dir / "configuration_audit_after_training.json", {"status": "passed", "reference_config": str(ORIGINAL_CONFIG.relative_to(PROJECT)), "resolved_config": str(resolved_config_path.relative_to(PROJECT)), "differences": resolved_differences, "verified_only_training_behaviour_change": "scale_jitter: True -> False", "combined_change_note": preflight_audit["behavioural_change"]["meaning"]})

    checkpoints: dict[int, Path] = {}
    checkpoint_audits: list[dict[str, Any]] = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        checkpoint = output_dir / f"checkpoint_{epoch - 1}.ckpt"
        checkpoints[epoch] = checkpoint
        checkpoint_audits.append(audit_full_checkpoint(checkpoint, epoch - 1, epoch * STEPS_PER_EPOCH))
    history = parse_ema_metric_history(output_dir / "metrics.csv")
    best_record = max(history, key=lambda item: float(item["EMA_validation"]["AP_50_95"]))
    best_inference = output_dir / "checkpoint_best_ema.pth"
    if not best_inference.is_file():
        raise RuntimeError("RF-DETR did not save the best EMA inference checkpoint")
    latest_full = checkpoints[TOTAL_EPOCHS]
    aliases = {
        "best_validation_AP_50_95_ema": create_alias(run_dir / "checkpoints/best_validation_ap_50_95_ema.pth", best_inference),
        "latest_full_state": create_alias(run_dir / "checkpoints/latest_full_state.ckpt", latest_full),
    }

    del model
    gc.collect()
    torch.mps.empty_cache()
    preview_reports = generate_previews(run_dir, preview_plan, checkpoints, device)
    per_size_dir = run_dir / "per_size_evaluations"
    per_size = {
        "original_best_epoch5_ema": evaluate_ema_checkpoint(ORIGINAL_EPOCH5_BEST_EMA, "inference_ema", "original_best_epoch5_ema", device, per_size_dir),
        "original_epoch10_ema": evaluate_ema_checkpoint(ORIGINAL_EPOCH10_FULL, "full_state_ema", "original_epoch10_ema", device, per_size_dir),
        "ablation_best_ema": evaluate_ema_checkpoint(best_inference, "inference_ema", "ablation_best_ema", device, per_size_dir),
        "ablation_epoch10_ema": evaluate_ema_checkpoint(latest_full, "full_state_ema", "ablation_epoch10_ema", device, per_size_dir),
    }
    torch.mps.synchronize()
    comparisons = {
        "best_validation_AP_50_95_within_epochs_1_to_10": {
            "original": EXPECTED_ORIGINAL_BEST_AP,
            "ablation": float(best_record["EMA_validation"]["AP_50_95"]),
            "ablation_epoch": int(best_record["completed_epoch"]),
            "ablation_minus_original": float(best_record["EMA_validation"]["AP_50_95"]) - EXPECTED_ORIGINAL_BEST_AP,
        },
        "epoch_10_AP_50_95": {
            "original": EXPECTED_ORIGINAL_EPOCH10_AP,
            "ablation": float(history[-1]["EMA_validation"]["AP_50_95"]),
            "ablation_minus_original": float(history[-1]["EMA_validation"]["AP_50_95"]) - EXPECTED_ORIGINAL_EPOCH10_AP,
        },
        "per_size_and_shared_threshold_evaluations": per_size,
        "shared_threshold_note": "Precision/recall/F1 under precision_recall_iou50_shared_threshold use exactly 0.43 for every selected checkpoint. Separately optimized threshold metrics are reported beside them but must not be substituted for the shared-threshold comparison.",
    }
    result = {
        "status": "passed", "run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fresh 10-epoch RF-DETR Small scale_jitter=False ablation on frozen raw TrashCan train/validation only. Test, enhanced, source, and website files were not used or changed.",
        "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "source": {"rfdetr_version": metadata.version("rfdetr"), "source_revision": git_revision(RFDETR_SOURCE), "pretrained_sha256": digest(pretrained_path)},
        "frozen_checks": frozen_checks, "configuration_audits": {"before": str((run_dir / "configuration_audit_before_training.json").relative_to(PROJECT)), "after": str((run_dir / "configuration_audit_after_training.json").relative_to(PROJECT))},
        "combined_augmentation_change": preflight_audit["behavioural_change"], "EMA_metric_history": history,
        "MPS_execution_audit": {"epochs_checked": mps_audit.epochs_checked, "cpu_fallback_disabled": True}, "epoch_telemetry": telemetry.records,
        "full_checkpoint_audits": checkpoint_audits, "checkpoint_aliases": aliases,
        "selected_best": {"completed_epoch": int(best_record["completed_epoch"]), "AP_50_95": float(best_record["EMA_validation"]["AP_50_95"]), "checkpoint": str(best_inference.relative_to(PROJECT)), "checkpoint_sha256": digest(best_inference)},
        "fixed_previews": preview_reports, "comparisons": comparisons,
        "timing_seconds": {"training_and_reporting_total": round(time.perf_counter() - launched, 3)},
        "memory": {"before_training": memory_before, "after_reporting": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(run_dir / "result.json", result)
    write_json(run_dir / "metric_history_ema.json", history)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
