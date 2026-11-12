#!/usr/bin/env python3
"""Continue the frozen raw-image RF-DETR Small run from epoch 5 to epoch 10.

This runner is deliberately a *full-state* continuation.  It resumes the
epoch-5 Lightning checkpoint containing the model, optimizer, LR scheduler,
loop state, and RF-DETR EMA callback state.  It creates a new read-only COCO
adapter over the frozen raw train/validation split; no test or enhanced image
is mounted in that adapter.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only continuation.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from pytorch_lightning import Callback

from run_rfdetr_raw_v2_pilot import (
    CLASS_MAPPING,
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


SOURCE_RUN_ID = "rfdetr_raw_v2_e1to5_fixed512_mps_20261001T091117Z"
EXPECTED_EPOCH5_CHECKPOINT_SHA256 = "0e77c5d93a990a8df810f4de1f7b781483aceba33ffb1bac21b267d200f97c0d"
STEPS_PER_EPOCH = 5048
SOURCE_COMPLETED_EPOCHS = 5
TOTAL_EPOCHS = 10
PREVIEW_COUNT = 9
EXPECTED_CATEGORY_NAME = "visible_plastic_debris"


def number(row: dict[str, str], key: str) -> float:
    """Read one finite scalar written by the pinned RF-DETR metrics logger."""
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"metrics.csv lacks {key}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"metrics.csv recorded non-finite {key}")
    return result


def step_as_int(value: Any) -> int:
    """Convert an AdamW optimizer step counter to a Python integer."""
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def source_paths(source_run_id: str) -> dict[str, Path]:
    """Locate immutable epoch-5 artifacts without writing into their run folder."""
    source_dir = RUNS_ROOT / source_run_id
    output_dir = source_dir / "training_output"
    paths = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "latest_full_state_alias": source_dir / "checkpoints/latest_full_state.ckpt",
        "best_ema": source_dir / "checkpoints/best_validation_ap_50_95_ema.pth",
        "result": source_dir / "continuation_result.json",
        "history": source_dir / "metric_history.json",
        "config": output_dir / "training_config.json",
    }
    missing = [str(path.relative_to(PROJECT)) for key, path in paths.items() if key not in {"source_dir", "output_dir"} and not path.is_file()]
    if missing:
        raise RuntimeError(f"The epoch-5 source run is incomplete; missing: {missing}")
    if paths["latest_full_state_alias"].resolve() != (output_dir / "checkpoint_4.ckpt").resolve():
        raise RuntimeError("The supplied latest_full_state alias must resolve to the preserved epoch-5 checkpoint_4.ckpt")
    return paths


def inspect_full_checkpoint(path: Path, *, expected_epoch: int, expected_step: int, require_epoch5_sha: bool = False) -> dict[str, Any]:
    """Verify all state needed for a faithful RF-DETR Lightning resume."""
    checkpoint_sha256 = digest(path)
    if require_epoch5_sha and checkpoint_sha256 != EXPECTED_EPOCH5_CHECKPOINT_SHA256:
        raise RuntimeError("The supplied epoch-5 full checkpoint SHA-256 differs from the preserved verified checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} is not a dictionary Lightning checkpoint")
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    absent = sorted(required.difference(payload))
    if absent:
        raise RuntimeError(f"{path.name} is weights-only or otherwise lacks full resume state: {absent}")
    if int(payload["epoch"]) != expected_epoch or int(payload["global_step"]) != expected_step:
        raise RuntimeError(
            f"{path.name} reports epoch/step {payload['epoch']}/{payload['global_step']}, expected {expected_epoch}/{expected_step}"
        )
    state_dict = payload["state_dict"]
    optimizer_states = payload["optimizer_states"]
    scheduler_states = payload["lr_schedulers"]
    callbacks = payload["callbacks"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"{path.name} has no model state")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1 or not isinstance(optimizer_states[0], dict):
        raise RuntimeError(f"{path.name} does not contain exactly one optimizer state")
    optimizer = optimizer_states[0]
    slots = optimizer.get("state")
    if not isinstance(slots, dict) or not slots:
        raise RuntimeError(f"{path.name} has no populated optimizer state slots")
    steps = {step_as_int(value["step"]) for value in slots.values() if isinstance(value, dict) and "step" in value}
    if steps != {expected_step}:
        raise RuntimeError(f"{path.name} optimizer steps are {sorted(steps)}, expected {expected_step}")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1 or not isinstance(scheduler_states[0], dict):
        raise RuntimeError(f"{path.name} does not contain exactly one scheduler state")
    scheduler = scheduler_states[0]
    if int(scheduler.get("last_epoch", -1)) != expected_step or int(scheduler.get("_step_count", -1)) != expected_step + 1:
        raise RuntimeError(f"{path.name} scheduler state does not match global step {expected_step}")
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict) or not average:
        raise RuntimeError(f"{path.name} does not contain resumable EMA state")
    if int(ema.get("latest_update_step", -1)) != expected_step:
        raise RuntimeError(f"{path.name} EMA update step does not match global step {expected_step}")
    return {
        "path": str(path.resolve().relative_to(PROJECT)),
        "sha256": checkpoint_sha256,
        "epoch_zero_based": expected_epoch,
        "completed_epoch": expected_epoch + 1,
        "global_step": expected_step,
        "model_state_keys": len(state_dict),
        "optimizer": {"state_slots": len(slots), "param_groups": len(optimizer.get("param_groups", [])), "steps": sorted(steps)},
        "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler["_step_count"])},
        "ema": {"latest_update_step": int(ema["latest_update_step"]), "state_keys": len(average)},
        "rng_state_present": any(key in payload for key in ("rng_state", "rng_states", "python_rng_state", "numpy_rng_state")),
    }


def scheduler_record(source_config: dict[str, Any], source_audit: dict[str, Any]) -> dict[str, Any]:
    """Record the saved scheduler and the limited consequence of a 10-epoch target."""
    train = source_config.get("train_config", {})
    expected = {
        "lr_scheduler": "step",
        "lr_scheduler_interval": "step",
        "lr_scheduler_kwargs": {},
        "warmup_epochs": 0.0,
        "lr": 0.0001,
        "lr_encoder": 0.00015,
        "lr_component_decay": 0.7,
        "lr_vit_layer_decay": 0.8,
    }
    mismatches = {key: {"expected": value, "actual": train.get(key)} for key, value in expected.items() if train.get(key) != value}
    if int(train.get("epochs", -1)) != SOURCE_COMPLETED_EPOCHS or mismatches:
        raise RuntimeError(f"The epoch-5 source scheduler/configuration is not the verified fixed-512 setup: {mismatches}")
    return {
        "source": {"epochs": SOURCE_COMPLETED_EPOCHS, "total_optimizer_steps": source_audit["global_step"], **expected, "managed_defaults": {"lr_drop": 100, "min_factor": 0.0}},
        "continuation": {"epochs": TOTAL_EPOCHS, "total_optimizer_steps": STEPS_PER_EPOCH * TOTAL_EPOCHS, "resume_global_step": source_audit["global_step"]},
        "effect_of_extending_epoch_budget": {
            "total_steps_changes": True,
            "from": source_audit["global_step"],
            "to": STEPS_PER_EPOCH * TOTAL_EPOCHS,
            "effective_learning_rate_change_for_epochs_6_to_10": False,
            "reason": "The pinned managed step scheduler has warmup_epochs=0 and its first 10x drop is at epoch 100. The restored scheduler/optimizer counters continue from step 25,240, and no scheduled rate drop occurs through completed epoch 10.",
        },
    }


def verify_preserved_configuration(source: dict[str, Any], continued: dict[str, Any]) -> dict[str, Any]:
    """Fail if a train or model setting changes beyond the documented resume fields."""
    source_train = source.get("train_config")
    continued_train = continued.get("train_config")
    source_model = source.get("model_config")
    continued_model = continued.get("model_config")
    if not all(isinstance(value, dict) for value in (source_train, continued_train, source_model, continued_model)):
        raise RuntimeError("Could not compare source and continuation RF-DETR configurations")
    allowed_train_changes = {"dataset_dir", "epochs", "notes", "output_dir", "resume"}
    train_differences = {
        key: {"source": source_train.get(key), "continuation": continued_train.get(key)}
        for key in sorted(set(source_train).union(continued_train).difference(allowed_train_changes))
        if source_train.get(key) != continued_train.get(key)
    }
    model_differences = {
        key: {"source": source_model.get(key), "continuation": continued_model.get(key)}
        for key in sorted(set(source_model).union(continued_model))
        if source_model.get(key) != continued_model.get(key)
    }
    if train_differences or model_differences:
        raise RuntimeError(f"Continuation changed RF-DETR settings beyond documented resume fields: train={train_differences}, model={model_differences}")
    return {
        "preserved": True,
        "allowed_train_config_changes": sorted(allowed_train_changes),
        "documented_values": {
            "epochs": {"source": source_train["epochs"], "continuation": continued_train["epochs"]},
            "resume": continued_train["resume"],
            "dataset_dir": continued_train["dataset_dir"],
            "output_dir": continued_train["output_dir"],
        },
    }


def coco_size(annotation: dict[str, Any]) -> str:
    """Use COCO's original-pixel area thresholds for preview stratification."""
    width, height = map(float, annotation["bbox"][2:])
    area = width * height
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def fixed_preview_plan() -> dict[str, Any]:
    """Preselect nine validation images before training, across groups/classes/sizes."""
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    images, annotations = check_coco_payload(payload, rows, "valid")
    image_by_path = {str(image["file_name"]): image for image in images.values()}
    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()

    def add(row: dict[str, str], role: str, required_size: str | None) -> None:
        source_path = row["source_image_path"]
        image = image_by_path[source_path]
        image_annotations = annotations.get(int(image["id"]), [])
        selected.append(
            {
                "selection_role": role,
                "required_coco_size": required_size,
                "source_image_path": source_path,
                "video_group": row["video_group"],
                "has_visible_plastic_debris": row["has_visible_plastic_debris"],
                "visible_plastic_debris_instances": int(row["visible_plastic_debris_instances"]),
                "image_dimensions": {"width": int(image["width"]), "height": int(image["height"])},
                "ground_truth_sizes": [coco_size(annotation) for annotation in image_annotations],
                "ground_truth_bboxes_coco_xywh": [annotation["bbox"] for annotation in image_annotations],
            }
        )
        selected_groups.add(row["video_group"])

    # Two positive examples for every COCO size, from distinct groups.  A group
    # is never reused by this panel, so adjacent video frames cannot dominate it.
    for size in ("small", "medium", "large"):
        chosen = 0
        candidates = sorted(
            (row for row in rows if row["has_visible_plastic_debris"] == "True"),
            key=lambda row: (row["video_group"], row["source_image_path"]),
        )
        for row in candidates:
            image = image_by_path[row["source_image_path"]]
            if row["video_group"] in selected_groups or size not in {coco_size(annotation) for annotation in annotations.get(int(image["id"]), [])}:
                continue
            add(row, "positive", size)
            chosen += 1
            if chosen == 2:
                break
        if chosen != 2:
            raise RuntimeError(f"Could not preselect two distinct-group positive validation previews for COCO {size} objects")

    # Three negatives, also from groups not used by the positive size strata.
    chosen_negative = 0
    for row in sorted((row for row in rows if row["has_visible_plastic_debris"] == "False"), key=lambda row: (row["video_group"], row["source_image_path"])):
        if row["video_group"] in selected_groups:
            continue
        add(row, "negative", None)
        chosen_negative += 1
        if chosen_negative == 3:
            break
    if chosen_negative != 3 or len(selected) != PREVIEW_COUNT or len(selected_groups) != PREVIEW_COUNT:
        raise RuntimeError("Could not build the required nine-group fixed validation preview panel")
    return {
        "selected_before_training": True,
        "selection_policy": "Two positive images for each COCO size stratum (small < 32^2 px, medium < 96^2 px, large >= 96^2 px), then three negatives. Candidates are ordered by video_group and source path; no video_group is reused.",
        "summary": {"images": len(selected), "distinct_video_groups": len(selected_groups), "positives": 6, "negatives": 3, "size_strata": {"small": 2, "medium": 2, "large": 2}},
        "examples": selected,
    }


class EpochTelemetry(Callback):
    """Record elapsed time and MPS allocator observations for each new epoch."""

    def __init__(self) -> None:
        super().__init__()
        self._starts: dict[int, tuple[float, dict[str, int | None]]] = {}
        self.records: list[dict[str, Any]] = []

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        torch.mps.synchronize()
        self._starts[int(trainer.current_epoch)] = (time.perf_counter(), mps_memory())

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        torch.mps.synchronize()
        epoch = int(trainer.current_epoch)
        started, before = self._starts.pop(epoch)
        self.records.append(
            {
                "epoch_zero_based": epoch,
                "completed_epoch": epoch + 1,
                "global_step": int(trainer.global_step),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "mps_memory_before_epoch": before,
                "mps_memory_after_epoch": mps_memory(),
            }
        )


class ResumeRuntimeAudit(Callback):
    """Fail before epoch 6 if Lightning did not restore the required live state."""

    def __init__(self, source_audit: dict[str, Any]) -> None:
        super().__init__()
        self._source_audit = source_audit
        self.record: dict[str, Any] | None = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        expected_step = int(self._source_audit["global_step"])
        expected_epoch = int(self._source_audit["epoch_zero_based"]) + 1
        if int(trainer.current_epoch) != expected_epoch or int(trainer.global_step) != expected_step:
            raise RuntimeError(
                "Lightning did not restore the expected epoch/global step before the first continuation batch: "
                f"epoch={trainer.current_epoch}, step={trainer.global_step}, expected={expected_epoch}/{expected_step}"
            )
        if len(trainer.optimizers) != 1:
            raise RuntimeError("The resumed trainer does not have exactly one optimizer")
        optimizer_state = trainer.optimizers[0].state_dict()
        optimizer_slots = optimizer_state.get("state")
        if not isinstance(optimizer_slots, dict) or len(optimizer_slots) != self._source_audit["optimizer"]["state_slots"]:
            raise RuntimeError("Lightning did not restore the expected populated optimizer state")
        optimizer_steps = {step_as_int(value["step"]) for value in optimizer_slots.values() if isinstance(value, dict) and "step" in value}
        if optimizer_steps != {expected_step}:
            raise RuntimeError("Lightning restored an optimizer with an unexpected step counter")
        scheduler_configs = getattr(trainer, "lr_scheduler_configs", [])
        if len(scheduler_configs) != 1:
            raise RuntimeError("The resumed trainer does not have exactly one LR scheduler")
        scheduler_state = scheduler_configs[0].scheduler.state_dict()
        if int(scheduler_state.get("last_epoch", -1)) != expected_step or int(scheduler_state.get("_step_count", -1)) != expected_step + 1:
            raise RuntimeError("Lightning did not restore the expected LR scheduler counter")
        ema_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "RFDETREMACallback"]
        if len(ema_callbacks) != 1:
            raise RuntimeError("The resumed trainer does not have exactly one RF-DETR EMA callback")
        ema = ema_callbacks[0]
        average_model = getattr(ema, "_average_model", None)
        if average_model is None or int(getattr(ema, "_latest_update_step", -1)) != expected_step:
            raise RuntimeError("Lightning did not restore the expected RF-DETR EMA callback state")
        ema_state_keys = len(average_model.state_dict())
        if ema_state_keys != self._source_audit["ema"]["state_keys"]:
            raise RuntimeError("The restored RF-DETR EMA averaged model has an unexpected state dictionary")
        model_state_keys = len(pl_module.state_dict())
        if model_state_keys != self._source_audit["model_state_keys"]:
            raise RuntimeError("The restored RF-DETR model has an unexpected state dictionary")
        self.record = {
            "passed_before_epoch_6": True,
            "model_state_keys": model_state_keys,
            "optimizer": {"state_slots": len(optimizer_slots), "steps": sorted(optimizer_steps)},
            "scheduler": {"last_epoch": int(scheduler_state["last_epoch"]), "step_count": int(scheduler_state["_step_count"])},
            "EMA": {"latest_update_step": int(getattr(ema, "_latest_update_step")), "state_keys": ema_state_keys},
            "epoch_zero_based": int(trainer.current_epoch),
            "completed_epoch_before_continuation": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }


def source_ema_history(path: Path) -> list[dict[str, Any]]:
    """Relabel the first five metrics as EMA, matching the source configuration."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, list) or [int(record.get("completed_epoch", -1)) for record in source] != list(range(1, SOURCE_COMPLETED_EPOCHS + 1)):
        raise RuntimeError("The epoch-5 source metric history is incomplete")
    records: list[dict[str, Any]] = []
    for record in source:
        validation = record.get("validation")
        losses = record.get("losses")
        if not isinstance(validation, dict) or not isinstance(losses, dict):
            raise RuntimeError("The epoch-5 source metric history lacks validation or loss values")
        values = {key: validation.get(key) for key in ("AP_50_95", "AP50", "AP75", "mAR")}
        if any(value is None or not math.isfinite(float(value)) for value in values.values()):
            raise RuntimeError("The epoch-5 source metric history has invalid AP/mAR values")
        records.append(
            {
                "completed_epoch": int(record["completed_epoch"]),
                "global_step": int(record["global_step"]),
                "source": "preserved_epoch_1_to_5_run",
                "validation_weights": "EMA",
                "EMA_validation": {key: float(value) for key, value in values.items()},
                "training_losses": {key: float(value) for key, value in losses.items() if isinstance(value, (int, float)) and math.isfinite(float(value))},
            }
        )
    return records


def full_ema_history(source_history: list[dict[str, Any]], metrics_path: Path) -> list[dict[str, Any]]:
    """Add explicit EMA validation metrics for epochs 6 through 10."""
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    loss_rows = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    expected = set(range(SOURCE_COMPLETED_EPOCHS, TOTAL_EPOCHS))
    if set(validation_rows) != expected or set(loss_rows) != expected:
        raise RuntimeError(f"Continuation must log validation and loss rows for zero-based epochs {sorted(expected)}; got validation={sorted(validation_rows)}, losses={sorted(loss_rows)}")
    validation_keys = {
        "AP_50_95": "val/mAP_50_95",
        "AP50": "val/mAP_50",
        "AP75": "val/mAP_75",
        "mAR": "val/mAR",
        "F1": "val/F1",
        "precision": "val/precision",
        "recall": "val/recall",
    }
    records = list(source_history)
    for epoch in range(SOURCE_COMPLETED_EPOCHS, TOTAL_EPOCHS):
        validation = validation_rows[epoch]
        losses = loss_rows[epoch]
        records.append(
            {
                "completed_epoch": epoch + 1,
                "global_step": int(validation["step"]) + 1,
                "source": "epoch_5_to_10_continuation",
                "validation_weights": "EMA",
                "EMA_validation": {name: number(validation, metric_key) for name, metric_key in validation_keys.items()},
                "training_losses": {key.removeprefix("train/"): number(losses, key) for key in losses if key.startswith("train/") and losses[key] not in (None, "")},
            }
        )
    return records


def load_ema_model(checkpoint_path: Path, device: torch.device) -> Any:
    """Load EMA weights embedded in a full checkpoint for labelled MPS previews."""
    from rfdetr import RFDETRSmall

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    callback_state = payload.get("callbacks")
    ema = callback_state.get("RFDETREMACallback") if isinstance(callback_state, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict):
        raise RuntimeError(f"{checkpoint_path.name} has no EMA state for preview generation")
    ema_state = {key.removeprefix("module.model."): value for key, value in average.items() if key != "n_averaged"}
    model = RFDETRSmall.from_checkpoint(str(checkpoint_path), device=str(device))
    incompatible = model.model.model.load_state_dict(ema_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys or model.model.device.type != "mps":
        raise RuntimeError(f"Could not load the MPS EMA model from {checkpoint_path.name}")
    return model


def generate_fixed_ema_previews(run_dir: Path, preview_plan: dict[str, Any], checkpoints: dict[int, Path], device: torch.device) -> list[dict[str, Any]]:
    """Make matching preselected raw/GT/EMA-prediction panels for epochs 5–10."""
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, rows, "valid")
    images_by_path = {str(image["file_name"]): image for image in images.values()}
    adapter_info = {"fixed_validation_previews": preview_plan["examples"]}
    reports: list[dict[str, Any]] = []
    for completed_epoch in range(SOURCE_COMPLETED_EPOCHS, TOTAL_EPOCHS + 1):
        checkpoint = checkpoints[completed_epoch]
        torch.mps.empty_cache()
        model = load_ema_model(checkpoint, device)
        preview_root = run_dir / "fixed_validation_previews" / f"epoch_{completed_epoch:03d}_ema"
        previews = create_labeled_previews(model, adapter_info, annotations, images_by_path, preview_root, device)
        reports.append(
            {
                "completed_epoch": completed_epoch,
                "validation_weights": "EMA",
                "checkpoint": str(checkpoint.relative_to(PROJECT)),
                "checkpoint_sha256": digest(checkpoint),
                "previews": previews,
            }
        )
        del model
        gc.collect()
        torch.mps.empty_cache()
    return reports


def create_alias(alias: Path, target: Path) -> dict[str, str]:
    """Create a relative stable alias without writing into a preserved source run."""
    if alias.exists() or alias.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing checkpoint alias: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(os.path.relpath(target, alias.parent))
    if alias.resolve() != target.resolve():
        raise RuntimeError(f"Checkpoint alias did not resolve to its intended target: {alias}")
    return {"alias": str(alias.relative_to(PROJECT)), "target": str(target.relative_to(PROJECT)), "target_sha256": digest(target)}


def main() -> None:
    """Run only completed epochs 6–10, then validate and freeze run artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/")
    parser.add_argument("--source-run-id", default=SOURCE_RUN_ID)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    args = parser.parse_args()
    if args.epochs != TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This continuation requires --epochs 10 --resolution 512 --batch-size 1 --preview-count 9")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")

    device = require_mps()
    sources = source_paths(args.source_run_id)
    source_config = json.loads(sources["config"].read_text(encoding="utf-8"))
    source_result = json.loads(sources["result"].read_text(encoding="utf-8"))
    source_audit = inspect_full_checkpoint(sources["latest_full_state_alias"], expected_epoch=4, expected_step=SOURCE_COMPLETED_EPOCHS * STEPS_PER_EPOCH, require_epoch5_sha=True)
    source_best_sha256 = digest(sources["best_ema"])
    if source_result.get("selected_best_validation_ap_50_95", {}).get("completed_epoch") != SOURCE_COMPLETED_EPOCHS:
        raise RuntimeError("The epoch-5 source result does not identify its preserved best EMA checkpoint as epoch 5")
    source_history = source_ema_history(sources["history"])
    scheduler = scheduler_record(source_config, source_audit)
    entries = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, entries) for path in (TRAIN_COCO, VAL_COCO, CLASS_MAPPING)}
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse existing run directory: {run_dir}")

    run_dir.mkdir(parents=True)
    # This adapter construction checks every frozen raw image and makes only
    # per-run symlinks.  It never moves, rewrites, or augments a source image.
    adapter_info = build_adapter(run_dir, 1)
    preview_plan = fixed_preview_plan()
    adapter_info["fixed_validation_previews"] = preview_plan["examples"]
    adapter_info["description"] = "Read-only symlink adapter for a full-state RF-DETR continuation. Every target is a frozen raw material_version image; no enhanced or test image is present."
    (run_dir / "preview_plan.json").write_text(json.dumps(preview_plan, indent=2) + "\n", encoding="utf-8")
    output_dir = run_dir / "training_output"
    started = time.perf_counter()
    memory_before = mps_memory()

    from rfdetr import RFDETRSmall
    import rfdetr.training as training_stack

    model = RFDETRSmall(resolution=512, device="mps")
    telemetry = EpochTelemetry()
    resume_runtime_audit = ResumeRuntimeAudit(source_audit)
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_telemetry(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.append(telemetry)
        # Appending after RF-DETR's own callback means its on_train_start hook
        # has already rebuilt/restored the averaged EMA model when we inspect it.
        trainer.callbacks.append(resume_runtime_audit)
        return trainer

    launch_config = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Full-state continuation from frozen raw material_version train/validation only; negatives retained; test split absent; enhanced images absent.",
        "device_requirement": "MPS only; PYTORCH_ENABLE_MPS_FALLBACK=0",
        "source_full_checkpoint": source_audit,
        "source_best_ema_checkpoint": {"path": str(sources["best_ema"].relative_to(PROJECT)), "sha256": source_best_sha256},
        "resume_semantics": {
            "full_state_resume": True,
            "restores": ["model", "optimizer", "LR scheduler", "Lightning loop epoch/global step", "RF-DETR EMA callback state"],
            "rng_state_saved": source_audit["rng_state_present"],
            "rng_limitation": "The full checkpoint contains no Python/NumPy/Torch RNG snapshot key. Model, optimizer, scheduler, loop, callback, and EMA state resume; bitwise replay of stochastic augmentation/shuffle state is not claimed.",
        },
        "scheduler": scheduler,
        "frozen_checks": frozen_checks,
        "adapter": adapter_info,
        "fixed_preview_plan": {"path": str((run_dir / "preview_plan.json").relative_to(PROJECT)), **preview_plan["summary"]},
        "training_call": {"epochs": TOTAL_EPOCHS, "batch_size": 1, "resolution": 512, "multi_scale": "off", "expanded_scales": False, "device": "mps", "resume": str(sources["latest_full_state_alias"].relative_to(PROJECT)), "checkpoint_interval": 1, "eval_interval": 1, "use_ema": True, "eval_base_model": False, "eval_ema_only": False},
        "rfdetr": {"source_revision": git_revision(RFDETR_SOURCE)},
    }
    (run_dir / "continuation_config.json").write_text(json.dumps(launch_config, indent=2) + "\n", encoding="utf-8")

    training_stack.build_trainer = build_trainer_with_telemetry
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512, resume=str(sources["latest_full_state_alias"]),
            optimizer="adamw", lr=0.0001, lr_encoder=0.00015, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step",
            lr_scheduler_kwargs={}, warmup_epochs=0.0, grad_accum_steps=1, amp_dtype=None,
            augmentation_backend="cpu", multi_scale="off", expanded_scales=False, scale_jitter=True,
            square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False,
            compute_val_loss="auto", log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False,
            wandb=False, early_stopping=False, run_test=False, seed=args.seed,
            notes={"dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used", "purpose": "AquaVision RF-DETR Small full-state continuation from the preserved epoch-5 fixed-512 raw-image run"},
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    memory_after_training = mps_memory()
    if resume_runtime_audit.record is None:
        raise RuntimeError("The runtime full-state resume audit did not execute")
    if digest(sources["latest_full_state_alias"]) != EXPECTED_EPOCH5_CHECKPOINT_SHA256 or digest(sources["best_ema"]) != source_best_sha256:
        raise RuntimeError("A preserved source checkpoint changed during continuation; refusing to report results")

    resolved = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    train = resolved.get("train_config", {})
    model_config = resolved.get("model_config", {})
    expected_train = {"epochs": TOTAL_EPOCHS, "batch_size": 1, "multi_scale": "off", "expanded_scales": False, "eval_interval": 1, "checkpoint_interval": 1, "run_test": False, "use_ema": True, "eval_base_model": False, "eval_ema_only": False, "lr_scheduler": "step", "lr_scheduler_interval": "step", "warmup_epochs": 0.0, "seed": args.seed}
    errors = {key: {"expected": value, "actual": train.get(key)} for key, value in expected_train.items() if train.get(key) != value}
    if resolved.get("class_names") != [EXPECTED_CATEGORY_NAME] or model_config.get("resolution") != 512 or model_config.get("device") != "mps" or errors:
        raise RuntimeError(f"Resolved configuration differs from the required fixed-512 MPS continuation: {errors}")
    preservation = verify_preserved_configuration(source_config, resolved)

    checkpoints: dict[int, Path] = {SOURCE_COMPLETED_EPOCHS: sources["latest_full_state_alias"]}
    continuation_audits: list[dict[str, Any]] = []
    for completed_epoch in range(SOURCE_COMPLETED_EPOCHS + 1, TOTAL_EPOCHS + 1):
        checkpoint = output_dir / f"checkpoint_{completed_epoch - 1}.ckpt"
        checkpoints[completed_epoch] = checkpoint
        continuation_audits.append(inspect_full_checkpoint(checkpoint, expected_epoch=completed_epoch - 1, expected_step=completed_epoch * STEPS_PER_EPOCH))
    history = full_ema_history(source_history, output_dir / "metrics.csv")
    best_record = max(history, key=lambda record: float(record["EMA_validation"]["AP_50_95"]))
    best_target = sources["best_ema"] if int(best_record["completed_epoch"]) <= SOURCE_COMPLETED_EPOCHS else output_dir / "checkpoint_best_ema.pth"
    if not best_target.is_file():
        raise RuntimeError("RF-DETR did not save the expected best EMA inference checkpoint")

    del model
    gc.collect()
    torch.mps.empty_cache()
    preview_reports = generate_fixed_ema_previews(run_dir, preview_plan, checkpoints, device)
    torch.mps.synchronize()
    memory_after_previews = mps_memory()
    aliases = {
        "latest_full_state": create_alias(run_dir / "checkpoints/latest_full_state.ckpt", checkpoints[TOTAL_EPOCHS]),
        "best_validation_ap_50_95_ema": create_alias(run_dir / "checkpoints/best_validation_ap_50_95_ema.pth", best_target),
    }
    result = {
        "status": "passed",
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_full_checkpoint": source_audit,
        "resume_semantics": launch_config["resume_semantics"],
        "runtime_resume_audit": resume_runtime_audit.record,
        "scheduler": scheduler,
        "resolved_configuration": {"training_config": str((output_dir / "training_config.json").relative_to(PROJECT)), "class_names": resolved["class_names"], "resolution": model_config["resolution"], "device": model_config["device"], "batch_size": train["batch_size"], "multi_scale": train["multi_scale"], "expanded_scales": train["expanded_scales"], "preservation": preservation},
        "frozen_checks": frozen_checks,
        "EMA_metric_history": history,
        "new_epoch_telemetry": telemetry.records,
        "new_full_checkpoints": continuation_audits,
        "selected_best_EMA_validation_AP_50_95": {"completed_epoch": best_record["completed_epoch"], "AP_50_95": best_record["EMA_validation"]["AP_50_95"], "source": best_record["source"], "checkpoint": str(best_target.relative_to(PROJECT)), "checkpoint_sha256": digest(best_target)},
        "checkpoint_aliases": aliases,
        "fixed_EMA_validation_previews": {"selection": preview_plan["summary"], "epochs": preview_reports},
        "timing_seconds": {"continuation_total": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_training": memory_after_training, "after_fixed_previews": memory_after_previews, "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    (run_dir / "continuation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metric_history_ema.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
