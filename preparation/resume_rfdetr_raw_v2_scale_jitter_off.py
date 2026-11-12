#!/usr/bin/env python3
"""Resume the interrupted MPS-only RF-DETR scale_jitter=False ablation.

The source experiment stopped part way through epoch 5.  This runner treats
only its epoch-4 full Lightning checkpoint as authoritative, starts epoch 5
again from its beginning, and stops after ten *total* epochs.  The source run,
its incomplete epoch-5 CSV rows, and its epoch-3 best EMA inference checkpoint
are never modified.  All newly created artifacts live in a new continuation
directory.
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
import random
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only continuation.")
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
    digest,
    frozen_entries,
    git_revision,
    mps_memory,
    require_mps,
    verify_frozen_file,
)
from run_rfdetr_raw_v2_scale_jitter_off import (
    EXPECTED_PRETRAIN_SHA256,
    ORIGINAL_CONFIG,
    ORIGINAL_EPOCH10_FULL,
    ORIGINAL_EPOCH5_BEST_EMA,
    ORIGINAL_HISTORY,
    ORIGINAL_PREVIEW_PLAN,
    PREVIEW_COUNT,
    SEED,
    STEPS_PER_EPOCH,
    TOTAL_EPOCHS,
    audit_full_checkpoint,
    config_differences,
    evaluate_ema_checkpoint,
    generate_previews,
    numeric,
    original_reference,
    validate_preview_plan,
    write_json,
)


SOURCE_RUN_ID = "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z"
SOURCE_DIR = RUNS_ROOT / SOURCE_RUN_ID
SOURCE_OUTPUT = SOURCE_DIR / "training_output"
SOURCE_CHECKPOINT = SOURCE_OUTPUT / "checkpoint_3.ckpt"
SOURCE_BEST_EMA = SOURCE_OUTPUT / "checkpoint_best_ema.pth"
SOURCE_CONFIG = SOURCE_OUTPUT / "training_config.json"
SOURCE_METRICS = SOURCE_OUTPUT / "metrics.csv"
SOURCE_COMPLETED_EPOCHS = 4
CONTINUATION_EPOCHS = tuple(range(SOURCE_COMPLETED_EPOCHS + 1, TOTAL_EPOCHS + 1))


def step_as_int(value: Any) -> int:
    """Convert a serialized AdamW step counter to an ordinary integer."""
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def rng_key_paths(value: Any, prefix: str = "") -> list[str]:
    """Locate explicitly serialized RNG state fields without walking tensors."""
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "rng" in str(key).lower() or "random" in str(key).lower():
                paths.append(path)
            if isinstance(child, (dict, list, tuple)):
                paths.extend(rng_key_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list, tuple)):
                paths.extend(rng_key_paths(child, f"{prefix}[{index}]"))
    return paths


def inspect_source_checkpoint(path: Path) -> dict[str, Any]:
    """Require every serialised state needed for the supported full resume."""
    if path.resolve() != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("The only permitted source is the preserved interrupted experiment's checkpoint_3.ckpt")
    if not path.is_file():
        raise RuntimeError(f"Missing required source checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Source checkpoint is not a Lightning checkpoint dictionary")
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Source checkpoint is not a full training state: missing {missing}")
    if int(payload["epoch"]) != 3 or int(payload["global_step"]) != SOURCE_COMPLETED_EPOCHS * STEPS_PER_EPOCH:
        raise RuntimeError(f"Source checkpoint is not completed epoch 4 at step 20192: epoch={payload['epoch']}, step={payload['global_step']}")
    state_dict = payload["state_dict"]
    optimizer_states = payload["optimizer_states"]
    scheduler_states = payload["lr_schedulers"]
    callbacks = payload["callbacks"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("Source checkpoint has no model state")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1 or not isinstance(optimizer_states[0], dict):
        raise RuntimeError("Source checkpoint does not contain exactly one optimizer state")
    optimizer = optimizer_states[0]
    slots = optimizer.get("state")
    if not isinstance(slots, dict) or not slots:
        raise RuntimeError("Source checkpoint has no populated optimizer slots")
    optimizer_steps = {step_as_int(slot["step"]) for slot in slots.values() if isinstance(slot, dict) and "step" in slot}
    expected_step = SOURCE_COMPLETED_EPOCHS * STEPS_PER_EPOCH
    if optimizer_steps != {expected_step}:
        raise RuntimeError(f"Source optimizer state is at {sorted(optimizer_steps)}, not {expected_step}")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1 or not isinstance(scheduler_states[0], dict):
        raise RuntimeError("Source checkpoint does not contain exactly one scheduler state")
    scheduler = scheduler_states[0]
    if int(scheduler.get("last_epoch", -1)) != expected_step or int(scheduler.get("_step_count", -1)) != expected_step + 1:
        raise RuntimeError("Source scheduler state does not match the completed epoch-4 global step")
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict) or not average or int(ema.get("latest_update_step", -1)) != expected_step:
        raise RuntimeError("Source checkpoint lacks the epoch-4 RF-DETR EMA state")
    loops = payload["loops"]
    fit_loop = loops.get("fit_loop") if isinstance(loops, dict) else None
    batch_state = fit_loop.get("epoch_loop.state_dict") if isinstance(fit_loop, dict) else None
    if not isinstance(batch_state, dict) or int(batch_state.get("_batches_that_stepped", -1)) != expected_step:
        raise RuntimeError("Source Lightning loop does not record the completed epoch-4 batch progress")
    return {
        "path": str(path.relative_to(PROJECT)),
        "sha256": digest(path),
        "epoch_zero_based": 3,
        "completed_epoch": SOURCE_COMPLETED_EPOCHS,
        "global_step": expected_step,
        "model_state_keys": len(state_dict),
        "optimizer": {"state_slots": len(slots), "param_groups": len(optimizer.get("param_groups", [])), "steps": sorted(optimizer_steps)},
        "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler["_step_count"])},
        "EMA": {"latest_update_step": int(ema["latest_update_step"]), "state_keys": len(average)},
        "trainer_loop": {"fit_epoch_progress": fit_loop.get("epoch_progress"), "batches_that_stepped": int(batch_state["_batches_that_stepped"])},
        "rng_state_paths": rng_key_paths(payload),
        "data_order_state": {
            "serialized_sampler_permutation": False,
            "available": "Lightning fit-loop batch progress only",
            "limitation": "No sampler permutation or dataloader generator state is serialized. The continuation deliberately restarts epoch 5, so it never attempts to continue its partial batch sequence.",
        },
    }


def seed_available_generators(seed: int) -> dict[str, Any]:
    """Set every project-visible generator before constructing the resumed model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    return {
        "seed": seed,
        "set_before_model_construction": ["python.random", "numpy.random", "torch", "torch.mps"],
        "limitation": "These are initial seed settings, not a restoration of a missing epoch-4 RNG snapshot. They support a reproducible fresh start of epoch 5 where the pinned implementation permits, but bitwise replay of stochastic augmentation and shuffle order is not claimed.",
    }


def source_metric_segments() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Separate epoch 1-4 results from discarded, incomplete epoch-5 rows."""
    if not SOURCE_METRICS.is_file():
        raise RuntimeError(f"Missing interrupted-run metrics: {SOURCE_METRICS}")
    with SOURCE_METRICS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    losses = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    required_epochs = set(range(SOURCE_COMPLETED_EPOCHS))
    if set(validation) != required_epochs or set(losses) != required_epochs:
        raise RuntimeError(f"Interrupted source must contain complete results only for epochs 0-3; validation={sorted(validation)}, losses={sorted(losses)}")
    history: list[dict[str, Any]] = []
    keys = {
        "AP_50_95": "val/mAP_50_95", "AP50": "val/mAP_50", "AP75": "val/mAP_75", "mAR": "val/mAR",
        "F1_optimized_threshold": "val/F1", "precision_optimized_threshold": "val/precision", "recall_optimized_threshold": "val/recall",
    }
    for epoch in range(SOURCE_COMPLETED_EPOCHS):
        validation_row, loss_row = validation[epoch], losses[epoch]
        lr_rows = [row for row in rows if int(row["epoch"]) == epoch and row.get("train/lr") not in (None, "")]
        if not lr_rows:
            raise RuntimeError(f"Source metrics lacks an LR row for completed epoch {epoch + 1}")
        lr_row = lr_rows[-1]
        history.append({
            "completed_epoch": epoch + 1,
            "segment": "preserved_completed_epochs_1_to_4",
            "global_step": int(validation_row["step"]) + 1,
            "validation_weights": "EMA",
            "EMA_validation": {name: numeric(validation_row, column) for name, column in keys.items()},
            "learning_rates": {name.removeprefix("train/"): numeric(lr_row, name) for name in ("train/lr", "train/lr_min", "train/lr_max")},
            "training_losses": {name.removeprefix("train/"): numeric(loss_row, name) for name in loss_row if name.startswith("train/") and loss_row[name] not in (None, "")},
        })
    partial_rows = [row for row in rows if int(row["epoch"]) == SOURCE_COMPLETED_EPOCHS]
    if not partial_rows or any(row.get("val/mAP_50_95") not in (None, "") or row.get("train/loss") not in (None, "") for row in partial_rows):
        raise RuntimeError("Interrupted epoch-5 rows must be partial training telemetry only, with no completed loss or validation record")
    steps = [int(row["step"]) for row in partial_rows if row.get("step") not in (None, "")]
    return history, {
        "status": "discarded_partial_epoch_5_telemetry",
        "source_metrics_csv": str(SOURCE_METRICS.relative_to(PROJECT)),
        "source_metrics_sha256": digest(SOURCE_METRICS),
        "zero_based_epoch": SOURCE_COMPLETED_EPOCHS,
        "intended_completed_epoch": SOURCE_COMPLETED_EPOCHS + 1,
        "rows": len(partial_rows),
        "first_logged_step": min(steps),
        "last_logged_step": max(steps),
        "completed_epoch_checkpoint_exists": False,
        "validation_or_completed_loss_present": False,
        "handling": "The continuation starts epoch 5 again from checkpoint_3. These rows are retained only as an interruption record and are excluded from all completed-epoch metrics and best-checkpoint selection.",
    }


def continuation_metric_history(metrics_path: Path) -> list[dict[str, Any]]:
    """Read only the six complete resumed epochs from the new output directory."""
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(range(SOURCE_COMPLETED_EPOCHS, TOTAL_EPOCHS))
    validation = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    losses = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    if set(validation) != expected or set(losses) != expected:
        raise RuntimeError(f"Continuation must contain exactly completed epochs 5-10; validation={sorted(validation)}, losses={sorted(losses)}")
    keys = {
        "AP_50_95": "val/mAP_50_95", "AP50": "val/mAP_50", "AP75": "val/mAP_75", "mAR": "val/mAR",
        "F1_optimized_threshold": "val/F1", "precision_optimized_threshold": "val/precision", "recall_optimized_threshold": "val/recall",
    }
    history: list[dict[str, Any]] = []
    for epoch in sorted(expected):
        validation_row, loss_row = validation[epoch], losses[epoch]
        lr_rows = [row for row in rows if int(row["epoch"]) == epoch and row.get("train/lr") not in (None, "")]
        if not lr_rows:
            raise RuntimeError(f"Continuation metrics lacks an LR row for completed epoch {epoch + 1}")
        lr_row = lr_rows[-1]
        history.append({
            "completed_epoch": epoch + 1,
            "segment": "resumed_epochs_5_to_10",
            "global_step": int(validation_row["step"]) + 1,
            "validation_weights": "EMA",
            "EMA_validation": {name: numeric(validation_row, column) for name, column in keys.items()},
            "learning_rates": {name.removeprefix("train/"): numeric(lr_row, name) for name in ("train/lr", "train/lr_min", "train/lr_max")},
            "training_losses": {name.removeprefix("train/"): numeric(loss_row, name) for name in loss_row if name.startswith("train/") and loss_row[name] not in (None, "")},
        })
    return history


def config_audit(source_config: dict[str, Any], original_config: dict[str, Any], continuation_config: dict[str, Any]) -> dict[str, Any]:
    """Ensure only the documented resume paths/notes differ after scale_jitter=False."""
    source_difference = config_differences(original_config, source_config)
    expected_source_difference = {"dataset_dir", "notes", "output_dir", "resume", "scale_jitter"}
    source_unexpected = set(source_difference["train_config"]).difference(expected_source_difference)
    if source_unexpected or source_difference["model_config"] or source_config.get("train_config", {}).get("scale_jitter") is not False:
        raise RuntimeError(f"Interrupted source config differs from its raw baseline unexpectedly: {source_unexpected}")
    continuation_difference = config_differences(source_config, continuation_config)
    allowed_continuation_difference = {"dataset_dir", "notes", "output_dir", "resume"}
    unexpected = set(continuation_difference["train_config"]).difference(allowed_continuation_difference)
    if unexpected or continuation_difference["model_config"] or continuation_config.get("train_config", {}).get("scale_jitter") is not False:
        raise RuntimeError(f"Resume changed settings beyond paths, notes, and resume: {unexpected}")
    return {
        "source_vs_original_raw_recipe": source_difference,
        "source_vs_continuation": continuation_difference,
        "approved_change_remains": "scale_jitter: True -> False relative to the original raw-image recipe. This changes the combined crop/scale augmentation behaviour, not box loss alone.",
    }


class EpochTelemetry(Callback):
    """Record duration and MPS allocator values for each restarted epoch."""

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
        self.records.append({"completed_epoch": epoch + 1, "global_step": int(trainer.global_step), "elapsed_seconds": round(time.perf_counter() - started, 3), "mps_memory_before_epoch": before, "mps_memory_after_epoch": mps_memory()})


class MPSExecutionAudit(Callback):
    """Fail if resume or any continuation epoch places the model off MPS."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs_checked: list[int] = []

    @staticmethod
    def verify(pl_module: Any) -> None:
        devices = {parameter.device.type for parameter in pl_module.parameters()}
        if devices != {"mps"}:
            raise RuntimeError(f"RF-DETR is not wholly MPS-resident: {sorted(devices)}")

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)
        self.epochs_checked.append(int(trainer.current_epoch) + 1)


class ResumeRuntimeAudit(Callback):
    """Confirm Lightning restored the serialized state before epoch 5 begins."""

    def __init__(self, source: dict[str, Any]) -> None:
        super().__init__()
        self.source = source
        self.record: dict[str, Any] | None = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        expected_epoch, expected_step = SOURCE_COMPLETED_EPOCHS, int(self.source["global_step"])
        if int(trainer.current_epoch) != expected_epoch or int(trainer.global_step) != expected_step:
            raise RuntimeError(f"Resume began at epoch/step {trainer.current_epoch}/{trainer.global_step}, not 4/{expected_step}")
        if len(trainer.optimizers) != 1:
            raise RuntimeError("Resume did not restore exactly one optimizer")
        optimizer = trainer.optimizers[0].state_dict()
        slots = optimizer.get("state")
        optimizer_steps = {step_as_int(slot["step"]) for slot in slots.values() if isinstance(slot, dict) and "step" in slot} if isinstance(slots, dict) else set()
        if not isinstance(slots, dict) or len(slots) != self.source["optimizer"]["state_slots"] or optimizer_steps != {expected_step}:
            raise RuntimeError("Resume did not restore the epoch-4 optimizer state")
        schedulers = getattr(trainer, "lr_scheduler_configs", [])
        if len(schedulers) != 1:
            raise RuntimeError("Resume did not restore exactly one scheduler")
        scheduler = schedulers[0].scheduler.state_dict()
        if int(scheduler.get("last_epoch", -1)) != expected_step or int(scheduler.get("_step_count", -1)) != expected_step + 1:
            raise RuntimeError("Resume did not restore the epoch-4 scheduler state")
        ema_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "RFDETREMACallback"]
        if len(ema_callbacks) != 1:
            raise RuntimeError("Resume did not restore exactly one RF-DETR EMA callback")
        ema = ema_callbacks[0]
        averaged_model = getattr(ema, "_average_model", None)
        if averaged_model is None or int(getattr(ema, "_latest_update_step", -1)) != expected_step or len(averaged_model.state_dict()) != self.source["EMA"]["state_keys"]:
            raise RuntimeError("Resume did not restore the epoch-4 EMA state")
        if len(pl_module.state_dict()) != self.source["model_state_keys"]:
            raise RuntimeError("Resume did not restore the epoch-4 model state")
        self.record = {"passed_before_epoch_5": True, "epoch_zero_based": int(trainer.current_epoch), "completed_epoch_before_resume": int(trainer.current_epoch), "global_step": int(trainer.global_step), "model_state_keys": len(pl_module.state_dict()), "optimizer_steps": sorted(optimizer_steps), "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler["_step_count"])}, "EMA_latest_update_step": int(getattr(ema, "_latest_update_step"))}


def create_alias(alias: Path, target: Path) -> dict[str, str]:
    """Create a stable link inside the new continuation directory only."""
    if alias.exists() or alias.is_symlink():
        raise RuntimeError(f"Refusing to overwrite an existing alias: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(os.path.relpath(target, alias.parent))
    if alias.resolve() != target.resolve():
        raise RuntimeError(f"Alias did not resolve to its selected checkpoint: {alias}")
    return {"alias": str(alias.relative_to(PROJECT)), "target": str(target.relative_to(PROJECT)), "target_sha256": digest(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New continuation directory name under training_runs/rfdetr/")
    parser.add_argument("--source-checkpoint", default=str(SOURCE_CHECKPOINT.relative_to(PROJECT)))
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    args = parser.parse_args()
    if args.epochs != TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.seed != SEED or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This resume requires --epochs 10 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple new directory name")
    requested_checkpoint = Path(args.source_checkpoint)
    if not requested_checkpoint.is_absolute():
        requested_checkpoint = PROJECT / requested_checkpoint
    source = inspect_source_checkpoint(requested_checkpoint)
    required = (FREEZE_FILE, TRAIN_COCO, VAL_COCO, CLASS_MAPPING, RFDETR_SOURCE, SOURCE_BEST_EMA, SOURCE_CONFIG, SOURCE_METRICS)
    if not all(path.exists() for path in required):
        missing = [str(path.relative_to(PROJECT)) for path in required if not path.exists()]
        raise SystemExit(f"Required frozen/source artifact is missing: {missing}")
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse an existing continuation directory: {run_dir}")
    expected_cache = (PROJECT / ".cache/rfdetr").resolve()
    if Path(os.environ.get("RF_HOME", "")).resolve() != expected_cache:
        raise SystemExit("RF_HOME must resolve to this project's .cache/rfdetr directory")
    device = require_mps()
    original_config, _, original_preview_plan = original_reference()
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source_history, partial_epoch5 = source_metric_segments()
    source_best_sha256 = digest(SOURCE_BEST_EMA)
    preview_plan = validate_preview_plan(original_preview_plan)
    entries = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, entries) for path in (TRAIN_COCO, VAL_COCO, CLASS_MAPPING)}

    import rfdetr
    import rfdetr.training as training_stack
    from rfdetr import RFDETRSmall
    from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir, validate_pretrain_weights

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The installed RF-DETR package is not the pinned project-local official source")
    if Path(get_model_cache_dir()).resolve() != expected_cache:
        raise RuntimeError("RF-DETR did not resolve the official pretrained asset inside the project cache")
    asset = ModelWeights.from_filename("rf-detr-small.pth")
    pretrained_path = expected_cache / "rf-detr-small.pth"
    if asset is None or asset.md5_hash is None or not pretrained_path.is_file() or digest(pretrained_path) != EXPECTED_PRETRAIN_SHA256 or not validate_pretrain_weights(str(pretrained_path), strict=True):
        raise RuntimeError("The original MD5-verified official RF-DETR Small pretrained checkpoint is unavailable or changed")

    run_dir.mkdir(parents=True)
    adapter = build_adapter(run_dir, PREVIEW_COUNT)
    adapter["fixed_validation_previews"] = preview_plan["examples"]
    adapter["description"] = "Read-only adapter over frozen raw TrashCan material_version train/validation images. Test and enhanced images are absent."
    write_json(run_dir / "preview_plan.json", preview_plan)
    write_json(run_dir / "interrupted_epoch_5_log.json", partial_epoch5)
    launch_seed = seed_available_generators(args.seed)
    intended = copy.deepcopy(source_config)
    intended_train = intended["train_config"]
    intended_train.update({"dataset_dir": str(run_dir / "raw_coco_adapter"), "output_dir": str(run_dir / "training_output"), "resume": str(SOURCE_CHECKPOINT.relative_to(PROJECT)), "notes": {"purpose": "Resume interrupted RF-DETR Small raw-image scale_jitter=False ablation from completed epoch 4", "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used"}})
    before_audit = config_audit(source_config, original_config, intended)
    write_json(run_dir / "continuation_config_before_training.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "source_checkpoint": source, "source_best_ema": {"path": str(SOURCE_BEST_EMA.relative_to(PROJECT)), "sha256": source_best_sha256}, "RNG_and_data_order": {"seed_action": launch_seed, "checkpoint_rng_state_paths": source["rng_state_paths"], "data_order_state": source["data_order_state"]}, "interrupted_epoch_5_log": partial_epoch5, "configuration_audit": before_audit, "frozen_checks": frozen_checks, "adapter": adapter, "continuation_scope": "Epoch 5 is deliberately restarted from its beginning; training stops after completed epoch 10."})

    model = RFDETRSmall(resolution=512, device="mps")
    configured_pretrained = Path(model.model_config.pretrain_weights)
    if configured_pretrained.resolve() != pretrained_path or not validate_pretrain_weights(str(configured_pretrained), strict=True):
        raise RuntimeError("RF-DETR did not initialize from the verified original official pretrained asset before checkpoint restoration")
    telemetry, mps_audit, resume_audit = EpochTelemetry(), MPSExecutionAudit(), ResumeRuntimeAudit(source)
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_audits(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.extend((telemetry, mps_audit, resume_audit))
        return trainer

    started, memory_before = time.perf_counter(), mps_memory()
    output_dir = run_dir / "training_output"
    training_stack.build_trainer = build_trainer_with_audits
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512, resume=str(SOURCE_CHECKPOINT),
            optimizer="adamw", lr=0.0001, lr_encoder=0.00015, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step", lr_scheduler_kwargs={}, warmup_epochs=0.0,
            grad_accum_steps=1, amp_dtype=None, augmentation_backend="cpu", multi_scale="off", expanded_scales=False,
            scale_jitter=False, square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False, compute_val_loss="auto",
            log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False, wandb=False, early_stopping=False, run_test=False,
            seed=args.seed, notes={"purpose": "Resume interrupted RF-DETR Small raw-image scale_jitter=False ablation from completed epoch 4", "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used"},
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    if resume_audit.record is None:
        raise RuntimeError("The runtime full-state resume audit did not run")
    if mps_audit.epochs_checked != list(CONTINUATION_EPOCHS):
        raise RuntimeError(f"MPS was not verified for exactly resumed epochs 5-10: {mps_audit.epochs_checked}")
    if digest(SOURCE_CHECKPOINT) != source["sha256"] or digest(SOURCE_BEST_EMA) != source_best_sha256:
        raise RuntimeError("A preserved source checkpoint changed while the continuation ran")

    resolved = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    after_audit = config_audit(source_config, original_config, resolved)
    if resolved.get("train_config", {}).get("resume") != str(SOURCE_CHECKPOINT):
        raise RuntimeError("Resolved training configuration does not record the required full-state source checkpoint")
    if resolved.get("model_config", {}).get("resolution") != 512 or resolved.get("model_config", {}).get("device") != "mps" or resolved.get("class_names") != ["visible_plastic_debris"]:
        raise RuntimeError("Resolved continuation configuration is not one-class, fixed-512 MPS RF-DETR Small")
    continuation_history = continuation_metric_history(output_dir / "metrics.csv")
    full_history = source_history + continuation_history
    checkpoints: dict[int, Path] = {epoch: SOURCE_OUTPUT / f"checkpoint_{epoch - 1}.ckpt" for epoch in range(1, SOURCE_COMPLETED_EPOCHS + 1)}
    resumed_checkpoint_audits: list[dict[str, Any]] = []
    for epoch in CONTINUATION_EPOCHS:
        checkpoint = output_dir / f"checkpoint_{epoch - 1}.ckpt"
        checkpoints[epoch] = checkpoint
        resumed_checkpoint_audits.append(audit_full_checkpoint(checkpoint, epoch - 1, epoch * STEPS_PER_EPOCH))
    best_record = max(full_history, key=lambda record: float(record["EMA_validation"]["AP_50_95"]))
    continuation_best = output_dir / "checkpoint_best_ema.pth"
    if not continuation_best.is_file():
        raise RuntimeError("RF-DETR did not retain a best EMA inference checkpoint for resumed epochs")
    best_target = SOURCE_BEST_EMA if int(best_record["completed_epoch"]) <= SOURCE_COMPLETED_EPOCHS else continuation_best
    latest_target = checkpoints[TOTAL_EPOCHS]

    del model
    gc.collect()
    torch.mps.empty_cache()
    preview_reports = generate_previews(run_dir, preview_plan, checkpoints, device)
    per_size_dir = run_dir / "per_size_evaluations"
    selected_evaluations = {
        "combined_best": evaluate_ema_checkpoint(best_target, "inference_ema" if best_target.suffix == ".pth" else "full_state_ema", "combined_best", device, per_size_dir),
        "epoch_10": evaluate_ema_checkpoint(latest_target, "full_state_ema", "epoch_10", device, per_size_dir),
    }
    torch.mps.synchronize()
    aliases = {"best_validation_AP_50_95_ema": create_alias(run_dir / "checkpoints/best_validation_ap_50_95_ema.pth", best_target), "latest_full_state": create_alias(run_dir / "checkpoints/latest_full_state.ckpt", latest_target)}
    result = {
        "status": "passed",
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MPS-only full-state resume of RF-DETR Small scale_jitter=False, using frozen raw train/validation data only. Epoch 5 was rerun from its beginning; test, enhanced, source-dataset, and website files were not used or changed.",
        "source_checkpoint": source,
        "runtime_restore_audit": resume_audit.record,
        "RNG_and_data_order_limitations": {"seed_action": launch_seed, "checkpoint_rng_state_paths": source["rng_state_paths"], "data_order_state": source["data_order_state"]},
        "interrupted_epoch_5_log": partial_epoch5,
        "configuration_audits": {"before": str((run_dir / "continuation_config_before_training.json").relative_to(PROJECT)), "after": after_audit},
        "source": {"rfdetr_version": metadata.version("rfdetr"), "source_revision": git_revision(RFDETR_SOURCE), "pretrained_sha256": digest(pretrained_path)},
        "frozen_checks": frozen_checks,
        "EMA_metric_history": full_history,
        "new_epoch_telemetry": telemetry.records,
        "MPS_execution_audit": {"epochs_checked": mps_audit.epochs_checked, "cpu_fallback_disabled": True},
        "new_full_checkpoint_audits": resumed_checkpoint_audits,
        "selected_best": {"completed_epoch": int(best_record["completed_epoch"]), "segment": best_record["segment"], "AP_50_95": float(best_record["EMA_validation"]["AP_50_95"]), "checkpoint": str(best_target.relative_to(PROJECT)), "checkpoint_sha256": digest(best_target)},
        "checkpoint_aliases": aliases,
        "fixed_EMA_validation_previews": preview_reports,
        "selected_per_size_evaluations": selected_evaluations,
        "timing_seconds": {"resume_training_and_postprocessing": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_postprocessing": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(run_dir / "continuation_result.json", result)
    write_json(run_dir / "metric_history_ema.json", full_history)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
