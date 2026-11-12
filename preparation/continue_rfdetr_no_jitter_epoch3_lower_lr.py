#!/usr/bin/env python3
"""Continue the no-jitter epoch-3 RF-DETR state with every LR divided by five.

This is an MPS-only, controlled exploratory continuation. It resumes the
full-state checkpoint that produced the preserved no-jitter epoch-3 EMA
candidate, verifies the restore, then divides every restored optimizer
parameter-group learning rate and every corresponding scheduler base/last
learning rate by five. It runs only completed epochs 4 through 8.

It never uses the inference-only EMA checkpoint for training resume, never
trains on enhanced inputs, never accesses test data, and never writes frozen
data, the website, or an existing experiment directory.
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
from run_rfdetr_raw_v2_scale_jitter_off import (
    EXPECTED_PRETRAIN_SHA256,
    PREVIEW_COUNT,
    SEED,
    STEPS_PER_EPOCH,
    create_alias,
    numeric,
    validate_preview_plan,
    write_json,
)


SOURCE_RUN_ID = "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z"
SOURCE_DIR = RUNS_ROOT / SOURCE_RUN_ID
SOURCE_OUTPUT = SOURCE_DIR / "training_output"
SOURCE_FULL_CHECKPOINT = SOURCE_OUTPUT / "checkpoint_2.ckpt"
SOURCE_EMA_INFERENCE = SOURCE_OUTPUT / "checkpoint_best_ema.pth"
SOURCE_CONFIG = SOURCE_OUTPUT / "training_config.json"
SOURCE_PREVIEW_PLAN = SOURCE_DIR / "preview_plan.json"
SOURCE_METRICS = SOURCE_OUTPUT / "metrics.csv"
ORIGINAL_RESUME_DIR = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e5to10_resume_mps_20261002T121244Z"
ORIGINAL_HISTORY = ORIGINAL_RESUME_DIR / "metric_history_ema.json"
SOURCE_FULL_SHA256 = "6f58bb8b3310b728563e831272aca42f65f45fe671c861820715b065ff3f2a70"
SOURCE_EMA_SHA256 = "6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440"
SOURCE_EPOCH3_AP = 0.17765460908412933
SOURCE_COMPLETED_EPOCH = 3
SOURCE_EPOCH_ZERO_BASED = 2
SOURCE_GLOBAL_STEP = SOURCE_COMPLETED_EPOCH * STEPS_PER_EPOCH
TARGET_TOTAL_EPOCHS = 8
CONTINUATION_EPOCHS = tuple(range(SOURCE_COMPLETED_EPOCH + 1, TARGET_TOTAL_EPOCHS + 1))
LR_DIVISOR = 5.0
MAX_LR_BEFORE = 1e-4
MAX_LR_AFTER = MAX_LR_BEFORE / LR_DIVISOR
ENCODER_LR_BEFORE = 1.5e-4
ENCODER_LR_AFTER = ENCODER_LR_BEFORE / LR_DIVISOR


def step_as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is non-finite")
    return result


def state_fingerprint(state: dict[str, Any], *, strip_ema_prefix: bool = False) -> str:
    """Hash an ordered tensor state so a runtime restore can be verified exactly."""
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if strip_ema_prefix:
            if key == "n_averaged":
                continue
            if not key.startswith("module.model."):
                raise RuntimeError(f"Unexpected EMA state key {key!r}")
            key = key.removeprefix("module.model.")
        normalized[key] = value
    hasher = hashlib.sha256()
    for key in sorted(normalized):
        value = normalized[key]
        if not torch.is_tensor(value):
            raise RuntimeError(f"State value {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def rng_key_paths(value: Any, prefix: str = "") -> list[str]:
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


def learning_rates(groups: list[dict[str, Any]], field: str = "lr") -> list[float]:
    values = [finite(group.get(field), f"optimizer group {index} {field}") for index, group in enumerate(groups)]
    if not values:
        raise RuntimeError("Optimizer has no parameter groups")
    return values


def within(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {left} does not equal {right}")


def inspect_source_checkpoint() -> dict[str, Any]:
    """Verify that checkpoint_2 is the full state behind epoch-3 best EMA."""
    required_paths = (SOURCE_FULL_CHECKPOINT, SOURCE_EMA_INFERENCE, SOURCE_CONFIG, SOURCE_PREVIEW_PLAN, SOURCE_METRICS, ORIGINAL_HISTORY)
    missing = [str(path.relative_to(PROJECT)) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required source artifacts are missing: {missing}")
    if digest(SOURCE_FULL_CHECKPOINT) != SOURCE_FULL_SHA256:
        raise RuntimeError("Epoch-3 full-state checkpoint hash changed")
    if digest(SOURCE_EMA_INFERENCE) != SOURCE_EMA_SHA256:
        raise RuntimeError("Epoch-3 EMA inference checkpoint hash changed")
    payload = torch.load(SOURCE_FULL_CHECKPOINT, map_location="cpu", weights_only=False)
    ema_file = torch.load(SOURCE_EMA_INFERENCE, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    missing_keys = sorted(required.difference(payload)) if isinstance(payload, dict) else sorted(required)
    if missing_keys or not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint_2.ckpt is not a full Lightning training state: missing {missing_keys}")
    if int(payload["epoch"]) != SOURCE_EPOCH_ZERO_BASED or int(payload["global_step"]) != SOURCE_GLOBAL_STEP:
        raise RuntimeError("checkpoint_2.ckpt is not the completed epoch-3 state at step 15,144")
    state = payload["state_dict"]
    optimizers = payload["optimizer_states"]
    schedulers = payload["lr_schedulers"]
    callbacks = payload["callbacks"]
    loops = payload["loops"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Epoch-3 full checkpoint has no model state")
    if not isinstance(optimizers, list) or len(optimizers) != 1 or not isinstance(optimizers[0], dict):
        raise RuntimeError("Epoch-3 full checkpoint lacks its single optimizer state")
    optimizer = optimizers[0]
    slots = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(slots, dict) or not slots or not isinstance(groups, list) or not groups:
        raise RuntimeError("Epoch-3 full checkpoint has incomplete optimizer state")
    optimizer_steps = {step_as_int(slot["step"]) for slot in slots.values() if isinstance(slot, dict) and "step" in slot}
    if optimizer_steps != {SOURCE_GLOBAL_STEP}:
        raise RuntimeError(f"Epoch-3 optimizer slots have unexpected steps: {sorted(optimizer_steps)}")
    if not isinstance(schedulers, list) or len(schedulers) != 1 or not isinstance(schedulers[0], dict):
        raise RuntimeError("Epoch-3 full checkpoint lacks its single scheduler state")
    scheduler = schedulers[0]
    if int(scheduler.get("last_epoch", -1)) != SOURCE_GLOBAL_STEP or int(scheduler.get("_step_count", -1)) != SOURCE_GLOBAL_STEP + 1:
        raise RuntimeError("Epoch-3 scheduler does not match the completed global step")
    base_lrs = [finite(value, f"source scheduler base_lrs[{index}]") for index, value in enumerate(scheduler.get("base_lrs", []))]
    last_lrs = [finite(value, f"source scheduler _last_lr[{index}]") for index, value in enumerate(scheduler.get("_last_lr", []))]
    group_lrs = learning_rates(groups)
    if len(base_lrs) != len(group_lrs) or len(last_lrs) != len(group_lrs):
        raise RuntimeError("Epoch-3 scheduler/optimizer parameter-group lengths differ")
    if max(group_lrs) != MAX_LR_BEFORE or max(base_lrs) != MAX_LR_BEFORE:
        raise RuntimeError("Epoch-3 checkpoint does not have the expected 1e-4 maximum learning rate")
    for index, (group_lr, base_lr, last_lr) in enumerate(zip(group_lrs, base_lrs, last_lrs, strict=True)):
        within(group_lr, base_lr, label=f"source optimizer/scheduler base LR group {index}")
        within(group_lr, last_lr, label=f"source optimizer/scheduler last LR group {index}")
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict) or not average or int(ema.get("latest_update_step", -1)) != SOURCE_GLOBAL_STEP:
        raise RuntimeError("Epoch-3 full checkpoint lacks EMA state at step 15,144")
    if not isinstance(ema_file, dict) or ema_file.get("optimizer_states") or ema_file.get("lr_schedulers"):
        raise RuntimeError("Expected epoch-3 EMA file to be inference-only, but it has training state")
    inference_model = ema_file.get("model")
    if not isinstance(inference_model, dict) or not inference_model:
        raise RuntimeError("Epoch-3 EMA inference file lacks its model state")
    average_hash = state_fingerprint(average, strip_ema_prefix=True)
    inference_hash = state_fingerprint(inference_model)
    if average_hash != inference_hash:
        raise RuntimeError("Epoch-3 full checkpoint EMA state does not match checkpoint_best_ema.pth")
    fit_loop = loops.get("fit_loop") if isinstance(loops, dict) else None
    epoch_loop = fit_loop.get("epoch_loop.state_dict") if isinstance(fit_loop, dict) else None
    if not isinstance(epoch_loop, dict) or int(epoch_loop.get("_batches_that_stepped", -1)) != SOURCE_GLOBAL_STEP:
        raise RuntimeError("Epoch-3 full checkpoint lacks matching Lightning fit-loop progress")
    return {
        "full_state_checkpoint": {
            "path": str(SOURCE_FULL_CHECKPOINT.relative_to(PROJECT)),
            "sha256": SOURCE_FULL_SHA256,
            "epoch_zero_based": SOURCE_EPOCH_ZERO_BASED,
            "completed_epoch": SOURCE_COMPLETED_EPOCH,
            "global_step": SOURCE_GLOBAL_STEP,
            "model_state_keys": len(state),
            "model_state_sha256": state_fingerprint(state),
            "optimizer": {
                "state_slots": len(slots),
                "param_groups": len(groups),
                "parameter_counts": [len(group.get("params", [])) for group in groups],
                "steps": sorted(optimizer_steps),
                "learning_rates": group_lrs,
                "initial_learning_rates": learning_rates(groups, "initial_lr"),
            },
            "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler["_step_count"]), "base_lrs": base_lrs, "last_lrs": last_lrs},
            "EMA": {"latest_update_step": int(ema["latest_update_step"]), "state_keys": len(average), "model_state_sha256": average_hash},
            "trainer_loop": {"batches_that_stepped": int(epoch_loop["_batches_that_stepped"]), "epoch_progress": fit_loop.get("epoch_progress")},
            "rng_state_paths": rng_key_paths(payload),
            "data_order_state": {
                "serialized_sampler_permutation": False,
                "available": "Lightning fit-loop completed-batch progress only",
                "limitation": "No Python, NumPy, Torch, MPS, sampler-permutation, or dataloader-generator state is serialized. Exact stochastic replay is unavailable.",
            },
        },
        "matching_inference_only_ema": {
            "path": str(SOURCE_EMA_INFERENCE.relative_to(PROJECT)),
            "sha256": SOURCE_EMA_SHA256,
            "epoch_zero_based": int(ema_file.get("epoch", -1)),
            "global_step": int(ema_file.get("global_step", -1)),
            "optimizer_state_present": False,
            "scheduler_state_present": False,
            "EMA_model_state_matches_full_checkpoint": True,
            "model_state_sha256": inference_hash,
            "resume_policy": "Preserved for inference and comparison only; never passed to model.train(resume=...).",
        },
    }


def seed_available_generators(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    return {
        "seed": seed,
        "set_before_model_construction": ["python.random", "numpy.random", "torch", "torch.mps"],
        "limitation": "The source has no restorable epoch-3 RNG snapshot. Re-seeding creates a documented exploratory continuation, not a bitwise stochastic replay.",
    }


def source_config_and_reference_history() -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    train = source_config.get("train_config")
    model = source_config.get("model_config")
    expected = {
        "epochs": 10, "batch_size": 1, "seed": SEED, "multi_scale": "off", "expanded_scales": False,
        "scale_jitter": False, "optimizer": "adamw", "lr": MAX_LR_BEFORE, "lr_encoder": ENCODER_LR_BEFORE,
        "lr_scheduler": "step", "lr_scheduler_interval": "step", "lr_scheduler_kwargs": {}, "warmup_epochs": 0.0,
        "use_ema": True, "ema_decay": 0.993, "ema_tau": 100, "ema_update_interval": 1,
        "eval_interval": 1, "checkpoint_interval": 1, "run_test": False, "num_workers": 0, "amp_dtype": None,
    }
    mismatches = {key: {"expected": value, "actual": train.get(key) if isinstance(train, dict) else None} for key, value in expected.items() if not isinstance(train, dict) or train.get(key) != value}
    if mismatches or not isinstance(model, dict) or model.get("model_name") != "RFDETRSmall" or model.get("resolution") != 512 or model.get("device") != "mps" or model.get("num_classes") != 1:
        raise RuntimeError(f"Source no-jitter configuration is not the expected frozen recipe: {mismatches}")
    history = json.loads(ORIGINAL_HISTORY.read_text(encoding="utf-8"))
    if not isinstance(history, list) or [int(entry.get("completed_epoch", -1)) for entry in history] != list(range(1, 11)):
        raise RuntimeError("Original no-jitter history does not contain complete epochs 1 through 10")
    by_epoch = {int(entry["completed_epoch"]): entry for entry in history}
    if float(by_epoch[3]["EMA_validation"]["AP_50_95"]) != SOURCE_EPOCH3_AP:
        raise RuntimeError("Original no-jitter epoch-3 AP does not match the preserved best EMA candidate")
    return source_config, by_epoch, json.loads(SOURCE_PREVIEW_PLAN.read_text(encoding="utf-8"))


def intended_config(source_config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    candidate = copy.deepcopy(source_config)
    train = candidate["train_config"]
    train.update({
        "epochs": TARGET_TOTAL_EPOCHS,
        "dataset_dir": str(run_dir / "raw_coco_adapter"),
        "output_dir": str(run_dir / "training_output"),
        "resume": str(SOURCE_FULL_CHECKPOINT.relative_to(PROJECT)),
        "lr": MAX_LR_AFTER,
        "lr_encoder": ENCODER_LR_AFTER,
        "notes": {
            "purpose": "Controlled RF-DETR Small no-jitter continuation from epoch-3 full state with every optimizer learning rate divided by five",
            "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent",
            "enhancement_input": "not used",
            "comparison": "Original no-jitter epochs 4-8 and epoch-3 EMA AP 0.1776546091",
        },
    })
    return candidate


def configuration_audit(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    source_train, candidate_train = source["train_config"], candidate["train_config"]
    source_model, candidate_model = source["model_config"], candidate["model_config"]
    train_diffs = {
        key: {"source": source_train.get(key), "candidate": candidate_train.get(key)}
        for key in sorted(set(source_train).union(candidate_train)) if source_train.get(key) != candidate_train.get(key)
    }
    model_diffs = {
        key: {"source": source_model.get(key), "candidate": candidate_model.get(key)}
        for key in sorted(set(source_model).union(candidate_model)) if source_model.get(key) != candidate_model.get(key)
    }
    allowed = {"epochs", "dataset_dir", "output_dir", "resume", "lr", "lr_encoder", "notes"}
    if set(train_diffs).difference(allowed) or model_diffs:
        raise RuntimeError(f"Continuation would change settings beyond the approved set: train={sorted(set(train_diffs).difference(allowed))}, model={model_diffs}")
    expected = {
        "epochs": {"source": 10, "candidate": 8},
        "lr": {"source": MAX_LR_BEFORE, "candidate": MAX_LR_AFTER},
        "lr_encoder": {"source": ENCODER_LR_BEFORE, "candidate": ENCODER_LR_AFTER},
    }
    if {key: train_diffs.get(key) for key in expected} != expected:
        raise RuntimeError("Continuation does not set the intended epoch budget and top-level learning rates")
    return {
        "approved_train_config_differences": train_diffs,
        "model_config_differences": model_diffs,
        "controlled_change": {
            "learning_rate_divisor": LR_DIVISOR,
            "maximum_group_learning_rate": {"before": MAX_LR_BEFORE, "after": MAX_LR_AFTER},
            "encoder_base_learning_rate": {"before": ENCODER_LR_BEFORE, "after": ENCODER_LR_AFTER},
            "method": "After Lightning restores the saved optimizer and scheduler state, every restored optimizer group LR, initial_lr, scheduler base_lr, and scheduler last_lr is divided by five. Relative group rates, optimizer slots, scheduler counters, EMA state, losses, and all other training settings are retained.",
        },
        "epoch_budget_note": "The max epoch value changes from 10 to 8 solely to stop after completed epoch 8. The pinned step scheduler's first drop is after epoch 100, so its schedule factor stays 1.0 for epochs 4-8; scheduler counters are resumed from step 15,144.",
        "reproducibility_note": "Because the source checkpoint has no RNG or sampler-permutation snapshot, this is a controlled exploratory comparison, not an exact stochastic replay.",
    }


class EpochTelemetry(Callback):
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
    def __init__(self) -> None:
        super().__init__()
        self.epochs_checked: list[int] = []

    @staticmethod
    def verify(pl_module: Any) -> None:
        devices = {parameter.device.type for parameter in pl_module.parameters()}
        if devices != {"mps"}:
            raise RuntimeError(f"RF-DETR training module is not entirely MPS-resident: {sorted(devices)}")

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.verify(pl_module)
        self.epochs_checked.append(int(trainer.current_epoch) + 1)


class RestoreAndLowerLRAudit(Callback):
    """Verify the restored full state, then apply and audit the fivefold LR change."""

    def __init__(self, source: dict[str, Any]) -> None:
        super().__init__()
        self.source = source["full_state_checkpoint"]
        self.restore_before_change: dict[str, Any] | None = None
        self.after_change: dict[str, Any] | None = None
        self.epoch_learning_rates: list[dict[str, Any]] = []

    @staticmethod
    def scheduler(trainer: Any) -> Any:
        configs = getattr(trainer, "lr_scheduler_configs", [])
        if len(configs) != 1:
            raise RuntimeError("Resume did not restore exactly one scheduler")
        return configs[0].scheduler

    def verify_restore(self, trainer: Any, pl_module: Any) -> tuple[Any, Any]:
        if int(trainer.current_epoch) != SOURCE_COMPLETED_EPOCH or int(trainer.global_step) != SOURCE_GLOBAL_STEP:
            raise RuntimeError(f"Full-state resume began at epoch/step {trainer.current_epoch}/{trainer.global_step}, not 3/{SOURCE_GLOBAL_STEP}")
        if len(trainer.optimizers) != 1:
            raise RuntimeError("Resume did not restore exactly one optimizer")
        optimizer = trainer.optimizers[0]
        state = optimizer.state_dict()
        slots = state.get("state")
        groups = state.get("param_groups")
        if not isinstance(slots, dict) or len(slots) != self.source["optimizer"]["state_slots"] or not isinstance(groups, list):
            raise RuntimeError("Resume did not restore the expected optimizer slots/groups")
        steps = {step_as_int(slot["step"]) for slot in slots.values() if isinstance(slot, dict) and "step" in slot}
        if steps != {SOURCE_GLOBAL_STEP} or [len(group.get("params", [])) for group in groups] != self.source["optimizer"]["parameter_counts"]:
            raise RuntimeError("Resume did not restore the epoch-3 optimizer step/parameter groups")
        current_lrs = learning_rates(optimizer.param_groups)
        current_initial = learning_rates(optimizer.param_groups, "initial_lr")
        for index, (actual, expected) in enumerate(zip(current_lrs, self.source["optimizer"]["learning_rates"], strict=True)):
            within(actual, expected, label=f"restored optimizer LR group {index}")
        for index, (actual, expected) in enumerate(zip(current_initial, self.source["optimizer"]["initial_learning_rates"], strict=True)):
            within(actual, expected, label=f"restored optimizer initial LR group {index}")
        scheduler = self.scheduler(trainer)
        state_scheduler = scheduler.state_dict()
        if int(state_scheduler.get("last_epoch", -1)) != SOURCE_GLOBAL_STEP or int(state_scheduler.get("_step_count", -1)) != SOURCE_GLOBAL_STEP + 1:
            raise RuntimeError("Resume did not restore the epoch-3 scheduler counters")
        for field, expected_values in (("base_lrs", self.source["scheduler"]["base_lrs"]), ("_last_lr", self.source["scheduler"]["last_lrs"])):
            actual_values = [finite(value, f"restored scheduler {field}") for value in state_scheduler.get(field, [])]
            if len(actual_values) != len(expected_values):
                raise RuntimeError(f"Resume scheduler {field} has wrong number of parameter groups")
            for index, (actual, expected) in enumerate(zip(actual_values, expected_values, strict=True)):
                within(actual, expected, label=f"restored scheduler {field} group {index}")
        ema_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "RFDETREMACallback"]
        if len(ema_callbacks) != 1:
            raise RuntimeError("Resume did not restore exactly one RF-DETR EMA callback")
        ema = ema_callbacks[0]
        averaged = getattr(ema, "_average_model", None)
        if averaged is None or int(getattr(ema, "_latest_update_step", -1)) != SOURCE_GLOBAL_STEP:
            raise RuntimeError("Resume did not restore the epoch-3 EMA callback state")
        runtime_model_hash = state_fingerprint(pl_module.state_dict())
        runtime_ema_hash = state_fingerprint(averaged.state_dict(), strip_ema_prefix=True)
        if runtime_model_hash != self.source["model_state_sha256"] or runtime_ema_hash != self.source["EMA"]["model_state_sha256"]:
            raise RuntimeError("Runtime model or EMA tensors do not match the restored epoch-3 source state")
        return optimizer, scheduler

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        optimizer, scheduler = self.verify_restore(trainer, pl_module)
        before_groups = learning_rates(optimizer.param_groups)
        before_initial = learning_rates(optimizer.param_groups, "initial_lr")
        before_scheduler = scheduler.state_dict()
        self.restore_before_change = {
            "passed": True,
            "epoch_zero_based": int(trainer.current_epoch),
            "completed_epoch": SOURCE_COMPLETED_EPOCH,
            "global_step": int(trainer.global_step),
            "model_state_sha256": self.source["model_state_sha256"],
            "EMA_model_state_sha256": self.source["EMA"]["model_state_sha256"],
            "optimizer_learning_rates": before_groups,
            "scheduler": {"last_epoch": int(before_scheduler["last_epoch"]), "step_count": int(before_scheduler["_step_count"]), "base_lrs": [float(value) for value in before_scheduler["base_lrs"]], "last_lrs": [float(value) for value in before_scheduler["_last_lr"]]},
        }
        for group in optimizer.param_groups:
            group["lr"] = finite(group["lr"], "group lr") / LR_DIVISOR
            group["initial_lr"] = finite(group["initial_lr"], "group initial_lr") / LR_DIVISOR
        scheduler.base_lrs = [finite(value, "scheduler base_lr") / LR_DIVISOR for value in scheduler.base_lrs]
        scheduler._last_lr = [finite(value, "scheduler last_lr") / LR_DIVISOR for value in scheduler._last_lr]
        after_groups = learning_rates(optimizer.param_groups)
        after_initial = learning_rates(optimizer.param_groups, "initial_lr")
        after_scheduler = scheduler.state_dict()
        for index, (before, after, before_initial_lr, after_initial_lr, before_base, after_base, before_last, after_last) in enumerate(zip(before_groups, after_groups, before_initial, after_initial, before_scheduler["base_lrs"], after_scheduler["base_lrs"], before_scheduler["_last_lr"], after_scheduler["_last_lr"], strict=True)):
            within(after, before / LR_DIVISOR, label=f"scaled optimizer LR group {index}")
            within(after_initial_lr, before_initial_lr / LR_DIVISOR, label=f"scaled optimizer initial LR group {index}")
            within(float(after_base), float(before_base) / LR_DIVISOR, label=f"scaled scheduler base LR group {index}")
            within(float(after_last), float(before_last) / LR_DIVISOR, label=f"scaled scheduler last LR group {index}")
        if max(after_groups) != MAX_LR_AFTER:
            raise RuntimeError("Fivefold learning-rate reduction did not yield maximum LR 2e-5")
        self.after_change = {
            "passed": True,
            "divisor": LR_DIVISOR,
            "maximum_learning_rate": max(after_groups),
            "optimizer_learning_rates": after_groups,
            "optimizer_initial_learning_rates": after_initial,
            "scheduler": {"last_epoch": int(after_scheduler["last_epoch"]), "step_count": int(after_scheduler["_step_count"]), "base_lrs": [float(value) for value in after_scheduler["base_lrs"]], "last_lrs": [float(value) for value in after_scheduler["_last_lr"]]},
        }
        self.capture_epoch(trainer, "before_epoch")

    def capture_epoch(self, trainer: Any, phase: str) -> None:
        if self.after_change is None:
            raise RuntimeError("LR scaling did not run before the first epoch")
        optimizer = trainer.optimizers[0]
        scheduler = self.scheduler(trainer)
        scheduler_state = scheduler.state_dict()
        base_lrs = [finite(value, "scheduler base_lr") for value in scheduler_state.get("base_lrs", [])]
        expected_bases = self.after_change["scheduler"]["base_lrs"]
        if len(base_lrs) != len(expected_bases):
            raise RuntimeError("Scheduler lost a parameter group after LR scaling")
        for index, (actual, expected) in enumerate(zip(base_lrs, expected_bases, strict=True)):
            within(actual, expected, label=f"scheduler retained scaled base LR group {index}")
        actual_lrs = learning_rates(optimizer.param_groups)
        reported_lrs = [finite(value, "scheduler last_lr") for value in scheduler.get_last_lr()]
        for index, (actual, reported) in enumerate(zip(actual_lrs, reported_lrs, strict=True)):
            within(actual, reported, label=f"scheduler/optimizer current LR group {index}")
        self.epoch_learning_rates.append({
            "completed_epoch": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
            "phase": phase,
            "optimizer_learning_rates": actual_lrs,
            "optimizer_initial_learning_rates": learning_rates(optimizer.param_groups, "initial_lr"),
            "scheduler": {"last_epoch": int(scheduler_state["last_epoch"]), "step_count": int(scheduler_state["_step_count"]), "base_lrs": base_lrs, "last_lrs": reported_lrs},
            "scaled_scheduler_base_lrs_retained": True,
        })

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.capture_epoch(trainer, "epoch_start")

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self.capture_epoch(trainer, "epoch_end")


def parse_new_history(metrics_path: Path, lr_audit: RestoreAndLowerLRAudit) -> list[dict[str, Any]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_zero_epochs = set(range(SOURCE_COMPLETED_EPOCH, TARGET_TOTAL_EPOCHS))
    validation = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    losses = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    if set(validation) != expected_zero_epochs or set(losses) != expected_zero_epochs:
        raise RuntimeError(f"Continuation must have complete epoch 4-8 metrics; validation={sorted(validation)}, loss={sorted(losses)}")
    end_lrs = {int(item["completed_epoch"]): item for item in lr_audit.epoch_learning_rates if item["phase"] == "epoch_end"}
    if set(end_lrs) != set(CONTINUATION_EPOCHS):
        raise RuntimeError(f"LR audit lacks a completed epoch record: {sorted(end_lrs)}")
    keys = {"AP_50_95": "val/mAP_50_95", "AP50": "val/ema_mAP_50", "AP75": "val/ema_mAP_75", "mAR": "val/ema_mAR", "F1_optimized_threshold": "val/F1", "precision_optimized_threshold": "val/precision", "recall_optimized_threshold": "val/recall"}
    history: list[dict[str, Any]] = []
    for zero_epoch in sorted(expected_zero_epochs):
        epoch = zero_epoch + 1
        validation_row, loss_row = validation[zero_epoch], losses[zero_epoch]
        lr_rows = [row for row in rows if int(row["epoch"]) == zero_epoch and row.get("train/lr") not in (None, "")]
        if not lr_rows:
            raise RuntimeError(f"Continuation metrics lacks an LR row for epoch {epoch}")
        lr_row = lr_rows[-1]
        history.append({
            "completed_epoch": epoch,
            "global_step": int(validation_row["step"]) + 1,
            "validation_weights": "EMA",
            "EMA_validation": {name: numeric(validation_row, column) for name, column in keys.items()},
            "logged_learning_rate_summary": {name.removeprefix("train/"): numeric(lr_row, name) for name in ("train/lr", "train/lr_min", "train/lr_max")},
            "actual_optimizer_learning_rates": end_lrs[epoch],
            "training_losses": {name.removeprefix("train/"): numeric(loss_row, name) for name in loss_row if name.startswith("train/") and loss_row[name] not in (None, "")},
        })
    return history


def audit_scaled_checkpoint(path: Path, epoch: int, source: dict[str, Any]) -> dict[str, Any]:
    """Verify saved full state remains resume-capable and retains scaled scheduler bases."""
    if not path.is_file():
        raise RuntimeError(f"Missing full-state checkpoint for completed epoch {epoch}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    if not isinstance(payload, dict) or required.difference(payload) or int(payload["epoch"]) != epoch - 1 or int(payload["global_step"]) != epoch * STEPS_PER_EPOCH:
        raise RuntimeError(f"{path.name} is not the expected full-state checkpoint for completed epoch {epoch}")
    optimizer = payload["optimizer_states"][0]
    scheduler = payload["lr_schedulers"][0]
    callbacks = payload["callbacks"]
    groups = optimizer.get("param_groups")
    slots = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(slots, dict) or not slots:
        raise RuntimeError(f"{path.name} lacks optimizer state")
    steps = {step_as_int(slot["step"]) for slot in slots.values() if isinstance(slot, dict) and "step" in slot}
    if steps != {epoch * STEPS_PER_EPOCH} or int(scheduler.get("last_epoch", -1)) != epoch * STEPS_PER_EPOCH:
        raise RuntimeError(f"{path.name} optimizer/scheduler counters do not match epoch {epoch}")
    expected_bases = [value / LR_DIVISOR for value in source["optimizer"]["learning_rates"]]
    actual_bases = [finite(value, "saved scheduler base_lr") for value in scheduler.get("base_lrs", [])]
    actual_lrs = learning_rates(groups)
    if len(actual_bases) != len(expected_bases) or len(actual_lrs) != len(expected_bases):
        raise RuntimeError(f"{path.name} changed the optimizer parameter-group count")
    for index, (base, expected) in enumerate(zip(actual_bases, expected_bases, strict=True)):
        within(base, expected, label=f"saved scaled scheduler base LR group {index}")
    if max(actual_lrs) > MAX_LR_AFTER + 1e-12:
        raise RuntimeError(f"{path.name} has an LR above the 2e-5 maximum")
    ema = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema.get("average_model_state_dict") if isinstance(ema, dict) else None
    if not isinstance(average, dict) or int(ema.get("latest_update_step", -1)) != epoch * STEPS_PER_EPOCH:
        raise RuntimeError(f"{path.name} lacks EMA state for completed epoch {epoch}")
    return {"path": str(path.relative_to(PROJECT)), "sha256": digest(path), "completed_epoch": epoch, "global_step": epoch * STEPS_PER_EPOCH, "optimizer_state_slots": len(slots), "scheduler": {"last_epoch": int(scheduler["last_epoch"]), "step_count": int(scheduler.get("_step_count", -1)), "base_lrs": actual_bases, "current_lrs": actual_lrs, "scaled_bases_retained": True}, "EMA_state_keys": len(average)}


def generate_previews_for_epochs(run_dir: Path, plan: dict[str, Any], checkpoints: dict[int, Path], device: torch.device) -> list[dict[str, Any]]:
    from run_rfdetr_raw_v2_scale_jitter_off import load_ema_model

    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, rows, "valid")
    images_by_path = {str(image["file_name"]): image for image in images.values()}
    adapter_info = {"fixed_validation_previews": plan["examples"]}
    reports: list[dict[str, Any]] = []
    for epoch, checkpoint in sorted(checkpoints.items()):
        model = load_ema_model(checkpoint, device)
        preview_root = run_dir / "fixed_validation_previews" / f"epoch_{epoch:03d}_ema"
        previews = create_labeled_previews(model, adapter_info, annotations, images_by_path, preview_root, device)
        reports.append({"completed_epoch": epoch, "validation_weights": "EMA", "checkpoint": str(checkpoint.relative_to(PROJECT)), "checkpoint_sha256": digest(checkpoint), "previews": previews})
        del model
        gc.collect()
        torch.mps.empty_cache()
    return reports


def comparison_against_original(new_history: list[dict[str, Any]], original: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for entry in new_history:
        epoch = int(entry["completed_epoch"])
        source = original[epoch]
        keys = ("AP_50_95", "AP50", "AP75", "mAR", "F1_optimized_threshold", "precision_optimized_threshold", "recall_optimized_threshold")
        comparisons.append({
            "completed_epoch": epoch,
            "original_no_jitter": source,
            "lower_lr": entry,
            "EMA_metric_deltas_lower_lr_minus_original": {key: float(entry["EMA_validation"][key]) - float(source["EMA_validation"][key]) for key in keys},
            "AP_50_95_delta_from_epoch3_start": float(entry["EMA_validation"]["AP_50_95"]) - SOURCE_EPOCH3_AP,
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New lower-LR continuation directory under training_runs/rfdetr/")
    parser.add_argument("--epochs", type=int, default=TARGET_TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    args = parser.parse_args()
    if args.epochs != TARGET_TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.seed != SEED or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This continuation requires --epochs 8 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple new directory name")
    source = inspect_source_checkpoint()
    source_config, original_history, source_preview = source_config_and_reference_history()
    preview_plan = validate_preview_plan(source_preview)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse an existing continuation directory: {run_dir}")
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
        raise RuntimeError("The installed RF-DETR package is not the pinned project-local source")
    if Path(get_model_cache_dir()).resolve() != expected_cache:
        raise RuntimeError("RF-DETR pretrained-weight cache is not project-local")
    asset = ModelWeights.from_filename("rf-detr-small.pth")
    pretrained_path = expected_cache / "rf-detr-small.pth"
    if asset is None or asset.md5_hash is None or not pretrained_path.is_file() or digest(pretrained_path) != EXPECTED_PRETRAIN_SHA256 or not validate_pretrain_weights(str(pretrained_path), strict=True):
        raise RuntimeError("The original verified official RF-DETR Small pretrained checkpoint is unavailable or changed")

    run_dir.mkdir(parents=True)
    adapter = build_adapter(run_dir, PREVIEW_COUNT)
    adapter["fixed_validation_previews"] = preview_plan["examples"]
    adapter["description"] = "Read-only adapter over frozen raw TrashCan material_version train/validation images. Test and enhanced images are absent."
    write_json(run_dir / "preview_plan.json", preview_plan)
    seed_record = seed_available_generators(args.seed)
    write_json(run_dir / "lower_lr_config_before_training.json", {
        "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_audit": source,
        "configuration_difference_audit": config_audit,
        "intended_resolved_config": intended,
        "frozen_checks": frozen_checks,
        "adapter": adapter,
        "RNG_and_data_order": {"seed_action": seed_record, "checkpoint_rng_state_paths": source["full_state_checkpoint"]["rng_state_paths"], "data_order_state": source["full_state_checkpoint"]["data_order_state"]},
        "original_no_jitter_reference": {"history_path": str(ORIGINAL_HISTORY.relative_to(PROJECT)), "history_sha256": digest(ORIGINAL_HISTORY), "epoch3_start_AP_50_95": SOURCE_EPOCH3_AP, "comparison_epochs": list(CONTINUATION_EPOCHS)},
        "scope": "Controlled exploratory continuation from no-jitter epoch 3 through epoch 8 only. No test, enhanced, source-dataset, or website files are used or changed.",
    })

    model = RFDETRSmall(resolution=512, device="mps")
    configured_pretrained = Path(model.model_config.pretrain_weights)
    if configured_pretrained.resolve() != pretrained_path or not validate_pretrain_weights(str(configured_pretrained), strict=True):
        raise RuntimeError("RF-DETR did not initialize from the project-local verified pretrained asset before full-state restore")
    telemetry, mps_audit, lr_audit = EpochTelemetry(), MPSExecutionAudit(), RestoreAndLowerLRAudit(source)
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_audits(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.extend((telemetry, mps_audit, lr_audit))
        return trainer

    output_dir = run_dir / "training_output"
    started, memory_before = time.perf_counter(), mps_memory()
    training_stack.build_trainer = build_trainer_with_audits
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=TARGET_TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512, resume=str(SOURCE_FULL_CHECKPOINT),
            optimizer="adamw", lr=MAX_LR_AFTER, lr_encoder=ENCODER_LR_AFTER, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step", lr_scheduler_kwargs={}, warmup_epochs=0.0,
            grad_accum_steps=1, amp_dtype=None, augmentation_backend="cpu", multi_scale="off", expanded_scales=False,
            scale_jitter=False, square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False, compute_val_loss="auto",
            log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False, wandb=False, early_stopping=False, run_test=False,
            seed=args.seed, notes=intended["train_config"]["notes"],
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    if lr_audit.restore_before_change is None or lr_audit.after_change is None:
        raise RuntimeError("Full-state restore/LR-scaling audit never ran")
    if mps_audit.epochs_checked != list(CONTINUATION_EPOCHS):
        raise RuntimeError(f"MPS was not verified for exactly epochs 4-8: {mps_audit.epochs_checked}")
    if digest(SOURCE_FULL_CHECKPOINT) != SOURCE_FULL_SHA256 or digest(SOURCE_EMA_INFERENCE) != SOURCE_EMA_SHA256:
        raise RuntimeError("A preserved epoch-3 source checkpoint changed while the continuation ran")

    resolved_path = output_dir / "training_config.json"
    if not resolved_path.is_file():
        raise RuntimeError("RF-DETR did not write its resolved training configuration")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    after_config_audit = configuration_audit(source_config, resolved)
    if resolved.get("train_config", {}).get("resume") != str(SOURCE_FULL_CHECKPOINT) or resolved.get("train_config", {}).get("scale_jitter") is not False:
        raise RuntimeError("Resolved configuration did not retain the required full-state resume or no-jitter setting")
    write_json(run_dir / "lower_lr_config_after_training.json", {"status": "passed", "resolved_config": str(resolved_path.relative_to(PROJECT)), "configuration_difference_audit": after_config_audit, "restore_before_lr_change": lr_audit.restore_before_change, "lr_change": lr_audit.after_change, "actual_learning_rates": lr_audit.epoch_learning_rates})

    history = parse_new_history(output_dir / "metrics.csv", lr_audit)
    checkpoints = {epoch: output_dir / f"checkpoint_{epoch - 1}.ckpt" for epoch in CONTINUATION_EPOCHS}
    checkpoint_audits = [audit_scaled_checkpoint(checkpoint, epoch, source["full_state_checkpoint"]) for epoch, checkpoint in checkpoints.items()]
    output_best = output_dir / "checkpoint_best_ema.pth"
    starting_record = original_history[SOURCE_COMPLETED_EPOCH]
    best_record = max([starting_record, *history], key=lambda record: float(record["EMA_validation"]["AP_50_95"]))
    if int(best_record["completed_epoch"]) == SOURCE_COMPLETED_EPOCH:
        best_target = SOURCE_EMA_INFERENCE
    else:
        if not output_best.is_file():
            raise RuntimeError("A newly selected best epoch has no RF-DETR EMA inference checkpoint")
        best_target = output_best
    latest_full = checkpoints[TARGET_TOTAL_EPOCHS]
    aliases = {
        "best_validation_AP_50_95_ema": create_alias(run_dir / "checkpoints/best_validation_ap_50_95_ema.pth", best_target),
        "latest_full_state": create_alias(run_dir / "checkpoints/latest_full_state.ckpt", latest_full),
    }

    del model
    gc.collect()
    torch.mps.empty_cache()
    previews = generate_previews_for_epochs(run_dir, preview_plan, checkpoints, device)
    comparisons = comparison_against_original(history, original_history)
    result = {
        "status": "passed",
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MPS-only controlled exploratory RF-DETR Small no-jitter continuation from full epoch-3 state through epoch 8. Every restored optimizer/scheduler learning rate was divided by five after restore. Frozen raw train/validation inputs include negatives; test, enhanced, source-dataset, and website files were not used or changed.",
        "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "cpu_fallback_disabled": True,
        "source_checkpoint": source,
        "restore_before_learning_rate_change": lr_audit.restore_before_change,
        "learning_rate_change": lr_audit.after_change,
        "actual_learning_rate_records": lr_audit.epoch_learning_rates,
        "RNG_and_data_order_limitations": {"seed_action": seed_record, "checkpoint_rng_state_paths": source["full_state_checkpoint"]["rng_state_paths"], "data_order_state": source["full_state_checkpoint"]["data_order_state"]},
        "configuration_audits": {"before": str((run_dir / "lower_lr_config_before_training.json").relative_to(PROJECT)), "after": str((run_dir / "lower_lr_config_after_training.json").relative_to(PROJECT))},
        "source": {"rfdetr_version": metadata.version("rfdetr"), "source_revision": git_revision(RFDETR_SOURCE), "pretrained_sha256": digest(pretrained_path)},
        "frozen_checks": frozen_checks,
        "EMA_metric_history": history,
        "comparison_against_original_no_jitter_epochs_4_to_8": comparisons,
        "starting_epoch3_AP_50_95": SOURCE_EPOCH3_AP,
        "MPS_execution_audit": {"epochs_checked": mps_audit.epochs_checked, "cpu_fallback_disabled": True},
        "epoch_telemetry": telemetry.records,
        "full_checkpoint_audits": checkpoint_audits,
        "checkpoint_aliases": aliases,
        "selected_best": {"completed_epoch": int(best_record["completed_epoch"]), "AP_50_95": float(best_record["EMA_validation"]["AP_50_95"]), "checkpoint": str(best_target.relative_to(PROJECT)), "checkpoint_sha256": digest(best_target), "selection_scope": "maximum of preserved epoch-3 candidate and newly completed epochs 4-8"},
        "fixed_EMA_validation_previews": previews,
        "timing_seconds": {"training_and_postprocessing": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_postprocessing": mps_memory(), "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    write_json(run_dir / "lower_lr_result.json", result)
    write_json(run_dir / "metric_history_ema.json", history)
    report = [
        "# RF-DETR no-jitter lower-learning-rate continuation: epochs 4-8",
        "",
        "This controlled exploratory continuation resumed the preserved epoch-3 full state, then divided every optimizer parameter-group and scheduler base learning rate by five. It uses raw frozen train/validation images only; test and enhanced images are excluded.",
        "",
        f"- Starting epoch-3 EMA AP@[.50:.95]: {SOURCE_EPOCH3_AP:.6f}",
        f"- Resume full-state checkpoint: `{SOURCE_FULL_CHECKPOINT.relative_to(PROJECT)}`",
        f"- Maximum LR: {MAX_LR_BEFORE:g} -> {MAX_LR_AFTER:g}",
        "- Exact stochastic replay is not claimed because the source does not serialize RNG or sampler-order state.",
        "",
        "| Epoch | Lower-LR EMA AP | Original no-jitter EMA AP | Lower-LR − original | Lower-LR − epoch-3 start |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in comparisons:
        epoch = entry["completed_epoch"]
        report.append(f"| {epoch} | {entry['lower_lr']['EMA_validation']['AP_50_95']:.6f} | {entry['original_no_jitter']['EMA_validation']['AP_50_95']:.6f} | {entry['EMA_metric_deltas_lower_lr_minus_original']['AP_50_95']:+.6f} | {entry['AP_50_95_delta_from_epoch3_start']:+.6f} |")
    report.extend(["", f"Selected best: epoch {result['selected_best']['completed_epoch']} with AP@[.50:.95] {result['selected_best']['AP_50_95']:.6f}."])
    (run_dir / "FINDINGS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
