#!/usr/bin/env python3
"""Continue the fixed-512 raw TrashCan RF-DETR Small pilot through five total epochs.

This is deliberately a continuation, not a weights-only restart. It resumes
the pilot's full Lightning checkpoint (model, optimizer, LR scheduler, loop
state, and EMA callback state) and creates a new, read-only COCO adapter over
the frozen raw-image train/validation splits. Test data and enhanced images are
never mounted in the adapter.
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


PILOT_RUN_ID = "rfdetr_raw_v2_fixed512_mps_20261001T065835Z"
EXPECTED_PILOT_CHECKPOINT_SHA256 = "7a2bb85878d4477a45532d47f26a1405a2a986406f22359fc741d4070d6b52d7"
STEPS_PER_EPOCH = 5048
TOTAL_EPOCHS = 5
PREVIEW_COUNT = 5
EXPECTED_CATEGORY_NAME = "visible_plastic_debris"


def number(row: dict[str, str], key: str) -> float:
    """Read a finite scalar from one CSV row."""
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"metrics.csv lacks {key}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"metrics.csv recorded non-finite {key}")
    return result


def step_as_int(value: Any) -> int:
    """Convert an optimizer-state step value to an integer."""
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def source_paths(pilot_run_id: str) -> dict[str, Path]:
    """Return the immutable pilot artifacts used as the continuation source."""
    pilot_dir = RUNS_ROOT / pilot_run_id
    output_dir = pilot_dir / "training_output"
    paths = {
        "pilot_dir": pilot_dir,
        "output_dir": output_dir,
        "checkpoint": output_dir / "checkpoint_0.ckpt",
        "best_ema": output_dir / "checkpoint_best_ema.pth",
        "result": pilot_dir / "result.json",
        "config": output_dir / "training_config.json",
    }
    missing = [str(path.relative_to(PROJECT)) for key, path in paths.items() if key not in {"pilot_dir", "output_dir"} and not path.is_file()]
    if missing:
        raise RuntimeError(f"The successful fixed-512 pilot is incomplete; missing: {missing}")
    return paths


def inspect_full_checkpoint(path: Path, *, expected_epoch: int, expected_step: int, require_pilot_sha: bool = False) -> dict[str, Any]:
    """Verify a Lightning checkpoint has the state required for a true RF-DETR resume."""
    checkpoint_sha256 = digest(path)
    if require_pilot_sha and checkpoint_sha256 != EXPECTED_PILOT_CHECKPOINT_SHA256:
        raise RuntimeError("The pilot full checkpoint SHA-256 differs from the verified successful pilot checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} is not a dictionary checkpoint")
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "epoch", "global_step", "loops", "model_config"}
    absent = sorted(required.difference(payload))
    if absent:
        raise RuntimeError(f"{path.name} is weights-only or otherwise lacks resume state: {absent}")
    if int(payload["epoch"]) != expected_epoch or int(payload["global_step"]) != expected_step:
        raise RuntimeError(f"{path.name} reports epoch/step {payload['epoch']}/{payload['global_step']}, expected {expected_epoch}/{expected_step}")
    state_dict = payload["state_dict"]
    optimizer_states = payload["optimizer_states"]
    scheduler_states = payload["lr_schedulers"]
    callbacks = payload["callbacks"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"{path.name} has no model state dictionary")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1 or not isinstance(optimizer_states[0], dict):
        raise RuntimeError(f"{path.name} does not contain exactly one optimizer state")
    optimizer_state = optimizer_states[0]
    optimizer_slots = optimizer_state.get("state")
    if not isinstance(optimizer_slots, dict) or not optimizer_slots:
        raise RuntimeError(f"{path.name} has no populated optimizer slots")
    optimizer_steps = {step_as_int(value["step"]) for value in optimizer_slots.values() if isinstance(value, dict) and "step" in value}
    if optimizer_steps != {expected_step}:
        raise RuntimeError(f"{path.name} optimizer steps are {sorted(optimizer_steps)}, expected {expected_step}")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1 or not isinstance(scheduler_states[0], dict):
        raise RuntimeError(f"{path.name} does not contain exactly one scheduler state")
    scheduler_state = scheduler_states[0]
    if int(scheduler_state.get("last_epoch", -1)) != expected_step or int(scheduler_state.get("_step_count", -1)) != expected_step + 1:
        raise RuntimeError(f"{path.name} scheduler does not match global step {expected_step}")
    ema_state = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    if not isinstance(ema_state, dict) or not isinstance(ema_state.get("average_model_state_dict"), dict):
        raise RuntimeError(f"{path.name} does not contain resumable EMA state")
    if int(ema_state.get("latest_update_step", -1)) != expected_step:
        raise RuntimeError(f"{path.name} EMA update step does not match global step {expected_step}")
    return {
        "path": str(path.resolve().relative_to(PROJECT)),
        "sha256": checkpoint_sha256,
        "epoch_zero_based": expected_epoch,
        "completed_epoch": expected_epoch + 1,
        "global_step": expected_step,
        "model_state_keys": len(state_dict),
        "optimizer": {"state_slots": len(optimizer_slots), "param_groups": len(optimizer_state.get("param_groups", [])), "steps": sorted(optimizer_steps)},
        "scheduler": {"last_epoch": int(scheduler_state["last_epoch"]), "step_count": int(scheduler_state["_step_count"])},
        "ema": {"latest_update_step": int(ema_state["latest_update_step"]), "state_keys": len(ema_state["average_model_state_dict"])},
        "rng_state_present": any(key in payload for key in ("rng_state", "rng_states", "python_rng_state", "numpy_rng_state")),
    }


def pilot_scheduler_record(training_config: dict[str, Any], checkpoint_audit: dict[str, Any]) -> dict[str, Any]:
    """Describe the saved scheduler and the explicit effect of the five-epoch target."""
    train = training_config.get("train_config", {})
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
    if mismatches:
        raise RuntimeError(f"Pilot scheduler/configuration does not match the successful fixed-512 run: {mismatches}")
    pilot_steps = checkpoint_audit["global_step"]
    continuation_steps = STEPS_PER_EPOCH * TOTAL_EPOCHS
    return {
        "pilot": {"epochs": int(train.get("epochs", -1)), "total_optimizer_steps": pilot_steps, **expected, "managed_defaults": {"lr_drop": 100, "min_factor": 0.0}},
        "continuation": {"epochs": TOTAL_EPOCHS, "total_optimizer_steps": continuation_steps, "resume_global_step": pilot_steps},
        "effect_of_extending_epoch_budget": {
            "total_steps_changes": True,
            "from": pilot_steps,
            "to": continuation_steps,
            "effective_learning_rate_change_for_epochs_1_to_5": False,
            "reason": "The pinned managed 'step' scheduler has warmup_epochs=0 and its first 10x drop is at epoch 100; epochs 1-5 remain at factor 1.0. The saved optimizer and scheduler counters still resume from step 5048.",
        },
    }


def verify_preserved_configuration(source_config: dict[str, Any], continuation_config: dict[str, Any]) -> dict[str, Any]:
    """Ensure continuation settings match the pilot except for documented resume fields."""
    source_train = source_config.get("train_config")
    continued_train = continuation_config.get("train_config")
    source_model = source_config.get("model_config")
    continued_model = continuation_config.get("model_config")
    if not all(isinstance(value, dict) for value in (source_train, continued_train, source_model, continued_model)):
        raise RuntimeError("Could not compare source and continuation RF-DETR configurations")
    allowed_train_changes = {"epochs", "dataset_dir", "output_dir", "resume", "notes"}
    train_differences = {
        key: {"pilot": source_train.get(key), "continuation": continued_train.get(key)}
        for key in sorted(set(source_train).union(continued_train).difference(allowed_train_changes))
        if source_train.get(key) != continued_train.get(key)
    }
    model_differences = {
        key: {"pilot": source_model.get(key), "continuation": continued_model.get(key)}
        for key in sorted(set(source_model).union(continued_model))
        if source_model.get(key) != continued_model.get(key)
    }
    if train_differences or model_differences:
        raise RuntimeError(f"Continuation changed RF-DETR settings beyond its documented resume fields: train={train_differences}, model={model_differences}")
    return {
        "preserved": True,
        "allowed_train_config_changes": sorted(allowed_train_changes),
        "documented_values": {
            "epochs": {"pilot": source_train["epochs"], "continuation": continued_train["epochs"]},
            "resume": continued_train["resume"],
            "dataset_dir": continued_train["dataset_dir"],
            "output_dir": continued_train["output_dir"],
        },
    }
class EpochTelemetry(Callback):
    """Record MPS memory and elapsed time for every newly completed epoch."""

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
        started, memory_before = self._starts.pop(epoch)
        self.records.append(
            {
                "epoch_zero_based": epoch,
                "completed_epoch": epoch + 1,
                "global_step": int(trainer.global_step),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "mps_memory_before_epoch": memory_before,
                "mps_memory_after_epoch": mps_memory(),
            }
        )


def metrics_history(source_result: dict[str, Any], metrics_path: Path) -> list[dict[str, Any]]:
    """Merge epoch one from the pilot with epochs two through five from CSV metrics."""
    source_metrics = source_result.get("validation_metrics")
    source_losses = source_result.get("losses")
    if not isinstance(source_metrics, dict) or not isinstance(source_losses, dict):
        raise RuntimeError("The source pilot result lacks validation metrics or losses")
    history: list[dict[str, Any]] = [{"completed_epoch": 1, "global_step": STEPS_PER_EPOCH, "source": "successful_fixed512_pilot", "validation": source_metrics, "losses": source_losses}]
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = {int(row["epoch"]): row for row in rows if row.get("val/mAP_50_95") not in (None, "")}
    loss_rows = {int(row["epoch"]): row for row in rows if row.get("train/loss") not in (None, "")}
    if set(validation_rows) != {1, 2, 3, 4} or set(loss_rows) != {1, 2, 3, 4}:
        raise RuntimeError(f"Continuation metrics must contain validation and losses for zero-based epochs 1-4; got validation={sorted(validation_rows)}, losses={sorted(loss_rows)}")
    validation_keys = ("val/mAP_50_95", "val/mAP_50", "val/mAP_75", "val/mAR", "val/F1", "val/precision", "val/recall")
    for epoch in range(1, TOTAL_EPOCHS):
        validation = validation_rows[epoch]
        losses = loss_rows[epoch]
        history.append(
            {
                "completed_epoch": epoch + 1,
                "global_step": int(validation["step"]) + 1,
                "source": "continuation",
                "validation": {"AP_50_95": number(validation, validation_keys[0]), "AP50": number(validation, validation_keys[1]), "AP75": number(validation, validation_keys[2]), "mAR": number(validation, validation_keys[3]), "f1": number(validation, validation_keys[4]), "precision": number(validation, validation_keys[5]), "recall": number(validation, validation_keys[6])},
                "losses": {key.removeprefix("train/"): number(losses, key) for key in losses if key.startswith("train/") and losses[key] not in (None, "")},
            }
        )
    return history


def load_ema_model(checkpoint_path: Path, device: torch.device) -> Any:
    """Load the EMA weights embedded in one full checkpoint for a matching preview."""
    from rfdetr import RFDETRSmall

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    callbacks = payload.get("callbacks")
    ema_callback = callbacks.get("RFDETREMACallback") if isinstance(callbacks, dict) else None
    average = ema_callback.get("average_model_state_dict") if isinstance(ema_callback, dict) else None
    if not isinstance(average, dict):
        raise RuntimeError(f"{checkpoint_path.name} has no EMA state for preview generation")
    ema_state = {key.removeprefix("module.model."): value for key, value in average.items() if key != "n_averaged"}
    model = RFDETRSmall.from_checkpoint(str(checkpoint_path), device=str(device))
    incompatible = model.model.model.load_state_dict(ema_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys or model.model.device.type != "mps":
        raise RuntimeError(f"Could not load the MPS EMA model from {checkpoint_path.name}")
    return model


def fixed_ema_previews(run_dir: Path, adapter_info: dict[str, Any], checkpoint_by_epoch: dict[int, Path], device: torch.device) -> list[dict[str, Any]]:
    """Generate the same positive/negative labeled validation previews for all five epochs."""
    payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    rows = [row for row in load_manifest() if row["derived_split"] == "val"]
    images, annotations = check_coco_payload(payload, rows, "valid")
    images_by_path = {str(image["file_name"]): image for image in images.values()}
    reports: list[dict[str, Any]] = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        checkpoint = checkpoint_by_epoch[epoch]
        torch.mps.empty_cache()
        model = load_ema_model(checkpoint, device)
        preview_root = run_dir / "fixed_validation_previews" / f"epoch_{epoch:03d}_ema"
        previews = create_labeled_previews(model, adapter_info, annotations, images_by_path, preview_root, device)
        reports.append({"completed_epoch": epoch, "weights": "EMA", "checkpoint": str(checkpoint.relative_to(PROJECT)), "checkpoint_sha256": digest(checkpoint), "previews": previews})
        del model
        gc.collect()
        torch.mps.empty_cache()
    return reports


def create_alias(alias: Path, target: Path) -> dict[str, str]:
    """Create a stable, relative checkpoint alias in the fresh continuation directory."""
    if alias.exists() or alias.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing checkpoint alias: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(os.path.relpath(target, alias.parent))
    if alias.resolve() != target.resolve():
        raise RuntimeError(f"Checkpoint alias did not resolve to its intended target: {alias}")
    return {"alias": str(alias.relative_to(PROJECT)), "target": str(target.relative_to(PROJECT)), "target_sha256": digest(target)}


def main() -> None:
    """Run the four remaining epochs and generate verified continuation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/")
    parser.add_argument("--pilot-run-id", default=PILOT_RUN_ID)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    args = parser.parse_args()
    if args.epochs != TOTAL_EPOCHS or args.resolution != 512 or args.batch_size != 1 or args.preview_count != PREVIEW_COUNT:
        raise SystemExit("This continuation requires --epochs 5 --resolution 512 --batch-size 1 --preview-count 5")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    device = require_mps()
    sources = source_paths(args.pilot_run_id)
    source_result = json.loads(sources["result"].read_text(encoding="utf-8"))
    source_config = json.loads(sources["config"].read_text(encoding="utf-8"))
    source_audit = inspect_full_checkpoint(sources["checkpoint"], expected_epoch=0, expected_step=STEPS_PER_EPOCH, require_pilot_sha=True)
    scheduler_record = pilot_scheduler_record(source_config, source_audit)
    entries = frozen_entries()
    frozen_checks = {str(path.relative_to(PROJECT)): verify_frozen_file(path, entries) for path in (TRAIN_COCO, VAL_COCO, CLASS_MAPPING)}
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    adapter_info = build_adapter(run_dir, args.preview_count)
    adapter_info["description"] = "Read-only symlink adapter for a full-state RF-DETR continuation. Every image target is a frozen raw material_version image; no enhanced image or test image is present."
    output_dir = run_dir / "training_output"
    started = time.perf_counter()
    memory_before = mps_memory()

    from rfdetr import RFDETRSmall
    import rfdetr.training as training_stack

    model = RFDETRSmall(resolution=512, device="mps")
    telemetry = EpochTelemetry()
    original_build_trainer = training_stack.build_trainer

    def build_trainer_with_telemetry(*trainer_args: Any, **trainer_kwargs: Any) -> Any:
        trainer = original_build_trainer(*trainer_args, **trainer_kwargs)
        trainer.callbacks.append(telemetry)
        return trainer

    launch_config = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Continuation from the successful RF-DETR raw-image fixed-512 pilot. Frozen train/validation raw material_version only; test split absent; no enhanced images.",
        "device_requirement": "MPS only; PYTORCH_ENABLE_MPS_FALLBACK=0",
        "source_full_checkpoint": source_audit,
        "resume_semantics": {"full_state_resume": True, "rng_state_saved": source_audit["rng_state_present"], "rng_limitation": "The full Lightning checkpoint has no Python/NumPy/Torch RNG snapshot key. Model, optimizer, scheduler, loop, callback, and EMA state resume; bitwise replay of random augmentation/shuffle state is not claimed."},
        "scheduler": scheduler_record,
        "frozen_checks": frozen_checks,
        "adapter": adapter_info,
        "training_call": {"epochs": TOTAL_EPOCHS, "batch_size": 1, "resolution": 512, "multi_scale": "off", "expanded_scales": False, "device": "mps", "resume": str(sources["checkpoint"].relative_to(PROJECT)), "checkpoint_interval": 1, "eval_interval": 1, "use_ema": True},
        "rfdetr": {"source_revision": git_revision(RFDETR_SOURCE)},
    }
    (run_dir / "continuation_config.json").write_text(json.dumps(launch_config, indent=2) + "\n", encoding="utf-8")

    training_stack.build_trainer = build_trainer_with_telemetry
    try:
        model.train(
            dataset_dir=str(run_dir / "raw_coco_adapter"), output_dir=str(output_dir), epochs=TOTAL_EPOCHS,
            batch_size=1, num_workers=0, device="mps", resolution=512, resume=str(sources["checkpoint"]),
            optimizer="adamw", lr=0.0001, lr_encoder=0.00015, lr_component_decay=0.7, lr_vit_layer_decay=0.8,
            weight_decay=0.0001, clip_max_norm=0.1, lr_scheduler="step", lr_scheduler_interval="step",
            lr_scheduler_kwargs={}, warmup_epochs=0.0, grad_accum_steps=1, amp_dtype=None,
            augmentation_backend="cpu", multi_scale="off", expanded_scales=False, scale_jitter=True,
            square_resize_div_64=True, use_ema=True, ema_decay=0.993, ema_tau=100, ema_update_interval=1,
            checkpoint_interval=1, eval_interval=1, eval_base_model=False, eval_ema_only=False,
            compute_val_loss="auto", log_per_class_metrics=True, save_dataset_grids=False, tensorboard=False,
            wandb=False, early_stopping=False, run_test=False, seed=args.seed,
            notes={"dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used", "purpose": "AquaVision RF-DETR Small full-state continuation from the successful fixed-512 raw-image pilot"},
        )
    finally:
        training_stack.build_trainer = original_build_trainer
    torch.mps.synchronize()
    memory_after_training = mps_memory()
    if digest(sources["checkpoint"]) != EXPECTED_PILOT_CHECKPOINT_SHA256:
        raise RuntimeError("The source full checkpoint changed during continuation; refusing to report results")

    resolved = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    train = resolved.get("train_config", {})
    model_config = resolved.get("model_config", {})
    expected_train = {"epochs": 5, "batch_size": 1, "multi_scale": "off", "expanded_scales": False, "eval_interval": 1, "checkpoint_interval": 1, "run_test": False, "use_ema": True, "lr_scheduler": "step", "lr_scheduler_interval": "step", "warmup_epochs": 0.0, "seed": args.seed}
    errors = {key: {"expected": value, "actual": train.get(key)} for key, value in expected_train.items() if train.get(key) != value}
    if resolved.get("class_names") != [EXPECTED_CATEGORY_NAME] or model_config.get("resolution") != 512 or model_config.get("device") != "mps" or errors:
        raise RuntimeError(f"Resolved continuation configuration differs from the required fixed-512 MPS continuation: {errors}")
    preservation_record = verify_preserved_configuration(source_config, resolved)

    checkpoints = {1: sources["checkpoint"]}
    continuation_audits: list[dict[str, Any]] = []
    for epoch in range(2, TOTAL_EPOCHS + 1):
        checkpoint = output_dir / f"checkpoint_{epoch - 1}.ckpt"
        checkpoints[epoch] = checkpoint
        continuation_audits.append(inspect_full_checkpoint(checkpoint, expected_epoch=epoch - 1, expected_step=epoch * STEPS_PER_EPOCH))
    history = metrics_history(source_result, output_dir / "metrics.csv")
    best_record = max(history, key=lambda record: float(record["validation"]["AP_50_95"]))
    best_target = sources["best_ema"] if int(best_record["completed_epoch"]) == 1 else output_dir / "checkpoint_best_ema.pth"
    if not best_target.is_file():
        raise RuntimeError("RF-DETR did not save the expected best EMA checkpoint")
    del model
    gc.collect()
    torch.mps.empty_cache()
    preview_reports = fixed_ema_previews(run_dir, adapter_info, checkpoints, device)
    torch.mps.synchronize()
    memory_after_previews = mps_memory()
    aliases = {
        "latest_full_state": create_alias(run_dir / "checkpoints" / "latest_full_state.ckpt", checkpoints[TOTAL_EPOCHS]),
        "best_validation_ap_50_95_ema": create_alias(run_dir / "checkpoints" / "best_validation_ap_50_95_ema.pth", best_target),
    }
    result = {
        "status": "passed", "run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_full_checkpoint": source_audit, "resume_semantics": launch_config["resume_semantics"],
        "scheduler": scheduler_record,
        "resolved_configuration": {"training_config": str((output_dir / "training_config.json").relative_to(PROJECT)), "class_names": resolved["class_names"], "resolution": model_config["resolution"], "device": model_config["device"], "batch_size": train["batch_size"], "multi_scale": train["multi_scale"], "expanded_scales": train["expanded_scales"], "preservation": preservation_record},
        "frozen_checks": frozen_checks, "metric_history": history, "new_epoch_telemetry": telemetry.records,
        "new_full_checkpoints": continuation_audits,
        "selected_best_validation_ap_50_95": {"completed_epoch": best_record["completed_epoch"], "AP_50_95": best_record["validation"]["AP_50_95"], "source": best_record["source"], "checkpoint": str(best_target.relative_to(PROJECT)), "checkpoint_sha256": digest(best_target)},
        "checkpoint_aliases": aliases, "fixed_ema_validation_previews": preview_reports,
        "timing_seconds": {"continuation_total": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_training": memory_after_training, "after_fixed_previews": memory_after_previews, "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    (run_dir / "continuation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metric_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
