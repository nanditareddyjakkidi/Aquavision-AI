#!/usr/bin/env python3
"""Run one raw-image, MPS-only RF-DETR Small pilot on frozen TrashCan v2.

The runner creates a separate Roboflow-COCO adapter made exclusively of
symlinks to the frozen *raw* material_version train/validation images. Test
images are deliberately absent. It requires a verified official RF-DETR Small
COCO-pretrained checkpoint, runs exactly one epoch at 512px and batch size 1,
then records official validation metrics, losses, memory, checkpoint reload,
and labeled validation previews.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 for this MPS-only RF-DETR pilot.")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT / "training_runs/rfdetr"
DATA_ROOT = PROJECT / "datasets/processed/trashcan_material_plastic_v2_leakage_resolved"
MANIFEST = DATA_ROOT / "split_manifest.csv"
TRAIN_COCO = DATA_ROOT / "annotations/train.json"
VAL_COCO = DATA_ROOT / "annotations/val.json"
CLASS_MAPPING = DATA_ROOT / "class_mapping.json"
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
RFDETR_SOURCE = PROJECT / "third_party/rf-detr"
EXPECTED_CATEGORY = {"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"}
EXPECTED_COUNTS = {
    "train": {"images": 5048, "positive_images": 1242, "negative_images": 3806, "annotations": 1299},
    "valid": {"images": 1082, "positive_images": 249, "negative_images": 833, "annotations": 289},
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def project_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise an absolute test path."""
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def mps_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        try:
            result[name] = int(getattr(torch.mps, name)())
        except (AttributeError, RuntimeError):
            result[name] = None
    return result


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise SystemExit("MPS is required and unavailable. This runner will not fall back to CPU.")
    device = torch.device("mps")
    probe = torch.ones(1, device=device)
    if probe.device.type != "mps":
        raise RuntimeError("MPS probe tensor was not placed on MPS")
    return device


def frozen_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in FREEZE_FILE.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split(maxsplit=1)
        entries[relative] = checksum
    return entries


def verify_frozen_file(path: Path, entries: dict[str, str]) -> str:
    relative = str(path.relative_to(PROJECT))
    expected = entries.get(relative)
    actual = digest(path)
    if expected is None or expected != actual:
        raise RuntimeError(f"Frozen file does not match its recorded checksum: {relative}")
    return actual


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_coco_payload(
    payload: dict[str, Any], rows: list[dict[str, str]], split: str
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    expected = EXPECTED_COUNTS[split]
    if payload.get("categories") != [EXPECTED_CATEGORY] or len(payload.get("images", [])) != expected["images"] or len(payload.get("annotations", [])) != expected["annotations"]:
        raise RuntimeError(f"Frozen {split} COCO contents do not match the verified one-class specification")
    images = {int(image["id"]): image for image in payload["images"]}
    if len(images) != len(payload["images"]):
        raise RuntimeError(f"Frozen {split} COCO contains duplicate image IDs")
    row_paths = {row["source_image_path"] for row in rows}
    image_paths = {str(image["file_name"]) for image in images.values()}
    if row_paths != image_paths:
        raise RuntimeError(f"Frozen {split} manifest and COCO image paths disagree")
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if int(annotation["category_id"]) != 1 or image_id not in images:
            raise RuntimeError(f"Frozen {split} COCO has an invalid annotation category/image reference")
        x, y, width, height = map(float, annotation["bbox"])
        image = images[image_id]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > float(image["width"]) or y + height > float(image["height"]):
            raise RuntimeError(f"Frozen {split} COCO has an invalid bounding box")
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return images, annotations_by_image


def selected_preview_rows(rows: list[dict[str, str]], split: str, count: int) -> list[dict[str, str]]:
    positives = sorted((row for row in rows if row["has_visible_plastic_debris"] == "True"), key=lambda row: row["source_image_path"])
    negatives = sorted((row for row in rows if row["has_visible_plastic_debris"] == "False"), key=lambda row: row["source_image_path"])
    selection: list[dict[str, str]] = []
    for index in range(count):
        source = positives[index // 2] if index % 2 == 0 else negatives[index // 2]
        selection.append(source)
    if len(selection) != count or len({row["source_image_path"] for row in selection}) != count:
        raise RuntimeError(f"Could not create {count} fixed positive/negative {split} previews")
    return selection


def build_adapter(run_dir: Path, preview_count: int) -> dict[str, Any]:
    entries = frozen_entries()
    freeze_checks = {
        str(path.relative_to(PROJECT)): verify_frozen_file(path, entries)
        for path in (MANIFEST, TRAIN_COCO, VAL_COCO, CLASS_MAPPING)
    }
    mapping = json.loads(CLASS_MAPPING.read_text(encoding="utf-8"))
    if mapping.get("derived_category") != EXPECTED_CATEGORY or mapping.get("negative_example_policy") != "Retain every source image without a trash_plastic annotation as a negative image.":
        raise RuntimeError("Class mapping does not match visible_plastic_debris plus negative-image policy")

    manifest = load_manifest()
    rows_by_split = {
        "train": [row for row in manifest if row["derived_split"] == "train"],
        "valid": [row for row in manifest if row["derived_split"] == "val"],
    }
    if len(rows_by_split["train"]) != EXPECTED_COUNTS["train"]["images"] or len(rows_by_split["valid"]) != EXPECTED_COUNTS["valid"]["images"]:
        raise RuntimeError("Frozen detector split manifest has unexpected train/validation counts")
    if any(row["derived_split"] == "test" for row in rows_by_split["train"] + rows_by_split["valid"]):
        raise RuntimeError("A test row was selected for the raw-image RF-DETR adapter")

    coco_by_split = {
        "train": json.loads(TRAIN_COCO.read_text(encoding="utf-8")),
        "valid": json.loads(VAL_COCO.read_text(encoding="utf-8")),
    }
    adapter_root = run_dir / "raw_coco_adapter"
    adapter_info: dict[str, Any] = {
        "description": "Read-only symlink adapter. Every image target is a frozen raw TrashCan material_version image; no enhanced image or test image is present.",
        "splits": {},
        "fixed_validation_previews": [],
        "freeze_checks": freeze_checks,
        "category_mapping": {"coco_category_id": 1, "class_name": "visible_plastic_debris", "rfdetr_contiguous_label": 0},
    }
    for split, folder in (("train", "train"), ("valid", "valid")):
        rows = rows_by_split[split]
        images, annotations_by_image = check_coco_payload(coco_by_split[split], rows, split)
        split_dir = adapter_root / folder
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=False)
        row_by_path = {row["source_image_path"]: row for row in rows}
        adapter_images: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        verified_raw_images = 0
        for image_id in sorted(images):
            image = images[image_id]
            source_relative = str(image["file_name"])
            row = row_by_path[source_relative]
            source_path = (PROJECT / source_relative).resolve()
            if not source_path.is_file() or not source_path.is_relative_to(PROJECT / "datasets/trashcan/material_version"):
                raise RuntimeError(f"Adapter source is not an in-project raw material_version image: {source_relative}")
            if digest(source_path) != row["sha256"]:
                raise RuntimeError(f"Raw source checksum changed since the frozen split: {source_relative}")
            with Image.open(source_path) as decoded:
                decoded.verify()
            with Image.open(source_path) as decoded:
                if decoded.size != (int(image["width"]), int(image["height"])):
                    raise RuntimeError(f"Raw source dimensions changed since COCO validation: {source_relative}")
            target_name = f"{image_id:07d}_{Path(source_relative).name}"
            target_path = images_dir / target_name
            target_path.symlink_to(source_path)
            adapter_image = dict(image)
            adapter_image["file_name"] = f"images/{target_name}"
            adapter_images.append(adapter_image)
            sources.append(
                {
                    "adapter_file": adapter_image["file_name"],
                    "source_image_path": source_relative,
                    "raw_image_sha256": row["sha256"],
                    "source_split": row["source_split"],
                    "derived_split": row["derived_split"],
                    "video_group": row["video_group"],
                    "has_visible_plastic_debris": row["has_visible_plastic_debris"],
                    "visible_plastic_debris_instances": row["visible_plastic_debris_instances"],
                }
            )
            verified_raw_images += 1
        adapter_payload = {
            "info": {"description": "AquaVision RF-DETR raw-image v2 pilot adapter; symlinks only"},
            "licenses": coco_by_split[split].get("licenses", []),
            "categories": [EXPECTED_CATEGORY],
            "images": adapter_images,
            "annotations": coco_by_split[split]["annotations"],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(adapter_payload, indent=2) + "\n", encoding="utf-8")
        (split_dir / "adapter_sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
        positives = sum(row["has_visible_plastic_debris"] == "True" for row in rows)
        negatives = sum(row["has_visible_plastic_debris"] == "False" for row in rows)
        instances = sum(int(row["visible_plastic_debris_instances"]) for row in rows)
        expected = EXPECTED_COUNTS[split]
        if (positives, negatives, instances) != (expected["positive_images"], expected["negative_images"], expected["annotations"]):
            raise RuntimeError(f"Unexpected positive/negative/instance counts for {split}")
        adapter_info["splits"][folder] = {
            "raw_images": verified_raw_images,
            "positive_images": positives,
            "negative_images": negatives,
            "annotations": instances,
            "groups": len({row["video_group"] for row in rows}),
            "annotations_file": project_path(split_dir / "_annotations.coco.json"),
        }
    adapter_info["fixed_validation_previews"] = selected_preview_rows(rows_by_split["valid"], "validation", preview_count)
    if (adapter_root / "test").exists() or (adapter_root / "valid" / "test").exists():
        raise RuntimeError("The raw-image pilot adapter must not contain any test split")
    return adapter_info


def checkpoint_candidate(output_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth", "last.ckpt", "checkpoint_0.ckpt"):
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    found = sorted(output_dir.rglob("*.pth")) + sorted(output_dir.rglob("*.ckpt"))
    if not found:
        raise RuntimeError("RF-DETR did not save a checkpoint")
    return found[0]


def parse_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise RuntimeError("RF-DETR did not write metrics.csv")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    val_rows = [row for row in rows if row.get("val/mAP_50_95") not in (None, "")]
    loss_rows = [row for row in rows if row.get("train/loss") not in (None, "")]
    if not val_rows or not loss_rows:
        raise RuntimeError("RF-DETR metrics.csv lacks the required validation metrics or training losses")
    val_row, loss_row = val_rows[-1], loss_rows[-1]
    def number(row: dict[str, str], key: str) -> float:
        value = row.get(key)
        if value in (None, ""):
            raise RuntimeError(f"RF-DETR metrics.csv lacks {key}")
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError(f"RF-DETR recorded non-finite {key}")
        return result
    validation = {
        "AP_50_95": number(val_row, "val/mAP_50_95"),
        "AP50": number(val_row, "val/mAP_50"),
        "AP75": number(val_row, "val/mAP_75"),
        "mAR": number(val_row, "val/mAR"),
        "f1": number(val_row, "val/F1"),
        "precision": number(val_row, "val/precision"),
        "recall": number(val_row, "val/recall"),
    }
    losses = {key.removeprefix("train/"): number(loss_row, key) for key in loss_row if key.startswith("train/") and loss_row[key] not in (None, "")}
    size_keys = [key for key in val_row if key.startswith("val/") and any(token in key.lower() for token in ("small", "medium", "large")) and val_row[key] not in (None, "")]
    validation["per_size"] = ({key.removeprefix("val/"): number(val_row, key) for key in size_keys} if size_keys else {"available": False, "reason": "The installed pinned RF-DETR callback logs aggregate AP/AP50/AP75/mAR, but no small/medium/large AP columns in metrics.csv."})
    return {"validation": validation, "losses": losses, "metrics_rows": len(rows)}


def create_labeled_previews(model: Any, adapter_info: dict[str, Any], annotations: dict[int, list[dict[str, Any]]], images_by_path: dict[str, dict[str, Any]], run_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    output_dir = run_dir / "validation_previews"
    output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    reports: list[dict[str, Any]] = []
    for index, row in enumerate(adapter_info["fixed_validation_previews"], start=1):
        source_relative = row["source_image_path"]
        source_path = PROJECT / source_relative
        prediction = model.predict(str(source_path), threshold=0.25, shape=(512, 512), include_source_image=False)
        if model.model.device.type != "mps":
            raise RuntimeError("Reloaded RF-DETR inference left MPS; refusing CPU fallback")
        boxes = np.asarray(prediction.xyxy, dtype=np.float32)
        scores = np.asarray(prediction.confidence, dtype=np.float32)
        classes = np.asarray(prediction.class_id, dtype=np.int64)
        if not (np.isfinite(boxes).all() and np.isfinite(scores).all()) or any(class_id != 0 for class_id in classes):
            raise RuntimeError("Reloaded RF-DETR inference produced invalid one-class predictions")
        image_record = images_by_path[source_relative]
        with Image.open(source_path) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for annotation in annotations.get(int(image_record["id"]), []):
            x, y, width, height = map(float, annotation["bbox"])
            draw.rectangle((x, y, x + width, y + height), outline="lime", width=3)
            draw.text((x + 2, y + 2), "GT visible_plastic_debris", fill="lime", font=font, stroke_width=1, stroke_fill="black")
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), f"pred plastic {score:.2f}", fill="red", font=font, stroke_width=1, stroke_fill="black")
        header = f"raw validation image | green=GT | red=RF-DETR prediction | {'positive' if row['has_visible_plastic_debris'] == 'True' else 'negative'}"
        draw.rectangle((0, 0, canvas.width, 18), fill="black")
        draw.text((3, 3), header, fill="white", font=font)
        output = output_dir / f"{index:02d}_{Path(source_relative).stem}.png"
        canvas.save(output)
        reports.append({"source_image_path": source_relative, "preview": str(output.relative_to(PROJECT)), "ground_truth_instances": len(annotations.get(int(image_record["id"]), [])), "prediction_count_threshold_0_25": int(len(prediction))})
    torch.mps.synchronize()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New directory name under training_runs/rfdetr/")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--preview-count", type=int, default=5)
    args = parser.parse_args()
    if args.epochs != 1 or args.resolution != 512 or args.batch_size != 1 or args.preview_count != 5:
        raise SystemExit("This raw-image pilot requires --epochs 1 --resolution 512 --batch-size 1 --preview-count 5")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    required = [FREEZE_FILE, MANIFEST, TRAIN_COCO, VAL_COCO, CLASS_MAPPING, RFDETR_SOURCE]
    if not all(path.exists() for path in required):
        raise SystemExit("Frozen TrashCan v2 artifacts and pinned RF-DETR source are required")
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite an existing RF-DETR training run: {run_dir}")
    expected_cache = (PROJECT / ".cache/rfdetr").resolve()
    if Path(os.environ.get("RF_HOME", "")).resolve() != expected_cache:
        raise SystemExit("RF_HOME must be set to this project's .cache/rfdetr directory")
    device = require_mps()
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)

    import importlib.metadata as metadata
    import rfdetr
    from rfdetr import RFDETRSmall
    from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir, validate_pretrain_weights

    if not Path(rfdetr.__file__).resolve().is_relative_to(RFDETR_SOURCE.resolve()):
        raise RuntimeError("The installed RF-DETR package is not the pinned project-local official source")
    if Path(get_model_cache_dir()).resolve() != expected_cache:
        raise RuntimeError("RF-DETR did not resolve its weights cache inside this project")
    asset = ModelWeights.from_filename("rf-detr-small.pth")
    if asset is None or asset.md5_hash is None:
        raise RuntimeError("Pinned RF-DETR source does not expose the official Small pretrained-weight checksum")

    # Build the adapter before model construction so a missing/corrupt raw source is
    # reported before training and no test image is ever presented to RF-DETR.
    run_dir.mkdir(parents=True)
    adapter_info = build_adapter(run_dir, args.preview_count)
    output_dir = run_dir / "training_output"
    started = time.perf_counter()
    memory_before = mps_memory()
    model = RFDETRSmall(resolution=args.resolution, device=str(device))
    pretrained = Path(model.model_config.pretrain_weights)
    if pretrained.resolve().parent != expected_cache or not pretrained.is_file() or not validate_pretrain_weights(str(pretrained), strict=True):
        raise RuntimeError("RF-DETR Small pretrained weights are not present and MD5-verified in the project cache")
    if model.model_config.resolution != 512 or model.model_config.device != "mps":
        raise RuntimeError("RF-DETR Small did not retain the requested 512px MPS model configuration")
    launch_config = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen TrashCan v2 raw image train/validation only; test split is absent and no enhancement output is used.",
        "device_requirement": "MPS only; PYTORCH_ENABLE_MPS_FALLBACK=0",
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "resolution": args.resolution,
        "seed": args.seed,
        "rfdetr": {"package_version": metadata.version("rfdetr"), "source_revision": git_revision(RFDETR_SOURCE), "source_path": str(RFDETR_SOURCE.relative_to(PROJECT))},
        "pretrained_weights": {"path": str(pretrained.relative_to(PROJECT)), "sha256": digest(pretrained), "md5": digest(pretrained, "md5"), "official_expected_md5": asset.md5_hash, "official_url": asset.url},
        "frozen_manifest_sha256": digest(FREEZE_FILE),
        "adapter": adapter_info,
        "training_call": {"amp_dtype": None, "augmentation_backend": "cpu", "multi_scale": "off", "expanded_scales": False, "checkpoint_interval": 1, "eval_interval": 1, "log_per_class_metrics": True, "run_test": False, "early_stopping": False},
    }
    (run_dir / "pilot_config.json").write_text(json.dumps(launch_config, indent=2) + "\n", encoding="utf-8")

    model.train(
        dataset_dir=str(run_dir / "raw_coco_adapter"),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=1,
        num_workers=0,
        device="mps",
        resolution=512,
        # RF-DETR otherwise defaults to per-batch multi-scale tensors.
        # This requested baseline must present fixed 512px tensors.
        multi_scale="off",
        expanded_scales=False,
        amp_dtype=None,
        augmentation_backend="cpu",
        checkpoint_interval=1,
        eval_interval=1,
        log_per_class_metrics=True,
        save_dataset_grids=False,
        tensorboard=False,
        wandb=False,
        early_stopping=False,
        run_test=False,
        seed=args.seed,
        notes={"purpose": "one-epoch AquaVision raw-image RF-DETR Small baseline pilot", "dataset_scope": "frozen v2 raw material_version train/validation; negatives retained; test absent", "enhancement_input": "not used"},
    )
    torch.mps.synchronize()
    memory_after_training = mps_memory()
    metrics = parse_metrics(output_dir / "metrics.csv")
    training_config_path = output_dir / "training_config.json"
    if not training_config_path.is_file():
        raise RuntimeError("RF-DETR did not save its resolved training_config.json")
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    resolved_train = training_config.get("train_config", {})
    resolved_model = training_config.get("model_config", {})
    if (
        training_config.get("model_config_type") != "RFDETRSmallConfig"
        or training_config.get("class_names") != ["visible_plastic_debris"]
        or training_config.get("num_classes") != 1
        or resolved_train.get("batch_size") != 1
        or resolved_train.get("epochs") != 1
        or resolved_train.get("run_test") is not False
        or resolved_train.get("multi_scale") != "off"
        or resolved_train.get("expanded_scales") is not False
        or resolved_model.get("resolution") != 512
        or resolved_model.get("device") != "mps"
    ):
        raise RuntimeError("Resolved RF-DETR configuration differs from the required one-class raw-image MPS pilot")
    checkpoint = checkpoint_candidate(output_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    global_step = payload.get("global_step") if isinstance(payload, dict) else None
    if not isinstance(global_step, int) or global_step < 1:
        raise RuntimeError("RF-DETR checkpoint has no recorded optimizer progress")
    restored = RFDETRSmall.from_checkpoint(str(checkpoint), device="mps")

    validation_payload = json.loads(VAL_COCO.read_text(encoding="utf-8"))
    images_by_path = {str(image["file_name"]): image for image in validation_payload["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in validation_payload["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    previews = create_labeled_previews(restored, adapter_info, annotations_by_image, images_by_path, run_dir, device)
    if restored.model.device.type != "mps":
        raise RuntimeError("Reloaded RF-DETR model did not run inference on MPS")
    memory_after_inference = mps_memory()
    result = {
        "status": "passed",
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": launch_config["scope"],
        "device": "mps",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rfdetr": launch_config["rfdetr"],
        "pretrained_weights": launch_config["pretrained_weights"],
        "settings_verified_against_pinned_code": {"model": "RFDETRSmall", "resolution": 512, "training_tensor_resolution": 512, "multi_scale": "off", "expanded_scales": False, "resolution_block_size": 32, "batch_size": 1, "coco_layout": "train/_annotations.coco.json and valid/_annotations.coco.json", "category_mapping": {"COCO_category_id": 1, "RF-DETR_contiguous_label": 0, "class_name": "visible_plastic_debris"}, "negative_images_retained": True, "test_adapter_absent": True},
        "adapter": adapter_info,
        "training_config": str(training_config_path.relative_to(PROJECT)),
        "validation_metrics": metrics["validation"],
        "losses": metrics["losses"],
        "checkpoint": {"path": str(checkpoint.relative_to(PROJECT)), "sha256": digest(checkpoint), "global_step": global_step, "reload": True},
        "inference": {"device": "mps", "preview_count": len(previews), "labeled_previews": previews},
        "timing_seconds": {"total": round(time.perf_counter() - started, 3)},
        "memory": {"before_training": memory_before, "after_training": memory_after_training, "after_reload_inference": memory_after_inference, "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "frozen_manifest_sha256": digest(FREEZE_FILE),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
