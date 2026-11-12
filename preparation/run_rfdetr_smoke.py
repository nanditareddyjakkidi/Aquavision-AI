#!/usr/bin/env python3
"""Build a read-only COCO adapter and run a bounded RF-DETR Small smoke test.

All adapter images are symlinks to the frozen derived *training* split.  The
project's held-out validation/test images are intentionally never placed in the
adapter.  This program requires explicit opt-in to CPU when MPS is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
FREEZE_FILE = PROJECT / "datasets/processed/FINAL_MANIFEST_FREEZE_v2.sha256"
MANIFEST = PROJECT / "datasets/processed/trashcan_material_plastic_v2_leakage_resolved/split_manifest.csv"
TRAIN_COCO = PROJECT / "datasets/processed/trashcan_material_plastic_v2_leakage_resolved/annotations/train.json"
RFDETR_SOURCE = PROJECT / "third_party/rf-detr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["derived_split"] == "train"]


def select_examples(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    positive = [row for row in rows if row["has_visible_plastic_debris"] == "True"]
    negative = [row for row in rows if row["has_visible_plastic_debris"] == "False"]
    if len(positive) < 2 or len(negative) < 2:
        raise RuntimeError("The frozen derived training split lacks the required positive/negative examples")
    train_rows = [positive[0], negative[0]]
    used_groups = {row["video_group"] for row in train_rows}

    def different_group(candidates: list[dict[str, str]]) -> dict[str, str]:
        for candidate in candidates:
            if candidate["video_group"] not in used_groups:
                used_groups.add(candidate["video_group"])
                return candidate
        raise RuntimeError("Could not form a group-disjoint positive/negative smoke validation subset")

    # Both adapter splits intentionally originate from derived 'train', never from frozen val/test.
    # Their source video groups remain disjoint so this smoke adapter does not create a misleading
    # within-adapter train/validation leakage path.
    return {"train": train_rows, "valid": [different_group(positive[1:]), different_group(negative[1:])]}


def build_adapter(adapter_root: Path) -> dict[str, object]:
    rows_by_path = {row["source_image_path"]: row for row in load_rows()}
    selected = select_examples(list(rows_by_path.values()))
    source_coco = json.loads(TRAIN_COCO.read_text())
    images_by_path = {image["file_name"]: image for image in source_coco["images"]}
    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in source_coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    adapter_info: dict[str, object] = {"splits": {}, "category_mapping": {"1": "visible_plastic_debris"}}
    for split, rows in selected.items():
        split_dir = adapter_root / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True)
        out_images: list[dict[str, object]] = []
        out_annotations: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            source_relative = row["source_image_path"]
            source_path = PROJECT / source_relative
            source_image = images_by_path.get(source_relative)
            if source_image is None:
                raise RuntimeError(f"Selected image not found in derived training COCO: {source_relative}")
            target_name = f"{index:02d}_{Path(source_relative).name}"
            target_path = images_dir / target_name
            target_path.symlink_to(source_path)
            file_name = f"images/{target_name}"
            out_image = dict(source_image)
            out_image["file_name"] = file_name
            out_images.append(out_image)
            selected_annotations = [dict(item) for item in annotations_by_image.get(int(source_image["id"]), [])]
            out_annotations.extend(selected_annotations)
            audit_rows.append({
                "adapter_file": file_name,
                "source_image_path": source_relative,
                "source_split": row["source_split"],
                "derived_split": row["derived_split"],
                "has_visible_plastic_debris": row["has_visible_plastic_debris"],
                "visible_plastic_debris_instances": row["visible_plastic_debris_instances"],
            })
        coco = {
            "info": {"description": "AquaVision bounded RF-DETR smoke adapter; symlinks only"},
            "licenses": source_coco.get("licenses", []),
            "categories": [{"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"}],
            "images": out_images,
            "annotations": out_annotations,
        }
        annotation_path = split_dir / "_annotations.coco.json"
        annotation_path.write_text(json.dumps(coco, indent=2) + "\n")
        (split_dir / "adapter_sources.json").write_text(json.dumps(audit_rows, indent=2) + "\n")
        adapter_info["splits"][split] = {"images": len(out_images), "annotations": len(out_annotations), "examples": audit_rows}

    # Verify the version-specific Roboflow COCO contract before RF-DETR runs.
    for split in ("train", "valid"):
        payload = json.loads((adapter_root / split / "_annotations.coco.json").read_text())
        if payload["categories"] != [{"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"}]:
            raise RuntimeError(f"Unexpected category mapping in smoke adapter {split}")
        image_index = {int(image["id"]): image for image in payload["images"]}
        for image in payload["images"]:
            image_path = adapter_root / split / image["file_name"]
            if not image_path.is_file():
                raise RuntimeError(f"Adapter image link is broken: {image_path}")
            with Image.open(image_path) as decoded:
                if decoded.size != (int(image["width"]), int(image["height"])):
                    raise RuntimeError(f"Adapter image dimensions do not match COCO: {image_path}")
        for annotation in payload["annotations"]:
            if int(annotation["category_id"]) != 1 or int(annotation["image_id"]) not in image_index:
                raise RuntimeError(f"Invalid annotation reference in adapter {split}")
            x, y, width, height = map(float, annotation["bbox"])
            image = image_index[int(annotation["image_id"])]
            if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > float(image["width"]) or y + height > float(image["height"]):
                raise RuntimeError(f"Invalid annotation bounding box in adapter {split}")
    return adapter_info


def checkpoint_candidate(output_dir: Path) -> Path:
    preferred = ["checkpoint_best_total.pth", "checkpoint_best_regular.pth", "last.ckpt"]
    for name in preferred:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    found = sorted(output_dir.rglob("*.pth")) + sorted(output_dir.rglob("*.ckpt"))
    if not found:
        raise RuntimeError(f"RF-DETR did not save a checkpoint under {output_dir}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu-compatibility", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    if args.resolution < 32 or args.resolution % 32:
        raise SystemExit("--resolution must be a multiple of 32 and at least 32 for RF-DETR Small")
    if not (FREEZE_FILE.is_file() and MANIFEST.is_file() and TRAIN_COCO.is_file()):
        raise SystemExit("Frozen v2 detector artifacts are required before a smoke test can run")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        device = torch.device("mps")
        device_note = "MPS selected"
    elif args.allow_cpu_compatibility:
        device = torch.device("cpu")
        device_note = "CPU explicitly selected because MPS is unavailable at runtime"
    else:
        raise SystemExit("MPS is unavailable. Re-run with --allow-cpu-compatibility to perform a labelled CPU test.")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("rfdetr_smoke_%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "smoke_tests" / "rfdetr" / run_id
    if run_root.exists():
        raise SystemExit(f"Refusing to overwrite existing smoke-test output: {run_root}")
    adapter_root = run_root / "coco_adapter"
    output_dir = run_root / "training_output"
    adapter_info = build_adapter(adapter_root)
    start = time.perf_counter()

    from rfdetr import RFDETRSmall

    # The two unique train images are batch-1 for one epoch. RF-DETR's tiny-dataset sampler
    # may repeat them to its documented minimum, so record the actual global step afterward
    # and enforce the ten-step ceiling below.
    model = RFDETRSmall(resolution=args.resolution, device=str(device))
    model.train(
        dataset_dir=str(adapter_root),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=1,
        num_workers=0,
        device=str(device),
        resolution=args.resolution,
        amp_dtype=None,
        augmentation_backend="cpu",
        checkpoint_interval=1,
        save_dataset_grids=False,
        tensorboard=False,
        wandb=False,
        early_stopping=False,
        notes={
            "purpose": "bounded AquaVision compatibility smoke test; not full training",
            "dataset_scope": "two positive/negative train-only symlinks; held-out validation/test excluded",
            "device_note": device_note,
        },
    )
    checkpoint = checkpoint_candidate(output_dir)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    global_step = checkpoint_payload.get("global_step") if isinstance(checkpoint_payload, dict) else None
    if not isinstance(global_step, int) or global_step < 1 or global_step > 10:
        raise RuntimeError(f"RF-DETR smoke run exceeded the 10-step ceiling or saved no global step: {global_step!r}")
    restored = RFDETRSmall.from_checkpoint(str(checkpoint))
    inference_image = next((adapter_root / "valid" / "images").iterdir())
    predictions = restored.predict(str(inference_image), threshold=0.99, shape=(args.resolution, args.resolution), include_source_image=False)
    prediction_count = int(len(predictions))
    config_path = output_dir / "training_config.json"
    training_config = json.loads(config_path.read_text()) if config_path.is_file() else None
    pretrained = PROJECT / ".cache/rfdetr/rf-detr-small.pth"
    report = {
        "status": "passed",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Derived frozen detector TRAIN images only: two train and two valid adapter symlinks, each split containing one positive and one negative. Frozen project validation/test sets were not accessed.",
        "device": str(device),
        "device_note": device_note,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rfdetr_source_revision": git_revision(RFDETR_SOURCE),
        "rfdetr_package_version": "1.12.0.dev0",
        "coco_layout_verified": "train/_annotations.coco.json and valid/_annotations.coco.json, with images beneath each split; test intentionally omitted because fit uses train/valid only",
        "category_mapping_verified": {"COCO_category_id": 1, "class_name": "visible_plastic_debris", "RF-DETR_contiguous_label": 0},
        "adapter": adapter_info,
        "batch_size": 1,
        "epochs": 1,
        "unique_train_images": 2,
        "optimizer_step_ceiling": 10,
        "observed_optimizer_steps": global_step,
        "checkpoint": str(checkpoint.relative_to(PROJECT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": True,
        "checkpoint_global_step": global_step,
        "inference": {"image": str(inference_image.relative_to(PROJECT)), "threshold": 0.99, "prediction_count": prediction_count, "completed": True},
        "forward_loss_backward_optimizer": "completed by the official RF-DETR PyTorch Lightning training path; successful checkpoint written after the group-disjoint two-image batch-1 smoke epoch",
        "training_config_written": str(config_path.relative_to(PROJECT)) if config_path.is_file() else None,
        "pretrained_checkpoint": {
            "path": str(pretrained.relative_to(PROJECT)),
            "exists": pretrained.is_file(),
            "sha256": sha256(pretrained) if pretrained.is_file() else None,
            "origin": "official RF-DETR Small Roboflow asset resolved by installed package",
            "url": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
            "expected_md5": "fb37061c1af7bace359c91b723a8d5c1",
        },
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "process_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "frozen_manifest_sha256": sha256(FREEZE_FILE),
        "training_config": training_config,
    }
    (run_root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
