#!/usr/bin/env python3
"""Create non-destructive, reproducible AquaVision training preparations.

Inputs are copied datasets plus their validation manifests. Outputs are written
only below datasets/processed/. The script refuses to overwrite an existing
preparation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
DATASETS = ROOT / "datasets"
TRASHCAN_MATERIAL = DATASETS / "trashcan" / "material_version"
UIEB = DATASETS / "uieb"
VALIDATION = ROOT / "validation"
PROCESSED = DATASETS / "processed"
MATERIAL_OUTPUT = PROCESSED / "trashcan_material_plastic_v1"
UIEB_OUTPUT = PROCESSED / "uieb_enhancement_v1"
VIDEO_NAME = re.compile(r"^vid_(\d+)_frame(\d+)\.jpg$", re.IGNORECASE)
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MATERIAL_SEED = "AquaVision-material-plastic-v1-20260930"
UIEB_SEED = "AquaVision-uieb-v1-20260930"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_image_manifest() -> dict[str, dict[str, Any]]:
    path = VALIDATION / "image_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior image manifest: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["path"]] = row
    return rows


def split_cost(
    counts: dict[str, dict[str, int]], targets: dict[str, dict[str, float]],
    total_images: int, total_positive_images: int, total_instances: int,
) -> float:
    # Image count is the primary target. Positive-image and instance terms keep
    # the single detection class represented without overclaiming stratification.
    image_cost = sum((counts[name]["images"] - targets[name]["images"]) ** 2 for name in SPLITS) / max(total_images**2, 1)
    positive_cost = sum((counts[name]["positive_images"] - targets[name]["positive_images"]) ** 2 for name in SPLITS) / max(total_positive_images**2, 1)
    instance_cost = sum((counts[name]["instances"] - targets[name]["instances"]) ** 2 for name in SPLITS) / max(total_instances**2, 1)
    return image_cost + 0.30 * positive_cost + 0.45 * instance_cost


def allocate_whole_groups(groups: list[dict[str, Any]], seed: str) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Assign immutable groups using deterministic greedy balancing plus moves."""
    total_images = sum(group["images"] for group in groups)
    total_positive = sum(group.get("positive_images", 0) for group in groups)
    total_instances = sum(group.get("instances", 0) for group in groups)
    targets = {
        split: {
            "images": total_images * SPLIT_RATIOS[split],
            "positive_images": total_positive * SPLIT_RATIOS[split],
            "instances": total_instances * SPLIT_RATIOS[split],
        }
        for split in SPLITS
    }
    counts = {split: {"images": 0, "positive_images": 0, "instances": 0} for split in SPLITS}
    assignments: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: (-group.get("instances", 0), -group.get("positive_images", 0), -group["images"], stable_key(seed, group["id"])),
    )
    for group in ordered:
        best_split = None
        best_cost = None
        for split in SPLITS:
            candidate = {name: dict(values) for name, values in counts.items()}
            for metric in candidate[split]:
                candidate[split][metric] += group.get(metric, 0)
            value = split_cost(candidate, targets, total_images, total_positive, total_instances)
            tie = (value, SPLITS.index(split), stable_key(seed, f"{group['id']}:{split}"))
            if best_cost is None or tie < best_cost:
                best_cost, best_split = tie, split
        assignments[group["id"]] = best_split  # type: ignore[assignment]
        for metric in counts[best_split]:  # type: ignore[index]
            counts[best_split][metric] += group.get(metric, 0)  # type: ignore[index]

    # A small deterministic local-improvement pass makes the image-ratio target
    # closer without ever splitting a group.
    for _ in range(10):
        changed = False
        for group in ordered:
            current = assignments[group["id"]]
            baseline = split_cost(counts, targets, total_images, total_positive, total_instances)
            best = current
            best_value = baseline
            for candidate_split in SPLITS:
                if candidate_split == current:
                    continue
                candidate = {name: dict(values) for name, values in counts.items()}
                for metric in candidate[current]:
                    candidate[current][metric] -= group.get(metric, 0)
                    candidate[candidate_split][metric] += group.get(metric, 0)
                value = split_cost(candidate, targets, total_images, total_positive, total_instances)
                if value + 1e-14 < best_value:
                    best, best_value = candidate_split, value
            if best != current:
                for metric in counts[current]:
                    counts[current][metric] -= group.get(metric, 0)
                    counts[best][metric] += group.get(metric, 0)
                assignments[group["id"]] = best
                changed = True
        if not changed:
            break
    return assignments, counts, targets


def load_material_sources(image_manifest: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    records: list[dict[str, Any]] = []
    categories: dict[int, str] = {}
    source_annotation_files: dict[str, str] = {}
    for source_split in ("train", "val"):
        annotation_file = TRASHCAN_MATERIAL / f"instances_{source_split}_trashcan.json"
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories.update({int(row["id"]): str(row["name"]) for row in data["categories"]})
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        source_annotation_files[source_split] = rel(annotation_file)
        for row in data["images"]:
            filename = str(row["file_name"])
            match = VIDEO_NAME.match(filename)
            image_path = TRASHCAN_MATERIAL / source_split / filename
            image_rel = rel(image_path)
            if image_rel not in image_manifest:
                raise ValueError(f"Image is absent from validated image manifest: {image_rel}")
            records.append(
                {
                    "source_split": source_split,
                    "source_annotation_file": rel(annotation_file),
                    "source_image_id": int(row["id"]),
                    "source_filename": filename,
                    "source_image_path": image_rel,
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "video_group": f"video:{match.group(1)}" if match else None,
                    "frame_number": int(match.group(2)) if match else None,
                    "group_confidence": "verified_filename_rule" if match else "uncertain",
                    "annotations": annotations_by_image[int(row["id"])],
                    "sha256": image_manifest[image_rel]["sha256"],
                    "dhash": image_manifest[image_rel]["dhash"],
                }
            )
    return records, categories, source_annotation_files


def material_groups(records: list[dict[str, Any]], plastic_category_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncertain: list[dict[str, Any]] = []
    for row in records:
        if row["video_group"] is None:
            uncertain.append(row)
            grouped[f"uncertain:{row['source_image_path']}"].append(row)
        else:
            grouped[row["video_group"]].append(row)
    groups: list[dict[str, Any]] = []
    for group_id, rows in grouped.items():
        instances = sum(sum(annotation["category_id"] == plastic_category_id for annotation in row["annotations"]) for row in rows)
        positive_images = sum(any(annotation["category_id"] == plastic_category_id for annotation in row["annotations"]) for row in rows)
        for row in rows:
            row["assignment_group"] = group_id
        groups.append({"id": group_id, "images": len(rows), "positive_images": positive_images, "instances": instances, "rows": rows})
    return groups, uncertain


def coalesce_groups(
    groups: list[dict[str, Any]], links: list[list[str]], plastic_category_id: int,
) -> list[dict[str, Any]]:
    """Join only explicitly unresolved review components before splitting."""
    parent = {group["id"]: group["id"] for group in groups}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for component in links:
        existing = [group for group in component if group in parent]
        for group in existing[1:]:
            union(existing[0], group)
    combined: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        combined[find(group["id"])].extend(group["rows"])
    output = []
    for root, rows in combined.items():
        members = sorted({row["video_group"] or f"uncertain:{row['source_image_path']}" for row in rows})
        group_id = "component:" + "+".join(members)
        for row in rows:
            row["assignment_group"] = group_id
        instances = sum(sum(annotation["category_id"] == plastic_category_id for annotation in row["annotations"]) for row in rows)
        positive_images = sum(any(annotation["category_id"] == plastic_category_id for annotation in row["annotations"]) for row in rows)
        output.append({"id": group_id, "members": members, "images": len(rows), "positive_images": positive_images, "instances": instances, "rows": rows})
    return output


def clipped_bbox(annotation: dict[str, Any], width: int, height: int) -> tuple[list[float], dict[str, Any] | None, bool]:
    raw = annotation.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Annotation {annotation.get('id')} does not have a COCO xywh bbox")
    x, y, box_width, box_height = (float(value) for value in raw)
    x1, y1, x2, y2 = x, y, x + box_width, y + box_height
    clipped = [min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height)), min(max(x2, 0.0), float(width)), min(max(y2, 0.0), float(height))]
    derived = [clipped[0], clipped[1], clipped[2] - clipped[0], clipped[3] - clipped[1]]
    changed = any(abs(before - after) > 1e-9 for before, after in zip((x1, y1, x2, y2), clipped))
    zero_area = derived[2] <= 0.0 or derived[3] <= 0.0
    if not changed and not zero_area:
        return derived, None, False
    deltas = [abs(before - after) for before, after in zip((x1, y1, x2, y2), clipped)]
    threshold = max(2.0, 0.01 * max(width, height))
    log = {
        "source_annotation_id": annotation.get("id"),
        "source_bbox_xywh": [x, y, box_width, box_height],
        "source_corners_xyxy": [x1, y1, x2, y2],
        "clipped_corners_xyxy": clipped,
        "derived_bbox_xywh": derived,
        "edge_corrections_pixels": deltas,
        "total_correction_pixels": sum(deltas),
        "zero_area_after_clip": zero_area,
        "unexpectedly_large_correction": max(deltas) > threshold,
        "large_correction_threshold_pixels": threshold,
    }
    return derived, log, zero_area


def make_plastic_coco(
    records: list[dict[str, Any]], assignments: dict[str, str], plastic_category_id: int,
    class_mapping: dict[str, Any], output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    image_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    split_manifest: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    stats = {split: Counter() for split in SPLITS}
    image_counter = 0
    annotation_counter = {split: 0 for split in SPLITS}
    for row in sorted(records, key=lambda item: item["source_image_path"]):
        split = assignments[row["assignment_group"]]
        image_counter += 1
        image_id = image_counter
        plastic_annotations = [annotation for annotation in row["annotations"] if int(annotation["category_id"]) == plastic_category_id]
        positive = bool(plastic_annotations)
        stats[split]["images"] += 1
        stats[split]["positive_images"] += int(positive)
        stats[split]["negative_images"] += int(not positive)
        stats[split]["source_train_images"] += int(row["source_split"] == "train")
        stats[split]["source_val_images"] += int(row["source_split"] == "val")
        image_rows[split].append(
            {
                "id": image_id,
                "file_name": row["source_image_path"],
                "width": row["width"],
                "height": row["height"],
                "source_split": row["source_split"],
                "source_image_id": row["source_image_id"],
                "source_video_group": row["video_group"],
            }
        )
        split_manifest.append(
            {
                "derived_split": split,
                "video_group": row["video_group"],
                "group_rule": "vid_<video_id>_frame<frame_id>.jpg; keep all frames from a video together",
                "group_confidence": row["group_confidence"],
                "source_split": row["source_split"],
                "source_annotation_file": row["source_annotation_file"],
                "source_image_id": row["source_image_id"],
                "source_filename": row["source_filename"],
                "source_image_path": row["source_image_path"],
                "width": row["width"],
                "height": row["height"],
                "visible_plastic_debris_instances": len(plastic_annotations),
                "has_visible_plastic_debris": positive,
                "sha256": row["sha256"],
                "dhash": row["dhash"],
            }
        )
        for annotation in plastic_annotations:
            bbox, adjustment, zero_area = clipped_bbox(annotation, row["width"], row["height"])
            if adjustment is not None:
                adjustment.update({
                    "derived_split": split,
                    "source_annotation_file": row["source_annotation_file"],
                    "source_image_path": row["source_image_path"],
                    "source_image_id": row["source_image_id"],
                    "included_in_derived_coco": not zero_area,
                })
                adjustments.append(adjustment)
            if zero_area:
                stats[split]["zero_area_annotations_excluded_with_log"] += 1
                continue
            annotation_counter[split] += 1
            stats[split]["visible_plastic_debris_instances"] += 1
            annotation_rows[split].append(
                {
                    "id": annotation_counter[split],
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                    "source_annotation_id": annotation.get("id"),
                    "source_category_id": plastic_category_id,
                }
            )
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True)
    for split in SPLITS:
        payload = {
            "info": {
                "description": "AquaVision derived one-class visible plastic debris detection annotations",
                "source_dataset": "TrashCAN 1.0 material_version",
                "preparation_seed": MATERIAL_SEED,
                "bbox_policy": "COCO xywh -> corners -> clip to image bounds -> COCO xywh",
                "image_root": "AquaVision project root; file_name is project-relative",
            },
            "licenses": [],
            "images": image_rows[split],
            "annotations": annotation_rows[split],
            "categories": [{"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"}],
            "aqua_vision_class_mapping": class_mapping,
        }
        write_json(annotation_dir / f"{split}.json", payload)
    return {split: dict(stats[split]) for split in SPLITS}, split_manifest, adjustments


def validate_processed_coco(output: Path, source_rows: list[dict[str, Any]], expected_instances: int) -> dict[str, Any]:
    input_paths = {row["source_image_path"] for row in source_rows}
    seen_paths: set[str] = set()
    issues: list[str] = []
    summary: dict[str, Any] = {"splits": {}, "issues": issues}
    total_instances = 0
    for split in SPLITS:
        data = json.loads((output / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        images = {int(row["id"]): row for row in data["images"]}
        if data.get("categories") != [{"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"}]:
            issues.append(f"{split}: unexpected category mapping")
        for row in images.values():
            path = ROOT / row["file_name"]
            if not path.is_file():
                issues.append(f"{split}: missing source image {row['file_name']}")
            if row["file_name"] in seen_paths:
                issues.append(f"{split}: source image appears in more than one derived split: {row['file_name']}")
            seen_paths.add(row["file_name"])
        for annotation in data["annotations"]:
            image = images.get(int(annotation["image_id"]))
            bbox = annotation.get("bbox")
            if image is None or not isinstance(bbox, list) or len(bbox) != 4 or annotation.get("category_id") != 1:
                issues.append(f"{split}: malformed derived annotation {annotation.get('id')}")
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image["width"] + 1e-9 or y + height > image["height"] + 1e-9:
                issues.append(f"{split}: invalid clipped bbox annotation {annotation.get('id')}")
        total_instances += len(data["annotations"])
        summary["splits"][split] = {"images": len(images), "annotations": len(data["annotations"])}
    if seen_paths != input_paths:
        issues.append(f"Derived image coverage mismatch: expected {len(input_paths)}, found {len(seen_paths)}")
    if total_instances != expected_instances:
        issues.append(f"Derived plastic annotation mismatch: expected {expected_instances}, found {total_instances}")
    summary["valid"] = not issues
    summary["derived_instances"] = total_instances
    return summary


def uieb_pairs(image_manifest: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = {Path(path).name: row for path, row in image_manifest.items() if path.startswith("datasets/uieb/raw/")}
    reference = {Path(path).name: row for path, row in image_manifest.items() if path.startswith("datasets/uieb/reference/")}
    names = sorted(set(raw) & set(reference))
    if len(names) != 890 or set(raw) != set(reference):
        raise ValueError("UIEB pair validation no longer matches the saved validation result")
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_raw_hash: dict[str, list[str]] = defaultdict(list)
    by_raw_dhash: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_raw_hash[raw[name]["sha256"]].append(name)
        by_raw_dhash[raw[name]["dhash"]].append(name)
    exact_duplicate_groups = [sorted(items) for items in by_raw_hash.values() if len(items) > 1]
    dhash_similarity_groups = [sorted(items) for items in by_raw_dhash.values() if len(items) > 1]
    # Exact dHash is a review signal, not proof of duplicate content. Holding
    # these pairs together is a conservative leakage safeguard.
    for group in dhash_similarity_groups:
        for name in group[1:]:
            union(group[0], name)
    members: dict[str, list[str]] = defaultdict(list)
    for name in names:
        members[find(name)].append(name)
    groups = {name: f"pair_group:{min(values)}" for values in members.values() for name in values}
    rows = []
    for name in names:
        rows.append({
            "filename": name,
            "pair_group": groups[name],
            "group_rule": "raw/reference filename pair; exact raw dHash candidates held together as a precaution",
            "scene_relationship": "unverified beyond filename pair and recorded dHash review",
            "raw_path": raw[name]["path"],
            "reference_path": reference[name]["path"],
            "raw_sha256": raw[name]["sha256"],
            "reference_sha256": reference[name]["sha256"],
            "raw_dhash": raw[name]["dhash"],
            "reference_dhash": reference[name]["dhash"],
            "width": raw[name]["width"],
            "height": raw[name]["height"],
        })
    challenging = [row for path, row in image_manifest.items() if path.startswith("datasets/uieb/challenging/")]
    return rows, challenging, [{"signal": "exact_raw_sha256", "pairs": group, "confirmed_duplicate": True} for group in exact_duplicate_groups] + [{"signal": "exact_raw_dhash", "pairs": group, "confirmed_duplicate": False} for group in dhash_similarity_groups]


def prepare_uieb(image_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows, challenging, review = uieb_pairs(image_manifest)
    group_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_members[row["pair_group"]].append(row)
    groups = [{"id": group_id, "images": len(items), "positive_images": 0, "instances": 0} for group_id, items in group_members.items()]
    assignments, counts, targets = allocate_whole_groups(groups, UIEB_SEED)
    for row in rows:
        row["derived_split"] = assignments[row["pair_group"]]
    UIEB_OUTPUT.mkdir(parents=True)
    manifest_fields = ["derived_split", "pair_group", "group_rule", "scene_relationship", "filename", "raw_path", "reference_path", "width", "height", "raw_sha256", "reference_sha256", "raw_dhash", "reference_dhash"]
    write_csv(UIEB_OUTPUT / "pair_split_manifest.csv", manifest_fields, sorted(rows, key=lambda item: (item["derived_split"], item["filename"])))
    challenge_rows = [{"path": row["path"], "width": row["width"], "height": row["height"], "sha256": row["sha256"], "dhash": row["dhash"], "policy": "challenging set held outside train/validation/test"} for row in sorted(challenging, key=lambda item: item["path"])]
    write_csv(UIEB_OUTPUT / "challenging_manifest.csv", ["path", "width", "height", "sha256", "dhash", "policy"], challenge_rows)
    write_json(UIEB_OUTPUT / "similarity_review.json", {"records": review, "policy": "Perceptual hash similarity is not proof of duplication; exact-dHash candidates were held together conservatively."})
    validation = {
        "valid": len(rows) == 890 and len(challenge_rows) == 60 and all(row["raw_path"] and row["reference_path"] for row in rows),
        "pair_count": len(rows),
        "challenging_count": len(challenge_rows),
        "pair_groups": len(groups),
        "exact_content_duplicate_groups": sum(record["signal"] == "exact_raw_sha256" for record in review),
        "exact_dhash_review_groups": sum(record["signal"] == "exact_raw_dhash" for record in review),
        "split_counts": {split: counts[split]["images"] for split in SPLITS},
        "target_pair_counts": {split: targets[split]["images"] for split in SPLITS},
        "rules": {
            "pair_integrity": "Raw and reference images with the same filename are assigned to the same split.",
            "scene_grouping": "No verified higher-level scene metadata exists; all non-dHash groups are filename-pair groups and marked unverified.",
            "test_policy": "The derived test split is held out from training and model selection.",
        },
    }
    write_json(UIEB_OUTPUT / "validation.json", validation)
    return validation


def material_similarity_review(records: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    by_dhash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_dhash[row["dhash"]].append(row)
    cross_group = []
    for digest, rows in by_dhash.items():
        derived_splits = {assignments[row["assignment_group"]] for row in rows}
        groups = {row["video_group"] for row in rows}
        if len(derived_splits) > 1 and len(groups) > 1:
            cross_group.append({
                "dhash": digest,
                "derived_splits": sorted(derived_splits),
                "video_groups": sorted(group for group in groups if group),
                "source_image_paths": sorted(row["source_image_path"] for row in rows),
                "interpretation": "Perceptual-hash similarity only; this is a review candidate, not proof of duplication.",
            })
    previous = json.loads((VALIDATION / "overlap_and_leakage.json").read_text(encoding="utf-8"))
    original_candidates = previous["split_leakage"]["material_version"]["nearby_same_video_visual_candidates"]
    now_separated = 0
    for item in original_candidates:
        train_path = rel(TRASHCAN_MATERIAL / "train" / item["train"])
        val_path = rel(TRASHCAN_MATERIAL / "val" / item["val"])
        train_row = next((row for row in records if row["source_image_path"] == train_path), None)
        val_row = next((row for row in records if row["source_image_path"] == val_path), None)
        if train_row and val_row and assignments[train_row["assignment_group"]] != assignments[val_row["assignment_group"]]:
            now_separated += 1
    return {
        "cross_group_exact_dhash_candidates": cross_group,
        "prior_supplied_split_nearby_same_video_candidates": len(original_candidates),
        "prior_same_video_candidates_still_cross_split_after_grouping": now_separated,
        "rule": "All verified video groups are atomic; perceptual similarity is reported for review and never treated as proof of duplication.",
    }


def render_summary(summary: dict[str, Any]) -> str:
    material = summary["material"]
    uieb = summary["uieb"]
    lines = [
        "# AquaVision Training Data Preparation",
        "",
        f"Generated: {summary['generated_at_utc']}",
        f"Project root: `{summary['project_root']}`",
        "",
        "## Safeguards",
        "",
        "- No training was started and no website file was changed.",
        "- Source datasets and source annotations are unchanged; all derived files live below `datasets/processed/`.",
        "- Existing TrashCan supplied train/validation files remain in place and are referenced in the benchmark manifest.",
        "",
        "## Material TrashCan: visible plastic debris",
        "",
        "- Source: TrashCAN `material_version` only.",
        "- Mapping: `trash_plastic` (source category 14) → `visible_plastic_debris` (derived category 1). This deliberately does not imply bottles, bags, crates, polymer type, or any other subtype.",
        "- Images without `trash_plastic` annotations are retained as negatives.",
        "- Group rule: `vid_<video_id>_frame<frame_id>.jpg`; every verified video stays wholly within one derived split.",
        "",
        "| Split | Images | Positive images | Negative images | Plastic instances | Video groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        stats = material["split_stats"][split]
        lines.append(f"| {split} | {stats['images']} | {stats['positive_images']} | {stats['negative_images']} | {stats['visible_plastic_debris_instances']} | {material['groups_per_split'][split]} |")
    lines.extend([
        "",
        f"- Bounding boxes adjusted: {material['bbox_adjustments']}; zero-area after clipping: {material['zero_area_after_clip']}; unexpectedly large corrections: {material['large_corrections']}. Every adjustment is logged.",
        f"- Cross-group exact-dHash review candidates: {len(material['similarity_review']['cross_group_exact_dhash_candidates'])}. This is a review signal only, not evidence of duplicates.",
        f"- The prior supplied-split review found {material['similarity_review']['prior_supplied_split_nearby_same_video_candidates']} nearby same-video candidates; the group split leaves {material['similarity_review']['prior_same_video_candidates_still_cross_split_after_grouping']} across derived splits.",
        "",
        "## UIEB enhancement",
        "",
        f"- Paired raw/reference images: {uieb['pair_count']}; challenging held separate: {uieb['challenging_count']}.",
        f"- Derived pair split: train {uieb['split_counts']['train']}, validation {uieb['split_counts']['val']}, test {uieb['split_counts']['test']}.",
        f"- Exact raw-content duplicate groups: {uieb['exact_content_duplicate_groups']}; exact-dHash review groups held together as a precaution: {uieb['exact_dhash_review_groups']}.",
        "- The test split is a holdout: it must not be used for training or model selection.",
        "",
        "## Validation",
        "",
        f"- Derived COCO validation: {'passed' if material['derived_validation']['valid'] else 'failed'}; issues: {len(material['derived_validation']['issues'])}.",
        f"- UIEB manifest validation: {'passed' if uieb['valid'] else 'failed'}.",
        "",
        "## Files created",
        "",
        f"- `{summary['material_output']}/annotations/{{train,val,test}}.json`",
        f"- `{summary['material_output']}/{{split_manifest.csv,supplied_benchmark_manifest.csv,bbox_adjustments.csv,class_mapping.json,similarity_review.json,validation.json}}`",
        f"- `{summary['uieb_output']}/{{pair_split_manifest.csv,challenging_manifest.csv,similarity_review.json,validation.json}}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v1", help="Version suffix for new, non-overwriting processed outputs.")
    parser.add_argument(
        "--review-decisions",
        type=Path,
        help="Decision JSON containing conservative_group_links from a completed visual leakage review.",
    )
    args = parser.parse_args()
    global MATERIAL_OUTPUT, UIEB_OUTPUT, MATERIAL_SEED, UIEB_SEED
    MATERIAL_OUTPUT = PROCESSED / f"trashcan_material_plastic_{args.run_id}"
    UIEB_OUTPUT = PROCESSED / f"uieb_enhancement_{args.run_id}"
    MATERIAL_SEED = f"AquaVision-material-plastic-{args.run_id}-20260930"
    UIEB_SEED = f"AquaVision-uieb-{args.run_id}-20260930"
    if not (ROOT / "website").is_dir() or not (ROOT / "serve.py").is_file():
        print("Run from the intended AquaVision AI project containing website/ and serve.py.", file=sys.stderr)
        return 2
    if MATERIAL_OUTPUT.exists() or UIEB_OUTPUT.exists():
        print("Refusing to overwrite an existing preparation output. Choose a new versioned output name for another run.", file=sys.stderr)
        return 3
    image_manifest = load_image_manifest()
    print("Preparing one-class TrashCan material annotations...", flush=True)
    records, categories, source_annotation_files = load_material_sources(image_manifest)
    source_plastic_ids = [category_id for category_id, name in categories.items() if name == "trash_plastic"]
    if source_plastic_ids != [14]:
        raise ValueError(f"Expected exactly source category 14 named trash_plastic; found {source_plastic_ids}")
    plastic_category_id = source_plastic_ids[0]
    groups, uncertain = material_groups(records, plastic_category_id)
    review_links: list[list[str]] = []
    review_decisions_path = None
    if args.review_decisions:
        review_decisions_path = args.review_decisions.resolve()
        decisions = json.loads(review_decisions_path.read_text(encoding="utf-8"))
        review_links = [list(component) for component in decisions.get("conservative_group_links", []) if len(component) > 1]
        groups = coalesce_groups(groups, review_links, plastic_category_id)
    assignments, _, targets = allocate_whole_groups(groups, MATERIAL_SEED)
    MATERIAL_OUTPUT.mkdir(parents=True)
    class_mapping = {
        "source_dataset": "TrashCAN 1.0 material_version",
        "source_category": {"id": plastic_category_id, "name": "trash_plastic"},
        "derived_category": {"id": 1, "name": "visible_plastic_debris", "supercategory": "trash"},
        "scope": "Visible plastic debris only.",
        "non_inferences": ["bottle", "bag", "crate", "polymer type", "other object subtype"],
        "negative_example_policy": "Retain every source image without a trash_plastic annotation as a negative image.",
    }
    write_json(MATERIAL_OUTPUT / "class_mapping.json", class_mapping)
    split_stats, split_manifest, adjustments = make_plastic_coco(records, assignments, plastic_category_id, class_mapping, MATERIAL_OUTPUT)
    write_csv(MATERIAL_OUTPUT / "split_manifest.csv", [
        "derived_split", "video_group", "group_rule", "group_confidence", "source_split", "source_annotation_file", "source_image_id", "source_filename", "source_image_path", "width", "height", "visible_plastic_debris_instances", "has_visible_plastic_debris", "sha256", "dhash",
    ], sorted(split_manifest, key=lambda row: (row["derived_split"], row["source_image_path"])))
    write_csv(MATERIAL_OUTPUT / "supplied_benchmark_manifest.csv", [
        "source_split", "source_annotation_file", "source_image_id", "source_filename", "source_image_path", "video_group", "derived_split",
    ], sorted([{key: row[key] for key in ("source_split", "source_annotation_file", "source_image_id", "source_filename", "source_image_path", "video_group")} | {"derived_split": row["derived_split"]} for row in split_manifest], key=lambda row: (row["source_split"], row["source_filename"])))
    write_csv(MATERIAL_OUTPUT / "bbox_adjustments.csv", [
        "derived_split", "source_annotation_file", "source_annotation_id", "source_image_path", "source_image_id", "source_bbox_xywh", "source_corners_xyxy", "clipped_corners_xyxy", "derived_bbox_xywh", "edge_corrections_pixels", "total_correction_pixels", "zero_area_after_clip", "unexpectedly_large_correction", "large_correction_threshold_pixels", "included_in_derived_coco",
    ], adjustments)
    expected_instances = sum(stats["visible_plastic_debris_instances"] for stats in split_stats.values())
    derived_validation = validate_processed_coco(MATERIAL_OUTPUT, records, expected_instances)
    write_json(MATERIAL_OUTPUT / "validation.json", derived_validation)
    similarity = material_similarity_review(records, assignments)
    write_json(MATERIAL_OUTPUT / "similarity_review.json", similarity)
    groups_per_split = Counter(assignments.values())
    material_summary = {
        "source_annotation_files": source_annotation_files,
        "seed": MATERIAL_SEED,
        "group_rule": "vid_<video_id>_frame<frame_id>.jpg; video ID is the atomic source group",
        "conservative_group_links": review_links,
        "review_decisions": rel(review_decisions_path) if review_decisions_path else None,
        "uncertain_group_assignments": [row["source_image_path"] for row in uncertain],
        "target_image_counts": {split: targets[split]["images"] for split in SPLITS},
        "split_stats": split_stats,
        "groups_per_split": dict(groups_per_split),
        "bbox_adjustments": len(adjustments),
        "zero_area_after_clip": sum(row["zero_area_after_clip"] for row in adjustments),
        "large_corrections": sum(row["unexpectedly_large_correction"] for row in adjustments),
        "derived_validation": derived_validation,
        "similarity_review": similarity,
    }
    write_json(MATERIAL_OUTPUT / "preparation_summary.json", material_summary)
    print("Preparing UIEB paired split manifests...", flush=True)
    uieb_summary = prepare_uieb(image_manifest)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": ROOT.as_posix(),
        "material": material_summary,
        "uieb": uieb_summary,
        "material_output": rel(MATERIAL_OUTPUT),
        "uieb_output": rel(UIEB_OUTPUT),
    }
    (PROCESSED / f"PREPARATION_SUMMARY_{args.run_id}.md").write_text(render_summary(summary), encoding="utf-8")
    write_json(PROCESSED / f"preparation_summary_{args.run_id}.json", summary)
    print("Preparation complete without modifying source datasets or website files.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
