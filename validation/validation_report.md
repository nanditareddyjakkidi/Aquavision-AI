# AquaVision Dataset Validation Report

Generated: 2026-09-30T09:34:50.004299+00:00
Project root: `/Users/nanditareddy/Documents/Codex/2026-09-30/referenced-chatgpt-conversation-this-is-an`

## Scope and safeguards

- Read-only validation of `datasets/uieb` and `datasets/trashcan`.
- No source archive files, dataset files, training configuration, splits, or website files were modified.
- The project root contains no detected AquaVision website files or common website configuration markers.

## UIEB enhancement data

- Images decoded: raw 890, reference 890, challenging 60; decode failures: 0.
- Raw/reference filename matches: 890; usable same-dimension enhancement pairs: **890**.
- Missing references: 0; extra references: 0; ambiguous names: 0; dimension mismatches: 0.
- `challenging/` remains separate and is not counted as paired training data.

## TrashCan inventory and annotation formats

- `material_version` and `instance_version` each contain supplied train/validation image directories and two COCO-style JSON annotation files. They are alternative label taxonomies over overlapping imagery, not independent datasets.
- `original_data` contains one Supervisely-style bitmap JSON annotation per image. It is not a COCO detection annotation set and has not been converted.

### material_version

| Split | Images | Annotations | Categories | Validation issues |
| --- | ---: | ---: | ---: | ---: |
| train | 6008 | 9741 | 16 | 0 |
| val | 1204 | 2595 | 16 | 0 |

Categories: 1: rov, 2: plant, 3: animal_fish, 4: animal_starfish, 5: animal_shells, 6: animal_crab, 7: animal_eel, 8: animal_etc, 9: trash_etc, 10: trash_fabric, 11: trash_fishing_gear, 12: trash_metal, 13: trash_paper, 14: trash_plastic, 15: trash_rubber, 16: trash_wood.
- Minor boundary overflows: 3346 boxes; maximum overflow 0.998 pixels. No box exceeds the image boundary by more than one pixel.

- Train/val filename overlap: 0; exact content-hash overlap: 0; exact dHash overlap: 85.
- Videos spanning train and validation: 127. Nearby same-video visual candidates: 892.

### instance_version

| Split | Images | Annotations | Categories | Validation issues |
| --- | ---: | ---: | ---: | ---: |
| train | 6065 | 9540 | 22 | 0 |
| val | 1147 | 2588 | 22 | 0 |

Categories: 1: rov, 2: plant, 3: animal_fish, 4: animal_starfish, 5: animal_shells, 6: animal_crab, 7: animal_eel, 8: animal_etc, 9: trash_clothing, 10: trash_pipe, 11: trash_bottle, 12: trash_bag, 13: trash_snack_wrapper, 14: trash_can, 15: trash_cup, 16: trash_container, 17: trash_unknown_instance, 18: trash_branch, 19: trash_wreckage, 20: trash_tarp, 21: trash_rope, 22: trash_net.
- Minor boundary overflows: 3286 boxes; maximum overflow 0.998 pixels. No box exceeds the image boundary by more than one pixel.

- Train/val filename overlap: 0; exact content-hash overlap: 0; exact dHash overlap: 80.
- Videos spanning train and validation: 113. Nearby same-video visual candidates: 839.

### original_data

- Images: 7212; annotation files: 7212; decoded annotation JSON errors: 0; missing image references: 0; dimension mismatches: 0.
- Object geometries: bitmap (12480).
- Object tags: animal (2304), crushed/broken (715), decay (637), instance (4549), material (4769), overgrown (1048), plant (508).

## RF-DETR readiness

- Both COCO-style material and instance JSON files have valid file references, category IDs, metadata dimensions, and positive boxes. They are usable COCO detection candidates for RF-DETR, but the listed one-pixel boundary overflows should be clipped or explicitly accepted by the loader before training. Select exactly one taxonomy for a run.
- `original_data` is not directly suitable: it stores Supervisely-style bitmap geometry rather than COCO image/category/bounding-box records. No conversion was performed.
- Preserve the provided train/validation splits. The same video can appear in both splits, so the leakage findings below should be reviewed before evaluating generalization.

## Duplicate and leakage findings

- Exact duplicate image hash groups across all copied image files: 7212 (21636 physical entries).
- Groups spanning TrashCan versions: 7212; these must not be combined as independent images.
- material_version: 127 videos span train/val. Examples: video 000002 (train 2, val 9, closest gap 1); video 000003 (train 8, val 14, closest gap 1); video 000020 (train 1, val 2, closest gap 1); video 000021 (train 15, val 10, closest gap 1); video 000022 (train 3, val 10, closest gap 1).
- instance_version: 113 videos span train/val. Examples: video 000002 (train 1, val 10, closest gap 1); video 000003 (train 14, val 8, closest gap 1); video 000004 (train 6, val 3, closest gap 1); video 000005 (train 5, val 1, closest gap 1); video 000021 (train 11, val 14, closest gap 1).

## Blockers and next decisions

- Do not combine `material_version`, `instance_version`, or `original_data` as separate images: they contain overlapping source frames.
- Use UIEB raw/reference pairs only for enhancement. Keep UIEB challenging images out of paired enhancement training unless a separate policy is chosen.
- For TrashCan detection, choose one taxonomy (material or instance) and review same-video train/validation leakage before relying on validation metrics.
- No annotation conversion or new split was created by this validation.

## Generated files

- `validation/dataset_file_manifest.sha256` — SHA-256 manifest for every copied dataset file (30706 files).
- `validation/image_manifest.jsonl` — decoded image paths, dimensions, exact hashes, and dHashes.
- `validation/overlap_and_leakage.json` — exact duplicate groups and split-leakage evidence.
- `validation/validation_summary.json` — machine-readable complete validation results.
