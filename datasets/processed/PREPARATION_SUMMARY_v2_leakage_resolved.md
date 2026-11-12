# AquaVision Training Data Preparation

Generated: 2026-09-30T14:32:49.975253+00:00
Project root: `/Users/nanditareddy/AquaVision AI`

## Safeguards

- No training was started and no website file was changed.
- Source datasets and source annotations are unchanged; all derived files live below `datasets/processed/`.
- Existing TrashCan supplied train/validation files remain in place and are referenced in the benchmark manifest.

## Material TrashCan: visible plastic debris

- Source: TrashCAN `material_version` only.
- Mapping: `trash_plastic` (source category 14) → `visible_plastic_debris` (derived category 1). This deliberately does not imply bottles, bags, crates, polymer type, or any other subtype.
- Images without `trash_plastic` annotations are retained as negatives.
- Group rule: `vid_<video_id>_frame<frame_id>.jpg`; every verified video stays wholly within one derived split.

| Split | Images | Positive images | Negative images | Plastic instances | Video groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 5048 | 1242 | 3806 | 1299 | 156 |
| val | 1082 | 249 | 833 | 289 | 79 |
| test | 1082 | 262 | 820 | 276 | 76 |

- Bounding boxes adjusted: 248; zero-area after clipping: 0; unexpectedly large corrections: 0. Every adjustment is logged.
- Cross-group exact-dHash review candidates: 40. This is a review signal only, not evidence of duplicates.
- The prior supplied-split review found 892 nearby same-video candidates; the group split leaves 0 across derived splits.

## UIEB enhancement

- Paired raw/reference images: 890; challenging held separate: 60.
- Derived pair split: train 623, validation 134, test 133.
- Exact raw-content duplicate groups: 0; exact-dHash review groups held together as a precaution: 8.
- The test split is a holdout: it must not be used for training or model selection.

## Validation

- Derived COCO validation: passed; issues: 0.
- UIEB manifest validation: passed.

## Files created

- `datasets/processed/trashcan_material_plastic_v2_leakage_resolved/annotations/{train,val,test}.json`
- `datasets/processed/trashcan_material_plastic_v2_leakage_resolved/{split_manifest.csv,supplied_benchmark_manifest.csv,bbox_adjustments.csv,class_mapping.json,similarity_review.json,validation.json}`
- `datasets/processed/uieb_enhancement_v2_leakage_resolved/{pair_split_manifest.csv,challenging_manifest.csv,similarity_review.json,validation.json}`
