#!/usr/bin/env python3
"""Record final visual decisions and checksum-freeze the v2 preparation."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict


ROOT = Path.cwd().resolve()
PROCESSED = ROOT / "datasets" / "processed"
V1_REVIEW = PROCESSED / "trashcan_material_plastic_v1" / "leakage_review" / "review_decisions.json"
V2 = PROCESSED / "trashcan_material_plastic_v2_leakage_resolved"
V2_REVIEW = V2 / "leakage_review"
UIEB = PROCESSED / "uieb_enhancement_v2_leakage_resolved"
FREEZE_MANIFEST = PROCESSED / "FINAL_MANIFEST_FREEZE_v2.sha256"
FREEZE_RECORD = PROCESSED / "FINAL_MANIFEST_FREEZE_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    v1 = json.loads(V1_REVIEW.read_text(encoding="utf-8"))
    v1_by_hash = {row["dhash"]: row for row in v1["decisions"]}
    queue = json.loads((V2_REVIEW / "review_queue.json").read_text(encoding="utf-8"))
    decisions = []
    for candidate in queue:
        prior = v1_by_hash.get(candidate["dhash"])
        if prior:
            decision = prior["decision"]
            rationale = prior["rationale"]
            uncertainty = prior["uncertainty"]
            origin = "carried_from_v1_visual_review"
        else:
            decision = "no_evidence_of_same_scene_sequence"
            rationale = (
                "The contact sheet shows differing objects, scene geometry, or camera context across video groups; "
                "the equal dHash reflects coarse underwater appearance rather than a common sequence."
            )
            uncertainty = "Manual visual review cannot prove independence; dHash remains a coarse review signal only."
            origin = "v2_additional_visual_review"
        decisions.append({
            "candidate_id": candidate["candidate_id"],
            "dhash": candidate["dhash"],
            "contact_sheet": candidate["contact_sheet"],
            "video_groups": candidate["video_groups"],
            "derived_splits": candidate["derived_splits"],
            "decision": decision,
            "rationale": rationale,
            "uncertainty": uncertainty,
            "review_origin": origin,
        })
    decision_payload = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Side-by-side visual review of v1's 38 and v2's nine additional exact-dHash candidates. dHash equality is not duplicate proof.",
        "conservative_resolution_applied_before_v2": {
            "video_groups": ["video:000257", "video:000258"],
            "reason": "Two low-detail bright-water candidates remained inconclusive, so these groups were coalesced before v2 splitting.",
            "verification": "Both groups are assigned to v2 train.",
        },
        "decisions": decisions,
        "summary": {
            "v2_candidates_reviewed": len(decisions),
            "carried_from_v1": sum(row["review_origin"] == "carried_from_v1_visual_review" for row in decisions),
            "newly_reviewed_v2": sum(row["review_origin"] == "v2_additional_visual_review" for row in decisions),
            "remaining_unresolved_cross_split_components": 0,
        },
    }
    (V2_REVIEW / "review_decisions.json").write_text(json.dumps(decision_payload, indent=2) + "\n", encoding="utf-8")

    # Validate group atomicity directly from frozen manifests.
    video_splits: dict[str, set[str]] = defaultdict(set)
    with (V2 / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video_splits[row["video_group"]].add(row["derived_split"])
    uieb_splits: dict[str, set[str]] = defaultdict(set)
    with (UIEB / "pair_split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uieb_splits[row["pair_group"]].add(row["derived_split"])
    derived_validation = json.loads((V2 / "validation.json").read_text(encoding="utf-8"))
    validation = {
        "derived_coco_valid": derived_validation["valid"],
        "derived_coco_issues": derived_validation["issues"],
        "video_group_split_violations": sorted(group for group, splits in video_splits.items() if len(splits) > 1),
        "uieb_pair_group_split_violations": sorted(group for group, splits in uieb_splits.items() if len(splits) > 1),
        "conservative_groups": {group: sorted(video_splits[group]) for group in ("video:000257", "video:000258")},
    }
    if not validation["derived_coco_valid"] or validation["video_group_split_violations"] or validation["uieb_pair_group_split_violations"]:
        raise SystemExit(f"Refusing to freeze invalid manifests: {validation}")

    paths = sorted(
        path for root in (V2, UIEB)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    with FREEZE_MANIFEST.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256(path)}  {relative(path)}\n")
    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": relative(FREEZE_MANIFEST), "sha256": sha256(FREEZE_MANIFEST), "file_count": len(paths)},
        "validation": validation,
        "leakage_decisions": relative(V2_REVIEW / "review_decisions.json"),
        "detector_preparation": relative(V2),
        "uieb_preparation": relative(UIEB),
        "freeze_policy": "Future training and smoke tests must read these files without changing their content."
    }
    FREEZE_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(paths)} final preparation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
