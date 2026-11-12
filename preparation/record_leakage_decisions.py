#!/usr/bin/env python3
"""Record the completed visual decisions for the generated leakage sheets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd().resolve()
REVIEW = ROOT / "datasets" / "processed" / "trashcan_material_plastic_v1" / "leakage_review"
# Candidate 25 and 28 both connect the same two low-detail bright-water video
# groups. Visual review cannot rule out a related capture session, so keep that
# component together in the regenerated split.
UNRESOLVED_GROUPS = {25, 28}


def main() -> int:
    queue = json.loads((REVIEW / "review_queue.json").read_text(encoding="utf-8"))
    decisions = []
    linked_groups: set[str] = set()
    for candidate in queue:
        candidate_id = candidate["candidate_id"]
        if candidate_id in UNRESOLVED_GROUPS:
            decision = "unresolved_group_conservatively"
            rationale = (
                "The bright, low-detail frames do not contain enough distinctive scene content to rule out a related capture session. "
                "The involved video groups are therefore kept together before any training."
            )
            uncertainty = "Visual inspection is inconclusive; equal dHash alone does not establish duplication."
            linked_groups.update(candidate["video_groups"])
        else:
            decision = "no_evidence_of_same_scene_sequence"
            rationale = (
                "Across video groups, the sheet shows differing objects, camera overlays, seafloor geometry, or illumination; "
                "the equal dHash is attributable to coarse low-detail underwater appearance."
            )
            uncertainty = "Manual visual review cannot prove independence, but no common scene/object sequence is visible."
        decisions.append({
            "candidate_id": candidate_id,
            "contact_sheet": candidate["contact_sheet"],
            "dhash": candidate["dhash"],
            "video_groups": candidate["video_groups"],
            "derived_splits_before_resolution": candidate["derived_splits"],
            "decision": decision,
            "rationale": rationale,
            "uncertainty": uncertainty,
        })
    payload = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Side-by-side manual visual review of all 38 exact-dHash contact sheets. dHash equality was treated as a review signal, not duplicate proof.",
        "decisions": decisions,
        "conservative_group_links": [sorted(linked_groups)],
        "summary": {
            "candidates_reviewed": len(decisions),
            "no_evidence_of_same_scene_sequence": len(decisions) - len(UNRESOLVED_GROUPS),
            "unresolved_candidates": sorted(UNRESOLVED_GROUPS),
            "action": "Regenerate group-safe splits with video:000257 and video:000258 held together.",
        },
    }
    (REVIEW / "review_decisions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Recorded 38 decisions and one conservative video-group component.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
