#!/usr/bin/env python3
"""Create labeled side-by-side visual review sheets for dHash candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path.cwd().resolve()
PREP = ROOT / "datasets" / "processed" / "trashcan_material_plastic_v1"
REVIEW = PREP / "leakage_review"
SHEETS = REVIEW / "contact_sheets"
THUMBNAIL = (180, 128)
COLUMNS = 4
COLORS = {"train": "#2155A3", "val": "#17834C", "test": "#B64A00"}


def text(draw: ImageDraw.ImageDraw, point: tuple[int, int], value: str, fill: str = "#FFFFFF") -> None:
    draw.text(point, value, font=ImageFont.load_default(), fill=fill)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", default="trashcan_material_plastic_v1")
    args = parser.parse_args()
    global PREP, REVIEW, SHEETS
    PREP = ROOT / "datasets" / "processed" / args.preparation_dir
    REVIEW = PREP / "leakage_review"
    SHEETS = REVIEW / "contact_sheets"
    if SHEETS.exists():
        raise SystemExit(f"Refusing to overwrite existing review sheets: {SHEETS}")
    review = json.loads((PREP / "similarity_review.json").read_text(encoding="utf-8"))
    candidates = review["cross_group_exact_dhash_candidates"]
    manifest = {row["source_image_path"]: row for row in json.loads("[]")}
    # Split assignments are authoritative for label colors and captions.
    import csv
    with (PREP / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = {row["source_image_path"]: row for row in csv.DictReader(handle)}
    SHEETS.mkdir(parents=True)
    queue = []
    for index, candidate in enumerate(candidates, start=1):
        rows = []
        for image_path in candidate["source_image_paths"]:
            item = manifest[image_path]
            rows.append({
                "source_image_path": image_path,
                "derived_split": item["derived_split"],
                "video_group": item["video_group"],
                "filename": item["source_filename"],
            })
        rows.sort(key=lambda row: (row["derived_split"], row["video_group"], row["filename"]))
        width = COLUMNS * THUMBNAIL[0]
        header_height = 54
        label_height = 30
        cell_height = THUMBNAIL[1] + label_height
        height = header_height + math.ceil(len(rows) / COLUMNS) * cell_height
        sheet = Image.new("RGB", (width, height), "#101820")
        draw = ImageDraw.Draw(sheet)
        text(draw, (8, 6), f"Candidate {index:02d} | dHash {candidate['dhash']}")
        text(draw, (8, 24), "Color: train=blue, validation=green, test=orange. Similarity is not proof of duplication.", "#D8E5F0")
        for item_index, row in enumerate(rows):
            col, line = item_index % COLUMNS, item_index // COLUMNS
            x, y = col * THUMBNAIL[0], header_height + line * cell_height
            path = ROOT / row["source_image_path"]
            with Image.open(path) as source:
                source.load()
                tile = ImageOps.contain(source.convert("RGB"), THUMBNAIL, Image.Resampling.LANCZOS)
            framed = Image.new("RGB", THUMBNAIL, "#000000")
            framed.paste(tile, ((THUMBNAIL[0] - tile.width) // 2, (THUMBNAIL[1] - tile.height) // 2))
            border = COLORS[row["derived_split"]]
            framed = ImageOps.expand(framed, border=3, fill=border)
            sheet.paste(framed.crop((0, 0, THUMBNAIL[0], THUMBNAIL[1])), (x, y))
            text(draw, (x + 4, y + THUMBNAIL[1] + 2), f"{row['derived_split']} | {row['video_group'].replace('video:', 'v')}", border)
            text(draw, (x + 4, y + THUMBNAIL[1] + 14), row["filename"].replace("vid_", "v").replace("_frame", " f"), "#D8E5F0")
        filename = f"candidate_{index:02d}_{candidate['dhash']}.jpg"
        sheet.save(SHEETS / filename, quality=92)
        queue.append({
            "candidate_id": index,
            "dhash": candidate["dhash"],
            "contact_sheet": (SHEETS / filename).relative_to(ROOT).as_posix(),
            "derived_splits": candidate["derived_splits"],
            "video_groups": candidate["video_groups"],
            "source_image_paths": candidate["source_image_paths"],
            "initial_status": "pending_visual_review",
            "decision": None,
            "uncertainty": "Equal dHash is only a coarse visual similarity signal.",
        })
    (REVIEW / "review_queue.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    (REVIEW / "README.md").write_text(
        "# Leakage contact-sheet review\n\n"
        "Each sheet shows one exact-dHash candidate side-by-side. Border colors indicate the derived split. "
        "Review whether frames indicate the same scene/object sequence or merely share coarse appearance. "
        "Do not treat dHash equality as duplicate proof. Decisions are saved separately in `review_decisions.json`.\n",
        encoding="utf-8",
    )
    print(f"Created {len(queue)} contact sheets in {SHEETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
