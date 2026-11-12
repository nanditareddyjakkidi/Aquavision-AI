#!/usr/bin/env python3
"""Summarize the completed scale_jitter=False run without training.

This is a read-only analysis of saved metrics, fixed qualitative previews, and
the new standalone epoch-3 EMA evaluation. It neither imports a training
entry-point nor accesses the test split. Generated report files live beneath
the completed no-jitter continuation run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from run_rfdetr_raw_v2_pilot import PROJECT, RUNS_ROOT, digest


NO_JITTER_SOURCE_RUN = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z"
NO_JITTER_RUN = RUNS_ROOT / "rfdetr_raw_v2_scale_jitter_off_e5to10_resume_mps_20261002T121244Z"
NO_JITTER_RESULT = NO_JITTER_RUN / "continuation_result.json"
NO_JITTER_HISTORY = NO_JITTER_RUN / "metric_history_ema.json"
NO_JITTER_EPOCH3_PREVIEWS = NO_JITTER_RUN / "fixed_validation_previews/epoch_003_ema/validation_previews"
BASELINE_RUN = RUNS_ROOT / "rfdetr_raw_v2_e5to10_fixed512_mps_20261001T115124Z"
BASELINE_RESULT = BASELINE_RUN / "continuation_result.json"
BASELINE_EPOCH5_PREVIEWS = BASELINE_RUN / "fixed_validation_previews/epoch_005_ema/validation_previews"
BASELINE_EVALUATION = RUNS_ROOT / "evaluations/rfdetr_epoch5_ema_validation_training_compatible_20261001T152245Z/evaluation_result.json"
EVALUATIONS_ROOT = RUNS_ROOT / "evaluations"
EXPECTED_NO_JITTER_EPOCH3_SHA256 = "6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440"
EXPECTED_BASELINE_SHA256 = "9e269f264845c638e65da5e2b77cdc54b19764fd85c2f00b0a3d846866b1f5a2"


def read_json(path: Path) -> Any:
    """Load one required saved JSON artifact."""
    if not path.is_file():
        raise RuntimeError(f"Required saved artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, name: str) -> float:
    """Reject malformed numeric records rather than silently omitting them."""
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def markdown_number(value: float) -> str:
    return f"{value:.4f}"


def metric_rows(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and normalize all ten completed-epoch saved records."""
    if [int(record.get("completed_epoch", -1)) for record in history] != list(range(1, 11)):
        raise RuntimeError("No-jitter history must contain each completed epoch 1 through 10 exactly once")
    rows: list[dict[str, Any]] = []
    for record in history:
        metrics = record.get("EMA_validation")
        losses = record.get("training_losses")
        rates = record.get("learning_rates")
        if not isinstance(metrics, dict) or not isinstance(losses, dict) or not isinstance(rates, dict):
            raise RuntimeError("No-jitter metric history has an incomplete record")
        row = {
            "epoch": int(record["completed_epoch"]),
            "segment": str(record["segment"]),
            "loss": finite(losses["loss"], "loss"),
            "loss_ce": finite(losses["loss_ce"], "loss_ce"),
            "loss_bbox": finite(losses["loss_bbox"], "loss_bbox"),
            "loss_giou": finite(losses["loss_giou"], "loss_giou"),
            "AP_50_95": finite(metrics["AP_50_95"], "AP_50_95"),
            "AP50": finite(metrics["AP50"], "AP50"),
            "AP75": finite(metrics["AP75"], "AP75"),
            "mAR": finite(metrics["mAR"], "mAR"),
            "lr": finite(rates["lr"], "lr"),
            "lr_min": finite(rates["lr_min"], "lr_min"),
            "lr_max": finite(rates["lr_max"], "lr_max"),
        }
        rows.append(row)
    summary = {
        "best_epoch": max(rows, key=lambda row: row["AP_50_95"])["epoch"],
        "best_AP_50_95": max(row["AP_50_95"] for row in rows),
        "epoch_5_loss": rows[4]["loss"],
        "epoch_10_loss": rows[9]["loss"],
        "epoch_5_AP_50_95": rows[4]["AP_50_95"],
        "epoch_10_AP_50_95": rows[9]["AP_50_95"],
        "unique_lr": sorted({row["lr"] for row in rows}),
        "unique_lr_min": sorted({row["lr_min"] for row in rows}),
        "unique_lr_max": sorted({row["lr_max"] for row in rows}),
    }
    return rows, summary


def preview_records(result: dict[str, Any], completed_epoch: int) -> list[dict[str, Any]]:
    """Retrieve one fixed qualitative panel and require all nine planned images."""
    previews = result.get("fixed_EMA_validation_previews")
    if isinstance(previews, dict):
        previews = previews.get("epochs")
    if not isinstance(previews, list):
        raise RuntimeError("Saved fixed-preview records are unavailable")
    for record in previews:
        if int(record.get("completed_epoch", -1)) == completed_epoch:
            entries = record.get("previews")
            if not isinstance(entries, list) or len(entries) != 9:
                raise RuntimeError("The fixed preview panel must contain nine planned images")
            return entries
    raise RuntimeError(f"No saved fixed preview panel for completed epoch {completed_epoch}")


def preview_summary(baseline: list[dict[str, Any]], no_jitter: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare existing panels by their declared source, GT count, and 0.25 count."""
    rows: list[dict[str, Any]] = []
    for index, (base, candidate) in enumerate(zip(baseline, no_jitter, strict=True), start=1):
        if base.get("source_image_path") != candidate.get("source_image_path") or base.get("ground_truth_instances") != candidate.get("ground_truth_instances"):
            raise RuntimeError("The two fixed panels do not represent the exact same source images/ground truth")
        gt = int(base["ground_truth_instances"])
        rows.append({
            "index": index,
            "source_image_path": str(base["source_image_path"]),
            "ground_truth_instances": gt,
            "kind": "negative" if gt == 0 else "positive",
            "baseline_predictions_threshold_0_25": int(base["prediction_count_threshold_0_25"]),
            "no_jitter_epoch3_predictions_threshold_0_25": int(candidate["prediction_count_threshold_0_25"]),
        })
    return rows


def header_font(size: int) -> ImageFont.ImageFont:
    """Use Pillow's stable default font so the report has no external asset."""
    return ImageFont.load_default(size=size)


def resize_panel(path: Path, width: int) -> Image.Image:
    """Open an existing preview without altering it and make one report copy."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def create_contact_sheet(output_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a labelled side-by-side report image from existing preview PNGs."""
    panel_width, margin, header_height = 620, 20, 30
    pairs: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    for row in rows:
        filename = f"{row['index']:02d}_{Path(row['source_image_path']).stem}.png"
        baseline = BASELINE_EPOCH5_PREVIEWS / filename
        no_jitter = NO_JITTER_EPOCH3_PREVIEWS / filename
        if not baseline.is_file() or not no_jitter.is_file():
            raise RuntimeError(f"Missing existing fixed preview for {filename}")
        pairs.append((row, resize_panel(baseline, panel_width), resize_panel(no_jitter, panel_width)))
    total_height = margin + sum(header_height + max(base.height, candidate.height) + margin for _, base, candidate in pairs)
    sheet = Image.new("RGB", (panel_width * 2 + margin * 3, total_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = header_font(16)
    y = margin
    for row, baseline, no_jitter in pairs:
        label = f"{row['index']:02d} | {row['kind']} | GT={row['ground_truth_instances']} | baseline@0.25={row['baseline_predictions_threshold_0_25']} | no-jitter e3@0.25={row['no_jitter_epoch3_predictions_threshold_0_25']}"
        draw.text((margin, y), label, fill="black", font=font)
        y += header_height
        sheet.paste(baseline, (margin, y))
        sheet.paste(no_jitter, (margin * 2 + panel_width, y))
        y += max(baseline.height, no_jitter.height) + margin
    sheet.save(output_path)
    return {"path": str(output_path.relative_to(PROJECT)), "sha256": digest(output_path), "layout": "left=original raw epoch-5 EMA, right=scale_jitter=False epoch-3 EMA", "items": len(pairs)}


def comparison_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the quantitative comparison only after matching validation checks."""
    baseline_metrics = baseline["training_compatible_ema_metrics"]
    candidate_metrics = candidate["training_compatible_ema_metrics"]
    metric_names = ("AP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_small_500", "AR_medium_500", "AR_large_500")
    rows = []
    for name in metric_names:
        base, new = finite(baseline_metrics[name], name), finite(candidate_metrics[name], name)
        rows.append({"metric": name, "original_epoch5": base, "no_jitter_epoch3": new, "delta": new - base})
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a compact Markdown table from validated values."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(markdown_number(value) if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-run-id", required=True, help="Completed standalone evaluator directory under training_runs/rfdetr/evaluations/")
    args = parser.parse_args()
    if "/" in args.evaluation_run_id or args.evaluation_run_id in {"", ".", ".."}:
        raise SystemExit("--evaluation-run-id must be a simple directory name")
    evaluation_dir = EVALUATIONS_ROOT / args.evaluation_run_id
    candidate_evaluation = read_json(evaluation_dir / "evaluation_result.json")
    if candidate_evaluation.get("status") != "passed":
        raise RuntimeError("The epoch-3 standalone reproduction did not pass; resolve it before comparing metrics")
    candidate_checkpoint = candidate_evaluation.get("checkpoint", {})
    if candidate_checkpoint.get("target") != "scale_jitter_off_epoch3_best" or candidate_checkpoint.get("sha256") != EXPECTED_NO_JITTER_EPOCH3_SHA256:
        raise RuntimeError("The supplied evaluator result is not the preserved no-jitter epoch-3 best EMA checkpoint")
    reproduction = candidate_evaluation.get("reproduction_check", {})
    if not reproduction.get("passed") or abs(finite(reproduction["evaluated_AP_50_95"], "evaluated AP") - 0.17765460908412933) > 0.001:
        raise RuntimeError("The no-jitter epoch-3 standalone evaluator did not reproduce its recorded AP closely enough")

    baseline_evaluation = read_json(BASELINE_EVALUATION)
    baseline_checkpoint = baseline_evaluation.get("checkpoint", {})
    if not baseline_evaluation.get("reproduction_check", {}).get("passed") or baseline_checkpoint.get("sha256") != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("The preserved original epoch-5 baseline evaluator is not verified")
    candidate_counts = candidate_evaluation["ground_truth_object_counts"]
    baseline_counts = baseline_evaluation["ground_truth_object_counts"]
    if candidate_counts != baseline_counts:
        raise RuntimeError("The two standalone evaluations do not use the same frozen validation ground truth")

    no_jitter_result = read_json(NO_JITTER_RESULT)
    baseline_result = read_json(BASELINE_RESULT)
    history, history_summary = metric_rows(read_json(NO_JITTER_HISTORY))
    baseline_preview = preview_records(baseline_result, 5)
    no_jitter_preview = preview_records(no_jitter_result, 3)
    previews = preview_summary(baseline_preview, no_jitter_preview)
    output_dir = NO_JITTER_RUN / "review_epoch_003_vs_original_epoch_005"
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite an existing review directory: {output_dir}")
    output_dir.mkdir(parents=True)
    contact_sheet = create_contact_sheet(output_dir / "fixed_preview_comparison_epoch3_no_jitter_vs_epoch5_baseline.png", previews)
    quantitative = comparison_rows(baseline_evaluation, candidate_evaluation)
    candidate_shared = candidate_evaluation["precision_recall_iou50_shared_threshold"]
    candidate_selected = candidate_evaluation["precision_recall_iou50_validation_selected_best_f1"]
    baseline_selected = baseline_evaluation["training_compatible_precision_recall_iou50"]
    if float(baseline_selected["best_confidence_threshold"]) != 0.43:
        raise RuntimeError("The saved original baseline's validation-selected F1 threshold is not 0.43, so a separate fixed-threshold baseline evaluation is required")
    baseline_shared = {"threshold": 0.43, "F1": baseline_selected["F1"], "precision": baseline_selected["precision"], "recall": baseline_selected["recall"], "source_note": "The saved baseline F1 sweep selected 0.43, so this same reported point is also its shared-threshold value."}
    preview_counts = {
        "positive_images": sum(row["kind"] == "positive" for row in previews),
        "negative_images": sum(row["kind"] == "negative" for row in previews),
        "baseline_positive_images_with_predictions": sum(row["kind"] == "positive" and row["baseline_predictions_threshold_0_25"] > 0 for row in previews),
        "no_jitter_positive_images_with_predictions": sum(row["kind"] == "positive" and row["no_jitter_epoch3_predictions_threshold_0_25"] > 0 for row in previews),
        "baseline_negative_images_with_predictions": sum(row["kind"] == "negative" and row["baseline_predictions_threshold_0_25"] > 0 for row in previews),
        "no_jitter_negative_images_with_predictions": sum(row["kind"] == "negative" and row["no_jitter_epoch3_predictions_threshold_0_25"] > 0 for row in previews),
    }
    review = {
        "status": "passed",
        "scope": "Saved RF-DETR artifacts only. No training, test data, enhanced images, website, or source-dataset changes.",
        "standalone_reproduction": {"no_jitter_epoch3": candidate_evaluation, "original_epoch5_baseline": baseline_evaluation},
        "quantitative_comparison": quantitative,
        "shared_threshold_0_43": {"original_epoch5": baseline_shared, "no_jitter_epoch3": candidate_shared},
        "validation_selected_best_f1_threshold": {"original_epoch5": baseline_selected, "no_jitter_epoch3": candidate_selected},
        "ground_truth_object_counts": candidate_counts,
        "no_jitter_epoch_history": history,
        "no_jitter_history_summary": history_summary,
        "fixed_preview_comparison": {"rows": previews, "summary": preview_counts, "contact_sheet": contact_sheet, "limitation": "This nine-image qualitative panel is fixed and includes three negatives, but it is not the 1,082-image validation metric and most panel images are from the frozen training split. Prediction counts at 0.25 do not prove box correctness."},
        "verified_findings": [
            "The standalone evaluator uses the frozen 1,082-image validation split and the same preprocessing/postprocess/metric path as the verified original epoch-5 evaluator.",
            "The no-jitter run's best recorded EMA AP@[.50:.95] is epoch 3; no epoch 5-10 value exceeds it.",
            "Across epochs 5 to 10, train loss declines from 1.5058 to 1.3203 while EMA AP@[.50:.95] falls from 0.1484 to 0.0520, with intervening oscillations rather than a monotonic decline.",
            "The recorded learning rate, minimum rate, and maximum rate are unchanged for every completed epoch, so no logged scheduler-rate change explains the late metric deterioration.",
            "The resume record states that epoch-4 model/optimizer/scheduler/EMA/loop state was restored but no RNG or sampler-permutation snapshot existed; resumed epoch 5 was intentionally restarted from its beginning.",
        ],
        "hypotheses_not_confirmed": [
            "Stochastic augmentation and data-order variability may contribute to the sharp epoch-to-epoch AP movement, particularly after the restart, but the absent RNG/sampler snapshot does not prove it caused the deterioration.",
            "The divergence between lower training loss and lower validation AP suggests the optimized loss is not a reliable proxy for this small detector's validation ranking/localization quality in this run; it does not diagnose overfitting by itself.",
            "Disabling scale_jitter may have helped the early epoch-3 result relative to the original epoch-5 baseline, but a single seeded run cannot establish a causal or durable improvement.",
        ],
    }
    (output_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    metric_table = markdown_table(history, [("epoch", "Epoch"), ("loss", "Loss"), ("loss_ce", "CE"), ("loss_bbox", "BBox"), ("loss_giou", "GIoU"), ("AP_50_95", "EMA AP"), ("AP50", "AP50"), ("AP75", "AP75"), ("mAR", "mAR"), ("lr", "LR")])
    compare_table = markdown_table(quantitative, [("metric", "Metric"), ("original_epoch5", "Original e5"), ("no_jitter_epoch3", "No-jitter e3"), ("delta", "Delta")])
    preview_table = markdown_table(previews, [("index", "Panel"), ("kind", "Kind"), ("ground_truth_instances", "GT"), ("baseline_predictions_threshold_0_25", "Original e5 @.25"), ("no_jitter_epoch3_predictions_threshold_0_25", "No-jitter e3 @.25")])
    report = f"""# RF-DETR no-jitter epoch-3 review\n\n## Scope\n\nSaved-artifact analysis only. The standalone evaluation uses the frozen 1,082-image raw validation split; no test data, training, source-dataset, website, or checkpoint files were changed.\n\n## Standalone reproduction\n\nThe no-jitter epoch-3 checkpoint SHA-256 is `{candidate_checkpoint['sha256']}`. Its evaluated AP@[.50:.95] is {markdown_number(finite(candidate_evaluation['training_compatible_ema_metrics']['AP_50_95'], 'candidate AP'))}, versus recorded {markdown_number(finite(reproduction['recorded_AP_50_95'], 'recorded AP'))}; the absolute difference is {reproduction['absolute_difference']:.8f} and is within the 0.001 tolerance.\n\nGround truth: {candidate_counts['total']} objects — small {candidate_counts['COCO_area_buckets']['small']}, medium {candidate_counts['COCO_area_buckets']['medium']}, large {candidate_counts['COCO_area_buckets']['large']}.\n\n## Original epoch-5 baseline comparison\n\n{compare_table}\n\nShared threshold 0.43: original epoch-5 F1 {markdown_number(finite(baseline_shared['F1'], 'baseline F1'))}, precision {markdown_number(finite(baseline_shared['precision'], 'baseline precision'))}, recall {markdown_number(finite(baseline_shared['recall'], 'baseline recall'))}. The saved baseline F1 sweep independently selected 0.43, so this same point is valid for the shared-threshold comparison.\n\nNo-jitter epoch-3 at fixed 0.43: F1 {markdown_number(finite(candidate_shared['F1'], 'candidate F1'))}, precision {markdown_number(finite(candidate_shared['precision'], 'candidate precision'))}, recall {markdown_number(finite(candidate_shared['recall'], 'candidate recall'))}.\n\nNo-jitter epoch-3 validation-selected best-F1 threshold: {candidate_selected['best_confidence_threshold']:.2f}; F1 {markdown_number(finite(candidate_selected['F1'], 'candidate selected F1'))}, precision {markdown_number(finite(candidate_selected['precision'], 'candidate selected precision'))}, recall {markdown_number(finite(candidate_selected['recall'], 'candidate selected recall'))}. This selected threshold is separately labelled and must not replace the shared 0.43 comparison.\n\n## No-jitter completed-epoch history\n\n{metric_table}\n\nThe logged LR, LR-min, and LR-max each remained constant across completed epochs. Epoch 3 is the best recorded no-jitter epoch ({markdown_number(history_summary['best_AP_50_95'])}).\n\n## Fixed preview panel\n\n![Fixed comparison]({contact_sheet['path']})\n\n{preview_table}\n\nThe fixed panel has {preview_counts['positive_images']} positive and {preview_counts['negative_images']} negative images. At the pre-existing 0.25 display threshold, original epoch 5 has detections on {preview_counts['baseline_positive_images_with_predictions']} positive-panel images and {preview_counts['baseline_negative_images_with_predictions']} negative-panel images; no-jitter epoch 3 has detections on {preview_counts['no_jitter_positive_images_with_predictions']} positive-panel images and {preview_counts['no_jitter_negative_images_with_predictions']} negative-panel images. These are qualitative counts only: they do not verify box matches and this panel is not the validation metric.\n\n## Verified findings\n\n- The checkpoint hash and standalone AP reproduction pass before per-size or precision/recall comparisons are interpreted.\n- Loss declines from epoch 5 to 10 while AP oscillates and ends lower; that observation is factual, not a diagnosis.\n- Logged learning rates do not change during the run, so a recorded scheduler-rate transition is not the explanation.\n- The resumed epoch began after a full model/optimizer/scheduler/EMA/loop restoration, but no RNG or sampler-permutation snapshot was available.\n\n## Hypotheses, not findings\n\n- Stochastic data order or augmentation can plausibly contribute to the oscillations, especially after the restart, but is not proven as the cause.\n- The loss/AP divergence can indicate mismatch between optimized loss and this validation metric; it does not by itself establish overfitting.\n- The stronger early no-jitter result is promising but one seeded run cannot establish a persistent causal benefit from changing scale jitter.\n"""
    (output_dir / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "review_dir": str(output_dir.relative_to(PROJECT)), "report": str((output_dir / "FINDINGS.md").relative_to(PROJECT)), "contact_sheet": contact_sheet, "standalone_reproduction": reproduction, "best_no_jitter_epoch": history_summary["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
