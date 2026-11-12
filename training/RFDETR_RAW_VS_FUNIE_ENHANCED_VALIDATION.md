# Validation-only raw versus FUnIE-enhanced detector comparison

This evaluation compares the preserved raw-trained, scale-jitter-disabled
RF-DETR Small epoch-3 EMA checkpoint on two versions of the same 1,082-image
frozen TrashCan validation split:

- **Raw:** the unmodified source images.
- **Enhanced:** new PNG derivatives produced by the preserved epoch-20
  frozen-BatchNorm FUnIE-GAN candidate.

It is inference-only. No detector training occurs, no test image is mounted,
and no dataset/checkpoint/site file is overwritten.

## Candidate integrity and coordinate alignment

The runner requires these exact checkpoint hashes before MPS inference:

- RF-DETR Small epoch-3 EMA: `6fbfadfa5509630952be4c0b129740a507ec455799a760f27824f9db98640440`
- FUnIE-GAN frozen-BatchNorm epoch-20: `c9259bc5257ef28c58c09090ee165c8c990eaab49f455bdc86c147363c5b77d5`

For every image, enhancement uses an aspect-preserving bicubic resize with
longest side 256, reflection padding to a 256×256 FUnIE canvas, removal of
the same padding after generation, then bicubic resize back to the exact
original width and height. It verifies every enhanced derivative decodes at
the original dimensions. Frozen COCO annotations are not copied, transformed,
or rewritten; the detector uses their unchanged original-pixel coordinates for
both inputs.

The detector preprocessing and validation settings are identical for both
branches: RGB decode, antialiased 512×512 resize, checkpoint normalization,
unthresholded RF-DETR postprocess, and the pinned `vernier` COCO backend.
Model inference and postprocess must remain on MPS; CPU work is restricted to
image encoding and metric aggregation.

## Run in macOS Terminal

```bash
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/evaluate_rfdetr_raw_vs_funie_enhanced.py --run-id "rfdetr_raw_vs_funie_enhanced_val_mps_$(date -u +%Y%m%dT%H%M%SZ)"
```

The new evaluation directory contains the enhanced derivatives, a per-image
shared-threshold comparison CSV, raw/enhanced AP and per-size metrics, timing,
and paired previews selected for improvements, regressions, and negatives.

Interpret the result narrowly: this evaluates **inference-time enhancement
with a detector trained on raw images**. A negative result does not establish
that detector training or fine-tuning on enhanced images could not help.
