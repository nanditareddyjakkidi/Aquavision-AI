# RF-DETR Small raw-image continuation: epochs 6–10

`preparation/run_rfdetr_raw_v2_e5_to_e10.py` is a bounded, separate
continuation from the preserved five-epoch raw-image run. It does not write in
that existing run directory, the frozen datasets, the website, the enhancement
outputs, or any test directory.

## Exact resume point

The runner uses only this full Lightning checkpoint:

`training_runs/rfdetr/rfdetr_raw_v2_e1to5_fixed512_mps_20261001T091117Z/checkpoints/latest_full_state.ckpt`

The alias must resolve to `training_output/checkpoint_4.ckpt` and have SHA-256
`0e77c5d93a990a8df810f4de1f7b781483aceba33ffb1bac21b267d200f97c0d`.
Before training, the runner requires and records:

- epoch `4` / completed epoch `5` and global step `25,240`;
- 488 model-state keys;
- one optimizer state with 486 populated slots, all at step `25,240`;
- the LR scheduler at `last_epoch=25,240`, `_step_count=25,241`; and
- RF-DETR EMA state with 489 averaged-state keys and
  `latest_update_step=25,240`.

It is therefore a model/optimizer/scheduler/loop/EMA continuation, not a
weights-only restart. The checkpoint has no saved Python, NumPy, or Torch RNG
snapshot, so it does not claim bitwise replay of random augmentation or
shuffling.

At live `on_train_start`, after RF-DETR has reconstructed its EMA model, the
runner verifies the restored model-state count, optimizer slots and step,
scheduler counters, EMA state/count, current epoch, and global step before it
allows the first epoch-6 batch to run.

The preserved source `latest_full_state.ckpt` and its existing best EMA
inference checkpoint are hash-checked again after the new run, but are never
modified.

## Unchanged training conditions

The new run creates a read-only symlink adapter over only the frozen raw
`material_version` train and validation images: 5,048 training images with
3,806 negatives, and 1,082 validation images with 833 negatives. It retains
the one COCO category ID 1 / RF-DETR label 0:
`visible_plastic_debris`. It mounts no test image and no enhanced image.

It requires native MPS and `PYTORCH_ENABLE_MPS_FALLBACK=0`, batch size 1,
512-pixel resolution, `multi_scale="off"`, `expanded_scales=False`, the same
AdamW/loss/augmentation settings, and the same fixed seed. The image
augmentation backend remains the pinned RF-DETR `"cpu"` preprocessing backend;
that is not model-operation CPU fallback.

The scheduler is still the pinned managed step scheduler: step interval,
`warmup_epochs=0`, LR `1e-4`, encoder LR `1.5e-4`, and the source defaults
`lr_drop=100`, `min_factor=0`. Its nominal total step budget changes from
25,240 to 50,480 for the new ten-epoch target, but there is no scheduled rate
change through completed epoch 10. The restored scheduler counter is checked
before running and each new full checkpoint is checked afterward.

## Validation and checkpoint artifacts

RF-DETR evaluates EMA each epoch. The source configuration has
`use_ema=true` and `eval_base_model=false`, so the runner labels every
validation series it writes explicitly as `EMA` rather than treating the
unprefixed logger columns as regular-model metrics.

The new run directory records:

- full-state `training_output/checkpoint_5.ckpt` through `checkpoint_9.ckpt`
  for completed epochs 6–10;
- `checkpoints/latest_full_state.ckpt`, an alias to the epoch-10 full state;
- a best validation `AP@[.50:.95]` **EMA** inference alias selected across
  all epochs 1–10, without replacing the old run's best checkpoint;
- `metric_history_ema.json` with explicit EMA AP@[.50:.95], AP50, AP75, mAR,
  F1, precision, recall and training losses;
- per-epoch timing and MPS allocator observations; and
- `preview_plan.json` written before training, plus matching epoch-5–10
  raw/ground-truth/EMA-prediction previews.

The fixed preview plan has nine images from nine distinct video groups: two
positive examples each for original-pixel COCO small, medium and large objects,
plus three negatives. It is selected deterministically by video group and path
before the continuation begins, not from the new results.

## Run from native macOS Terminal

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/run_rfdetr_raw_v2_e5_to_e10.py --run-id "rfdetr_raw_v2_e5to10_fixed512_mps_$(date -u +%Y%m%dT%H%M%SZ)" --source-run-id 'rfdetr_raw_v2_e1to5_fixed512_mps_20261001T091117Z' --epochs 10 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9
```

The runner refuses an existing target folder, a CPU fallback, a missing or
altered frozen input, a source checkpoint without full resume state, a changed
source checkpoint, a changed model/training configuration outside documented
resume fields, or a missing EMA validation/loss record for epochs 6–10. It
stops after completed epoch 10.
