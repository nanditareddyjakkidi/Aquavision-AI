# RF-DETR no-jitter lower-learning-rate continuation: epochs 4-8

`preparation/continue_rfdetr_no_jitter_epoch3_lower_lr.py` prepares a
separate controlled exploratory continuation of the raw-image, fixed-512,
`scale_jitter=False` RF-DETR Small experiment. It starts from the completed
epoch-3 full state and stops after completed epoch 8.

## Resume source

The only permitted resume source is:

`training_runs/rfdetr/rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_2.ckpt`

Its SHA-256 is
`6f58bb8b3310b728563e831272aca42f65f45fe671c861820715b065ff3f2a70`.
It records zero-based epoch 2, completed epoch 3, global step 15,144, model
state, populated AdamW slots, the step scheduler, EMA callback state, and
Lightning loop state. Its EMA weights exactly match the preserved epoch-3
best-inference checkpoint (AP@[.50:.95] 0.1776546091):

`training_runs/rfdetr/rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_best_ema.pth`

That `.pth` file is deliberately **not** used as a resume source because it
has no optimizer or scheduler state.

## Sole training change

After Lightning restores the full training state, the runner divides all 28
optimizer parameter-group learning rates by five. It applies the same factor
to `initial_lr`, scheduler `base_lrs`, and scheduler `_last_lr`, preserving
each group’s relative rate and ensuring later scheduler steps retain the
change. The maximum learning rate changes from `1e-4` to `2e-5`; the top-level
encoder rate changes from `1.5e-4` to `3e-5`.

The runner fails if it cannot verify the restored model, optimizer, scheduler,
EMA, epoch, step, or the fivefold learning-rate change before epoch 4 begins.
It records all parameter-group rates at the start and end of every new epoch,
and verifies the saved full-state checkpoints retain scaled scheduler bases.

All other settings remain fixed: frozen raw train/validation splits,
negatives, one `visible_plastic_debris` class, batch size 1, resolution 512,
`multi_scale="off"`, `scale_jitter=False`, losses, AdamW state, scheduler
counters, EMA settings, and MPS-only execution with CPU fallback disabled.

The full checkpoint has no serialized Python, NumPy, Torch, MPS, sampler, or
dataloader-generator RNG state. The runner re-seeds the known generators with
`20260930` before model construction and restores the available Lightning loop
progress. It documents that this is not a bitwise stochastic replay.

## Outputs

A new run directory contains the configuration and restoration audits, exact
fixed preview images for EMA epochs 4-8, per-epoch losses/metrics, actual
learning-rate records, full-state checkpoint audits, and comparisons against
the previous no-jitter epochs 4-8 and starting epoch-3 AP. It makes run-local
links for the best inference checkpoint across the preserved epoch-3 candidate
and new epochs, plus the latest epoch-8 full-state checkpoint.

## macOS Terminal command

```bash
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/continue_rfdetr_no_jitter_epoch3_lower_lr.py --run-id "rfdetr_no_jitter_epoch3_lower_lr_e4to8_mps_$(date -u +%Y%m%dT%H%M%SZ)" --epochs 8 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9
```

## Runtime estimate

The prior no-jitter continuation took 56.1 minutes for its first resumed
epoch, then averaged 34.1 minutes for each of the next five epochs. For five
epochs (4-8) plus checkpointing and previews, budget roughly **3 hours 15
minutes**, with **3.5 hours** a sensible allowance. The lower learning rate is
not expected to materially change runtime.
