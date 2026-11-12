# RF-DETR `scale_jitter=False` resume: epochs 5–10

This procedure resumes the interrupted raw-image `scale_jitter=False` RF-DETR
Small ablation from its completed epoch-4 full-state checkpoint. It reruns
epoch 5 from the beginning and stops after ten total completed epochs. It does
not use the test split, enhanced images, the website, or source datasets.

## Immutable source and checkpoint audit

The only accepted source checkpoint is:

`training_runs/rfdetr/rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_3.ckpt`

Before training, the runner requires the checkpoint to contain model weights,
one populated optimizer state, one scheduler state, RF-DETR EMA state, and
Lightning loop state. It requires zero-based epoch `3`, completed epoch `4`,
and global step `20,192` (four times 5,048 train batches). At runtime it checks
that Lightning actually restores those model, optimizer, scheduler, EMA,
epoch, and step states before it permits epoch 5 to start.

The checkpoint has no serialized Python, NumPy, Torch, MPS RNG, sampler
permutation, or dataloader-generator snapshot. The runner re-seeds those
available generators with `20260930` before model construction and passes the
same seed to the pinned RF-DETR implementation. This supports a reproducible
fresh epoch-5 start as far as the implementation permits; it does **not** claim
bitwise replay of stochastic augmentation or shuffle order. The interrupted
epoch-5 CSV telemetry remains in the source run and is recorded separately,
never treated as completed results.

## Preserved configuration

The continuation retains frozen raw TrashCan v2 train/validation data, the
one-class `visible_plastic_debris` mapping, negatives, batch size 1, resolution
512, `multi_scale="off"`, `scale_jitter=False`, AdamW, the step scheduler,
EMA, and all augmentation/evaluation settings. It requires MPS and refuses any
nonzero `PYTORCH_ENABLE_MPS_FALLBACK` value. It writes a new read-only COCO
adapter and all continuation outputs under a new run directory.

After epochs 5–10, it verifies every new full-state checkpoint, makes the
fixed preview panel, evaluates the combined best and epoch-10 EMA checkpoints,
and creates stable aliases for the latest full-state checkpoint and the best
EMA inference checkpoint. Best selection considers epochs 1–4 and 5–10, so the
preserved source epoch-3 EMA checkpoint remains selected if no later epoch
exceeds it.

## Run in macOS Terminal

```bash
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/resume_rfdetr_raw_v2_scale_jitter_off.py --run-id "rfdetr_raw_v2_scale_jitter_off_e5to10_resume_mps_$(date -u +%Y%m%dT%H%M%SZ)" --source-checkpoint 'training_runs/rfdetr/rfdetr_raw_v2_scale_jitter_off_e1to10_mps_20261001T155341Z/training_output/checkpoint_3.ckpt' --epochs 10 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9
```
