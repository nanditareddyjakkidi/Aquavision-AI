# RF-DETR Small raw-image continuation: epochs 1–5

`preparation/run_rfdetr_raw_v2_continuation.py` continues the successful
fixed-512 raw-image pilot to five total epochs. It does not start a new
weights-only fine-tuning run.

## Resume source

The continuation resumes only from the pilot's full Lightning checkpoint:

`training_runs/rfdetr/rfdetr_raw_v2_fixed512_mps_20261001T065835Z/training_output/checkpoint_0.ckpt`

Its recorded SHA-256 is
`7a2bb85878d4477a45532d47f26a1405a2a986406f22359fc741d4070d6b52d7`.
The runner refuses a changed source checkpoint. Before training, it verifies:

- 488 model-state keys;
- 486 populated optimizer-state slots at global step 5,048;
- the saved LR-scheduler state at step 5,048;
- epoch 0 / completed epoch 1 and Lightning loop progress; and
- the EMA callback's averaged model state at update step 5,048.

This is a full-state PyTorch Lightning resume via `resume=checkpoint_0.ckpt`.
The checkpoint does not contain a Python, NumPy, or PyTorch RNG snapshot, so
the run does **not** claim bitwise replay of stochastic augmentation/shuffle
state. Model, optimizer, scheduler, loop, callback, and EMA state do resume.

## Fixed data and model settings

The runner recreates a new read-only symlink adapter containing only the
frozen v2 `material_version` raw train/validation files. It verifies their
checksums and COCO annotations before training. The adapter has no test
directory and never contains enhancement output.

It preserves the one-class `visible_plastic_debris` mapping, all 5,048 train
images (including 3,806 negatives), all 1,082 validation images (including 833
negatives), batch size 1, 512×512 tensors, `multi_scale="off"`,
`expanded_scales=False`, MPS-only execution, and
`PYTORCH_ENABLE_MPS_FALLBACK=0`.

## Scheduler effect

The pilot used RF-DETR's managed `step` scheduler with a step interval,
`warmup_epochs=0`, no scheduler kwargs, and the pinned defaults `lr_drop=100`
and `min_factor=0`. The new target changes the configured total optimizer-step
budget from 5,048 to 25,240, but does not change the effective learning rate
over epochs 1–5: the first 10× drop is at epoch 100 and there is no warm-up.
The runner records this explicitly and verifies the restored scheduler counter
and the four newly saved full checkpoints.

## Outputs

The new run directory retains the original pilot unmodified and records:

- `training_output/checkpoint_1.ckpt` through `checkpoint_4.ckpt`: full-state
  checkpoints for completed epochs 2–5;
- RF-DETR's EMA best-validation checkpoint plus a stable
  `checkpoints/best_validation_ap_50_95_ema.pth` alias, selected across all
  five epochs including epoch 1;
- `checkpoints/latest_full_state.ckpt`, an alias to the epoch-5 full state;
- `metric_history.json`, including validation AP@[.50:.95], AP50, AP75, mAR,
  F1, precision/recall, and loss history for every epoch;
- per-epoch MPS memory/timing observations; and
- five fixed positive/negative raw-validation preview sets, one generated from
  the EMA state of each epoch's full checkpoint.

Run only from native macOS Terminal. `caffeinate -i` keeps the Mac awake while
the bounded four remaining epochs complete:

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/run_rfdetr_raw_v2_continuation.py --run-id "rfdetr_raw_v2_e1to5_fixed512_mps_$(date -u +%Y%m%dT%H%M%SZ)" --pilot-run-id 'rfdetr_raw_v2_fixed512_mps_20261001T065835Z' --epochs 5 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 5
```

The runner refuses CPU fallback, a changed pilot checkpoint, changed frozen
files, an altered resolved configuration, an existing output directory, or a
checkpoint that lacks optimizer/scheduler/EMA state. It stops after epoch 5.
