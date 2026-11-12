# FUnIE-GAN BatchNorm-statistics ablation

`preparation/run_funie_batchnorm_ablation.py` is a deliberately narrow,
MPS-only experiment. It copies the preserved original epoch-1 checkpoint into
a new run folder and continues only epochs 2–4.

It retains the source checkpoint's generator, discriminator, both Adam
optimizer states, epoch-2 data order, seed, 256x256 paired preprocessing,
batch size 1, official GAN/L1/VGG losses, VGG input convention, frozen UIEB
splits, and MPS-only execution. It does not read UIEB test/challenging images.

The single changed behavior is applied after every `generator.train()` and
`discriminator.train()` call: each `BatchNorm2d` module is switched to eval
mode. Its affine weight and bias remain `requires_grad=True` and are left in
the original optimizer parameter groups. The runner verifies at the first and
last training batches of every epoch that running means, running variances,
and batch counters remain identical to epoch 1; it also requires finite,
nonzero affine gradients for every BatchNorm layer.

The resulting separate run contains a byte-checked copy of its source
checkpoint, its own checkpoints, validation metrics, fixed previews, and a
matched epoch-2/3/4 comparison against the original run. Each comparison reads
the same 134 frozen validation pairs, computes per-image changes relative to
the raw baseline and the original checkpoint, and produces sheets containing
raw input, original output, BatchNorm-frozen output, and reference. The sheets
explicitly include the prior color-shift cases `725_img_.png`, `809_img_.png`,
`832_img_.png`, `727_img_.png`, and `920_img_.png`.

This freezes epoch-1 BatchNorm statistics. It is a targeted test of one
batch-size-one risk; it cannot conclusively prove or disprove the underlying
cause of the earlier instability.

Run it only in native macOS Terminal:

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/run_funie_batchnorm_ablation.py --run-id 'funie_batchnorm_frozen_e2to4_mps_20260930' --source-checkpoint 'training_runs/funie/funie_pilot_mps_20260930T155101Z/checkpoints/epoch_001.pt' --epochs 4 --seed 20260930 --preview-count 5
```

The runner refuses to overwrite that run directory and does not launch RF-DETR
or modify the website, source datasets, frozen manifests, original FUnIE-GAN
run, or its checkpoints.

## Bounded continuation through epoch 20

`preparation/continue_funie_batchnorm_ablation.py` resumes the existing
separate ablation **only** from its verified epoch-4 checkpoint and stops after
epoch 20. It refuses to alter the initial configuration, checks generator,
discriminator, both Adam optimizer states, completed epoch/global step, Python/
NumPy/CPU/MPS RNG states, frozen-manifest checksum, and the recorded epoch-1
BatchNorm buffers before any optimization.

The continuation re-applies BatchNorm eval mode after every `model.train()`
call and verifies the epoch-1 running mean, running variance, and batch counter
buffers remain unchanged while affine parameters receive finite, nonzero
gradients. It validates all 134 frozen pairs at every epoch; saves the latest
checkpoint, retains the best-PSNR checkpoint, writes standard and known
color-shift previews, and compares each epoch against both its matching
original checkpoint and the raw-input baseline. It writes full per-image
metrics plus an explicit raw-baseline regression CSV for every continued epoch.

Run the bounded continuation only in native macOS Terminal:

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/continue_funie_batchnorm_ablation.py --run-id 'funie_batchnorm_frozen_e2to4_mps_20260930' --resume 'training_runs/funie/funie_batchnorm_frozen_e2to4_mps_20260930/checkpoints/latest.pt' --epochs 20 --seed 20260930 --preview-count 5
```

It will not automatically extend past epoch 20. It preserves the original run,
the test and challenging UIEB images, the website, and RF-DETR unchanged.

## Evaluation-only finalization

If all three ablation epochs finish but comparison artifact generation is
interrupted, do **not** relaunch training. Instead, use the finalizer below. It
verifies that the existing epoch-2/3/4 checkpoints preserved their epoch-1
BatchNorm buffers, then creates only the per-image comparison CSVs, sheets,
and report:

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/finalize_funie_batchnorm_ablation.py --run-id 'funie_batchnorm_frozen_e2to4_mps_20260930'
```
