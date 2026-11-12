# Staged FUnIE-GAN training

`preparation/run_funie_staged.py` is the MPS-only runner for the frozen UIEB v2
enhancement data. It reads only the 623 `train` pairs and 134 `val` pairs from
`datasets/processed/uieb_enhancement_v2_leakage_resolved/pair_split_manifest.csv`.
It never reads the 133 derived test pairs or the 60 challenging images.

All run outputs are isolated below `training_runs/funie/<run-id>/`; this is not
the smoke-test directory and is not a final-model location. A run directory
contains its immutable launch configuration, epoch checkpoints, JSON-lines
metrics, and fixed validation comparison images. A resume uses the same
`--run-id` and a local checkpoint from that directory.

## Recipe preserved from the pinned official implementation

Pinned source: `third_party/FUnIE-GAN/PyTorch/train_funiegan.py` at revision
`8f934c834c94e007b00866186b9ee624dc2b7b69`.

- Models: `GeneratorFunieGAN` and `DiscriminatorFunieGAN`, initialized with
  the source `Weights_Normal` routine.
- Paired preprocessing: RGB bicubic resize to 256×256 and normalization to
  `[-1, 1]`. Training uses deterministic, aligned horizontal flips with
  probability 0.5; validation has no flip.
- Discriminator: `0.5 × (MSE(D(reference, raw), 1) + MSE(D(G(raw), raw), 0)) × 10`.
- Generator: `MSE(D(G(raw), raw), 1) + 7 × L1(G(raw), reference) + 3 × VGG19_conv5_2_MSE(G(raw), reference)`.
- Perceptual component: the official frozen pretrained VGG19 feature extractor
  at module 30 (`conv5_2`). In accordance with the pinned source, the already
  `[-1, 1]` tensors are passed directly to VGG19; no additional ImageNet
  normalization is introduced.

The implementation creates the discriminator fake image under `no_grad` during
the discriminator update. This avoids an unused generator gradient while
preserving the official discriminator objective. It is otherwise the full
official three-term generator recipe, not the earlier adversarial-plus-L1 smoke
loss.

## Pilot command

Run only from native macOS Terminal after confirming MPS is available. Do not
pass a CPU-compatibility option; the script aborts if MPS is unavailable or if
`PYTORCH_ENABLE_MPS_FALLBACK` is nonzero.

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/run_funie_staged.py --run-id "funie_pilot_mps_$(date -u +%Y%m%dT%H%M%SZ)" --epochs 1 --seed 20260930 --preview-count 5
```

The command runs one complete 623-pair, batch-1 epoch and then all 134 frozen
validation pairs. It saves an epoch checkpoint, reloads it on MPS for inference,
records finite loss checks, parameter updates, validation PSNR/SSIM, MPS/process
memory observations, and a fixed five-pair raw/generated/reference comparison.
It does not train RF-DETR.

The runner also evaluates and saves the unenhanced raw-input PSNR/SSIM baseline
on the exact same 134 validation pairs, with identical resize and metric
preprocessing. After every epoch it writes JSON curve data and a PNG covering
validation PSNR/SSIM (including baseline lines), mean train losses, epoch
duration, and MPS driver allocation. The JSON retains current MPS allocation
and process high-water RSS as well. It keeps `latest.pt` and
`best_validation_psnr.pt` while retaining numbered epoch checkpoints and fixed
validation previews.

## Resume example

Use the existing run ID and checkpoint, then raise total epochs. This trusts
only a checkpoint created by the same local run.

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/run_funie_staged.py --run-id '<existing-run-id>' --epochs 2 --seed 20260930 --resume "training_runs/funie/<existing-run-id>/checkpoints/latest.pt"
```

## Continue the completed pilot to 20 total epochs

The pilot checkpoint contains generator, discriminator, both Adam optimizer
states, epoch, global step, best validation PSNR, and Python/NumPy/CPU-torch
RNG state. The version created for epoch 1 has no saved MPS RNG state. That
does not alter this recipe's continuation because it uses no stochastic MPS
layers and its data order plus paired flips are deterministic by epoch and
index; later checkpoints also save/restore MPS RNG state. Resume checkpoints
are deserialized on CPU first so their RNG byte tensors can be restored, then
their model and optimizer states are loaded onto MPS. Every resume is audited
in `resume_events.jsonl` before it is allowed to train.

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/run_funie_staged.py --run-id 'funie_pilot_mps_20260930T155101Z' --epochs 20 --seed 20260930 --preview-count 5 --resume 'training_runs/funie/funie_pilot_mps_20260930T155101Z/checkpoints/latest.pt'
```
