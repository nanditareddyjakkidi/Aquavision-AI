# AquaVision training readiness report

Date: 2026-09-30

## Scope and protections

- Project root: `/Users/nanditareddy/AquaVision AI`.
- `website/`, `serve.py`, the copied source datasets, and the frozen derived
  test sets were not changed. The website checksum manifest passed after this
  work.
- No full training was launched. All temporary adapters, checkpoints, logs, and
  model downloads are under `smoke_tests/` or `.cache/` in this project.
- The final preparation freeze remains valid: all 58 listed files passed
  checksum verification. Freeze manifest SHA-256:
  `1593fa11d21cf9a21d21e0a3f0c8d72de7b89b76aba05a30d13ecb6db7c66c37`.

## Leakage review and frozen data

The original exact-dHash review had 38 cross-split candidates. Side-by-side
contact sheets were created and reviewed in
`datasets/processed/trashcan_material_plastic_v1/leakage_review/contact_sheets/`.
Equal dHash was treated only as a review signal, not duplicate proof.

The conservative action was to coalesce `video:000257` and `video:000258`:
their low-detail bright-water frames could not be confidently distinguished as
unrelated. Both groups are now in the final training split. The regenerated v2
split produced 40 review candidates (31 carried forward and 9 new); all 40
have side-by-side sheets and decisions in
`datasets/processed/trashcan_material_plastic_v2_leakage_resolved/leakage_review/`.
Review found no visible evidence of a shared scene/object sequence among the
remaining cross-split candidates, while retaining the uncertainty that manual
visual review cannot prove independence.

Final validation reports zero video-group split violations and zero UIEB pair
group split violations. The frozen manifests are:

- Detector: `datasets/processed/trashcan_material_plastic_v2_leakage_resolved/`
- UIEB: `datasets/processed/uieb_enhancement_v2_leakage_resolved/`
- Freeze record: `datasets/processed/FINAL_MANIFEST_FREEZE_v2.json`

Usable frozen data:

| Task | Train | Validation | Held-out test | Other held out |
| --- | ---: | ---: | ---: | --- |
| UIEB paired enhancement | 623 pairs | 134 pairs | 133 pairs | 60 challenging images, never used for training/model selection |
| One-class plastic detection | 5,048 images / 1,299 instances | 1,082 / 289 | 1,082 / 276 | Supplied benchmark splits retained separately |

The detector's only category is COCO category ID `1`,
`visible_plastic_debris`; RF-DETR maps it to contiguous foreground label `0`.
The detector training split includes negative images.

## Local environment and provenance

Hardware reported by macOS: Apple M3 Pro, 18 GB memory. A project-local
Python 3.12.12 environment was created at `.venv/`; no global Python packages
were installed. Exact installed packages are in
`environment/smoke-test-requirements.txt`.

Key versions:

- PyTorch 2.14.0; torchvision 0.29.0
- RF-DETR package 1.12.0.dev0, installed editable from official source revision
  `f765186e79ac063b358cc279c361ef0cf4cb82fd`
- FUnIE-GAN official source revision
  `8f934c834c94e007b00866186b9ee624dc2b7b69`

Official sources were cloned in `third_party/FUnIE-GAN/` and
`third_party/rf-detr/`. Pip, Torch, Hugging Face, RF-DETR, XDG, and Matplotlib
caches were configured under `.cache/` for the final runs.

### MPS status

`torch.backends.mps.is_built()` is `true`, but
`torch.backends.mps.is_available()` is `false` in this runtime. Consequently:

- No MPS operation was executed or silently replaced with CPU.
- MPS operator compatibility is **untested**, not passed or failed. There are
  no individual unsupported operators to report because the MPS device could
  not be initialized.
- Both smoke tests required the explicit `--allow-cpu-compatibility` opt-in and
  record `device: cpu`. Restore MPS availability before any real MPS training.

### MPS investigation — 2026-09-30

This investigation made no dependency, website, dataset, or frozen-manifest
change. The actual host and Python process are:

- Host: MacBook Pro with Apple M3 Pro (14 GPU cores), 18 GB memory, and Metal
  reported as supported by `system_profiler`.
- OS: macOS 26.6.2 (build 25G83), Darwin kernel 25.6.0, `arm64`.
- Python executable:
  `/Users/nanditareddy/AquaVision AI/.venv/bin/python`.
  `file` identifies it as a Mach-O 64-bit `arm64` executable; Python reports
  `platform.machine() == "arm64"` and
  `platform.platform() == "macOS-26.6.2-arm64-arm-64bit"`. This is native
  Apple Silicon Python, not Rosetta.
- PyTorch: 2.14.0 from
  `.venv/lib/python3.12/site-packages/torch/`; `torchvision` is 0.29.0.
- MPS probes: `is_built() == true`, `is_available() == false`,
  `torch.mps.device_count() == 0`. PyTorch's low-level macOS version probes
  report both macOS 14+ and macOS 26+ as true.
- A direct `torch.ones(1, device="mps")` fails before allocation with:
  `RuntimeError: The MPS backend is supported on macOS 14.0+. Current OS version can be queried using sw_vers`.
  This message conflicts with the detected macOS 26.6.2 version; it is not an
  unsupported-operation error.
- No inherited `MPS_*`, `PYTORCH_*`, `METAL_*`, `DYLD_*`, or CUDA environment
  settings were present. The installed `libtorch_cpu.dylib` links to the Apple
  Metal, MetalPerformanceShaders, and MetalPerformanceShadersGraph frameworks.

The execution shell itself is permission-restricted: attempts to query normal
process/system state through `ps` and `sysctl` received `Operation not
permitted`. That makes Metal access unavailable to this Codex-run process even
though the host advertises a Metal-capable Apple GPU. This is therefore **not an
incompatible x86/Rosetta Python or a CPU-only PyTorch wheel**.

There is also a separate, unresolved host-compatibility possibility: the
PyTorch project's open issue
[#177819](https://github.com/pytorch/pytorch/issues/177819) documents the same
native-arm64 symptoms and misleading macOS-14 error on macOS 26.3.1, including
with a nightly build. The current PyTorch MPS guide says that `is_built()` only
means the wheel contains MPS support; `is_available()` additionally requires a
working MPS runtime/device. See the [official MPS
guide](https://docs.pytorch.org/docs/main/notes/mps.html). On the evidence
available here, the immediate blocker is the restricted execution environment;
the macOS 26/PyTorch issue cannot be ruled out until the same environment is
tested from ordinary macOS Terminal.

Run this one exact diagnostic command in **macOS Terminal** (not inside Codex)
to test the same project virtual environment without a CPU fallback:

```sh
cd '/Users/nanditareddy/AquaVision AI' && PYTHONUNBUFFERED=1 .venv/bin/python -c 'import sys,platform,torch; print("executable=",sys.executable); print("platform=",platform.platform()); print("torch=",torch.__version__); print("mps_built=",torch.backends.mps.is_built()); print("mps_available=",torch.backends.mps.is_available()); print("mps_device_count=",torch.mps.device_count()); print("mps_tensor=",torch.ones(1,device="mps"))'
```

Interpretation:

- If it prints `mps_available= True` and an MPS tensor, the project virtual
  environment is sound and this Codex execution environment is the blocker.
  The two bounded smoke scripts must then be run from a Metal-capable execution
  environment, one at a time, without `--allow-cpu-compatibility`.
- If it repeats the current failure, preserve the output and treat it as the
  macOS 26/PyTorch MPS compatibility issue. Do not reinstall, downgrade, or
  silently switch to CPU without choosing an explicit remediation after the
  upstream issue changes.

No MPS smoke-test timing exists because MPS device allocation fails. The only
measured model timings remain the explicit CPU compatibility results above;
they are not MPS performance estimates.

### MPS smoke reruns in macOS Terminal — 2026-09-30

The same project `.venv` was then tested from ordinary macOS Terminal, where
MPS is available. The user explicitly set `PYTORCH_ENABLE_MPS_FALLBACK=0` for
the RF-DETR invocation, omitted the scripts' CPU-compatibility flag for both
runs, and saved the following independent result files:

| Model | Device | Bounded work | Elapsed | Process high-water RSS | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| FUnIE-GAN | `mps` | 2 batch-1 steps | 6.043 s | 563,281,920 bytes | Passed finite losses, backward pass, parameter update, checkpoint reload, and inference |
| RF-DETR Small | `mps` | 5 observed batch-1 optimizer steps (ceiling 10) | 29.673 s | 1,792,770,048 bytes | Passed official training path, checkpoint reload, and inference |

Both result files record `mps_built: true`, `mps_available: true`,
`device: "mps"`, and the same frozen-manifest SHA-256. The RF-DETR Lightning
log also reports `GPU available: True (mps), used: True`. Its
`augmentation_backend: "cpu"` setting is the chosen image-preprocessing
backend; it is not a PyTorch model-operation CPU fallback.

Artifacts:

- `smoke_tests/funie/funie_smoke_mps_20260930T150907Z/result.json`
- `smoke_tests/funie/funie_smoke_mps_20260930T150907Z/funie_smoke_checkpoint.pt`
- `smoke_tests/rfdetr/rfdetr_smoke_mps_20260930T151002Z/result.json`
- `smoke_tests/rfdetr/rfdetr_smoke_mps_20260930T151002Z/training_output/checkpoint_best_total.pth`

This resolves the project-environment readiness blocker: MPS works from native
macOS Terminal with the existing virtual environment. The earlier MPS failure
is specific to the restricted Codex execution environment, not a project
dependency defect. These tests still provide no accuracy result and do not
authorize full training.

### Full-recipe FUnIE-GAN one-epoch MPS pilot — 2026-09-30

The staged runner was executed from native macOS Terminal with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. This was an authorized bounded pilot, not a
longer training run. It used the pinned FUnIE-GAN revision
`8f934c834c94e007b00866186b9ee624dc2b7b69` and the documented full objective:
PatchGAN MSE, `7 ×` L1 reconstruction, and `3 ×` frozen pretrained VGG19
conv5_2 feature MSE. The downloaded official VGG19 weights have SHA-256
`dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44`.

Results from `funie_pilot_mps_20260930T155101Z`:

| Item | Result |
| --- | --- |
| Device | `mps`; no CPU fallback |
| Training | 623 batch-1 frozen train pairs, one epoch |
| Validation | all 134 frozen validation pairs |
| Elapsed time | 74.437 s |
| Process high-water RSS | 1,611,333,632 bytes |
| MPS current / driver allocation | 208,188,160 / 1,275,691,008 bytes |
| MPS recommended maximum | 14,302,248,960 bytes |
| Mean validation PSNR / SSIM | 11.6724 dB / 0.2835 |
| Checkpoint reload | Passed on MPS; output shape `[1, 3, 256, 256]` |

The first-epoch PSNR/SSIM values are pilot diagnostics only; they are not a
quality, accuracy, or deployment claim. The test split (133 pairs) and
challenging set (60 images) were not accessed. The frozen-manifest and website
checksum verification passed after the pilot.

Pilot artifacts:

- `training_runs/funie/funie_pilot_mps_20260930T155101Z/run_config.json`
- `training_runs/funie/funie_pilot_mps_20260930T155101Z/metrics.jsonl`
- `training_runs/funie/funie_pilot_mps_20260930T155101Z/checkpoints/latest.pt`
- `training_runs/funie/funie_pilot_mps_20260930T155101Z/validation_comparisons/epoch_001.png`

The torchvision `pretrained=` deprecation warning is caused by the pinned
official FUnIE-GAN source's legacy VGG19 constructor. It resolved to
`VGG19_Weights.IMAGENET1K_V1` and did not affect the successful run. No source
or dependency was changed as part of the pilot.

## Compatibility tests

### FUnIE-GAN

Command run:

```sh
.venv/bin/python preparation/run_funie_smoke.py \
  --steps 2 --image-size 128 --allow-cpu-compatibility
```

Result: passed on explicit CPU. Two batch-1 steps used only frozen UIEB train
pairs (`100_img_.png`, `10139.png`). Forward pass, finite GAN losses,
backward passes, optimizer updates, checkpoint save/reload, and finite
inference all passed. Elapsed time was 0.567 s; the process high-water RSS as
reported by macOS was 629,522,432 bytes.

The smoke checkpoint is
`smoke_tests/funie/funie_smoke_20260930T144811Z/funie_smoke_checkpoint.pt`
(SHA-256 `a7163164a4593175bfa435999806ad64412cef7ea5f2d29237f690d8d4d99d4e`).
It is randomly initialized; no pretrained enhancement checkpoint was used.

The official FUnIE-GAN PyTorch entry point is a legacy EUVP/UFO-120 loader and
does not directly consume this UIEB layout. The compatibility test imported
the official GeneratorFunieGAN and DiscriminatorFunieGAN architecture, with
adversarial plus L1 reconstruction loss. It intentionally did not test the
legacy VGG perceptual-loss path, so this is architecture/runtime compatibility,
not reproduction of the paper's full training recipe or a quality result.

### RF-DETR Small

Command run:

```sh
.venv/bin/python preparation/run_rfdetr_smoke.py \
  --allow-cpu-compatibility --resolution 64
```

Result: passed on explicit CPU. Before training, the installed RF-DETR source
was verified to expect the Roboflow COCO convention
`train/_annotations.coco.json` and `valid/_annotations.coco.json`. The temporary
adapter used symlinks only, with two group-disjoint train images and two
group-disjoint validation images, all from the frozen detector **training**
split. Each adapter split has one positive and one negative image; no project
validation or test image was read.

The official PyTorch Lightning training path completed forward/loss/backward/
optimizer work, saved and reloaded a checkpoint, and ran inference. Batch size
was 1. RF-DETR's tiny-dataset sampler repeated the two unique train images to
five observed optimizer steps, still below the 10-step ceiling. Inference at
threshold 0.99 returned zero detections; that has no accuracy meaning. Elapsed
time was 29.689 s; process high-water RSS was 2,269,151,232 bytes.

Final smoke artifacts:

- Result: `smoke_tests/rfdetr/rfdetr_smoke_20260930T145409Z/result.json`
- Reloaded checkpoint: `smoke_tests/rfdetr/rfdetr_smoke_20260930T145409Z/training_output/checkpoint_best_total.pth`
  (SHA-256 `2bc2d4b92dfc602a52b1e45bd34927b91e8220e7fac0152307a99cb3bef5efce`)
- Full recorded training configuration:
  `smoke_tests/rfdetr/rfdetr_smoke_20260930T145409Z/training_output/training_config.json`

The official RF-DETR Small pretrained COCO checkpoint was downloaded to
`.cache/rfdetr/rf-detr-small.pth`, passed the package MD5 verification
(`fb37061c1af7bace359c91b723a8d5c1`), and has local SHA-256
`d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33`.
Its source URL is
`https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth`.

The 64-pixel smoke resolution issued the repository's DINOv2 positional-
encoding/patch warning. It was chosen only to cap test cost and must not be
used as a full-training setting. The first temporary RF-DETR attempt is retained
only as a diagnostic record because its adapter used related frames across its
temporary train/valid folders; see its `INVALID_FOR_READINESS.md`. It is not
part of the readiness result.

## Proposed full-training starting point (not launched)

1. Completed: both bounded smoke tests have now run successfully with
   `device: mps` from native macOS Terminal. Do not use the earlier CPU runs as
   MPS performance estimates.
2. For FUnIE-GAN, start a separate, staged run with frozen UIEB train pairs
   only, 256-pixel crops, batch size 1, and frozen validation pairs for model
   selection. Keep 133 test pairs and the 60 challenging images untouched until
   one final evaluation.
3. For RF-DETR Small, start from the official default 512-pixel resolution,
   batch size 1, frozen one-class train/validation manifests, and the verified
   category mapping. Keep group-aware test images completely out of tuning.
   Increase batch size only after an MPS memory probe; retain negative images.
4. Use separate final output directories (not `smoke_tests/`), record the
   frozen manifest SHA, source revisions, package lock file, device, peak memory
   and checkpoint origin in every real run. Set a fixed seed and choose epoch
   count/early stopping only after validation curves exist.

These tiny smoke tests establish execution compatibility only. They make no
accuracy, robustness, or deployment claim.

## Scripts and repeatability

- `preparation/create_leakage_contact_sheets.py`
- `preparation/record_leakage_decisions.py`
- `preparation/prepare_validated_datasets.py`
- `preparation/freeze_final_manifests.py`
- `preparation/run_funie_smoke.py`
- `preparation/run_rfdetr_smoke.py`

Before a future run, use project-local cache settings such as:

```sh
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export RF_HOME="$PWD/.cache/rfdetr"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```
