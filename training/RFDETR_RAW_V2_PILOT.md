# RF-DETR Small raw-image v2 pilot

`preparation/run_rfdetr_raw_v2_pilot.py` is a one-epoch, MPS-only baseline
pilot. It uses the frozen TrashCan v2 derived splits solely through a separate
COCO adapter of symlinks to the original `material_version` raw images:

| Split | Images | Positive images | Negative images | Plastic instances |
| --- | ---: | ---: | ---: | ---: |
| Train | 5,048 | 1,242 | 3,806 | 1,299 |
| Validation | 1,082 | 249 | 833 | 289 |

The adapter contains no `test/` directory. Every selected source image is
checksum-checked against the frozen manifest and its decoded dimensions are
verified before training. RF-DETR receives the one derived COCO category,
`visible_plastic_debris` (COCO ID 1, contiguous RF-DETR foreground label 0).
Negative images are retained. No enhancement output is used as detector input.

Training uses fixed 512-pixel square tensors: the runner explicitly sets
`multi_scale="off"` and `expanded_scales=False`, preventing the pinned
RF-DETR package's per-batch multi-scale default from changing the tensor
resolution.

The runner verifies the pinned project-local RF-DETR source, the official
COCO-pretrained Small weights using the upstream MD5, the installed source's
COCO layout, the MPS-only configuration, and the resolved training config. It
then writes one-epoch losses, validation AP@[.50:.95]/AP50/AP75/mAR/F1,
per-size values if the installed callback emits them, timing and memory data,
checkpoint reload/inference results, and labeled raw-validation previews under
its own `training_runs/rfdetr/<run-id>/` directory.

Run it only in native macOS Terminal:

```sh
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && .venv/bin/python preparation/run_rfdetr_raw_v2_pilot.py --run-id "rfdetr_raw_v2_pilot_mps_$(date -u +%Y%m%dT%H%M%SZ)" --epochs 1 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 5
```

The runner refuses a missing MPS device, any CPU fallback, altered frozen files,
or an existing run directory. It runs exactly one epoch and does not touch UIEB
test/challenging data, TrashCan test data, website files, or RF-DETR inputs
outside this raw-image baseline.
