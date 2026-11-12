# RF-DETR Small raw-image scale-jitter ablation: epochs 1–10

`preparation/run_rfdetr_raw_v2_scale_jitter_off.py` prepares a fresh,
ten-epoch RF-DETR Small experiment. It starts from the same official,
MD5-verified `rf-detr-small.pth` used by the original raw-image run; it does
not resume from any epoch-5 or epoch-10 AquaVision checkpoint.

The frozen raw TrashCan v2 train/validation splits, negative images, one-class
`visible_plastic_debris` mapping, MPS device, batch size 1, 512-pixel fixed
resolution, `multi_scale="off"`, AdamW, learning rates, step scheduler, EMA,
seed, and pinned RF-DETR source are retained. CPU fallback is prohibited.
Test images, enhanced images, the website, and source datasets are absent from
the adapter or untouched.

## Controlled change

The sole training-behaviour change is `scale_jitter=True` to
`scale_jitter=False`. In this pinned RF-DETR code, that removes the stochastic
resize-crop-resize branch as well as its scale changes. It tests their combined
effect; it does not separately establish the effect of cropping or scaling.

Before training, the runner writes
`configuration_audit_before_training.json`, which enumerates every intended
difference from the preserved ten-epoch raw-image configuration. The allowed
non-behavioural differences are the new adapter/output paths, `resume=None`
for the intentional fresh pretrained start, and explanatory notes. It refuses
to run if any model configuration field or unapproved training field differs.
It writes a second audit after RF-DETR resolves its actual configuration.

## Outputs after a successful run

- EMA validation metrics, losses, and learning rates for every epoch 1–10;
- MPS time and memory telemetry for every epoch;
- a full-state checkpoint for each completed epoch, plus run-local
  `latest_full_state.ckpt` and best-AP EMA aliases;
- the same preselected nine-image qualitative panel used by the original
  expanded preview plan, regenerated for each EMA epoch;
- original-best, original-epoch-10, ablation-best, and ablation-epoch-10
  per-size AP/AR evaluations using the proven antialiased validation path and
  pinned `vernier` evaluator;
- precision, recall, and F1 at the common threshold `0.43`, alongside clearly
  labeled per-checkpoint optimized-threshold results; and
- fixed comparison fields for original best AP `0.1212069318` and original
  epoch-10 AP `0.1105767041`.

The reused qualitative panel contains frozen train/validation examples only;
it is not a substitute for the all-1,082-image validation metrics.

## macOS Terminal command

```bash
cd '/Users/nanditareddy/AquaVision AI' && export TORCH_HOME="$PWD/.cache/torch" HF_HOME="$PWD/.cache/huggingface" RF_HOME="$PWD/.cache/rfdetr" XDG_CACHE_HOME="$PWD/.cache/xdg" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=0 && caffeinate -i .venv/bin/python preparation/run_rfdetr_raw_v2_scale_jitter_off.py --run-id "rfdetr_raw_v2_scale_jitter_off_e1to10_mps_$(date -u +%Y%m%dT%H%M%SZ)" --epochs 10 --resolution 512 --batch-size 1 --seed 20260930 --preview-count 9
```

The runner stops after epoch 10. It does not initiate a later experiment.
