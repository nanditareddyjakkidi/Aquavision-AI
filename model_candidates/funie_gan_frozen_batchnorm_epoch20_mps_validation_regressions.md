# Epoch-20 frozen-BatchNorm FUnIE-GAN validation regressions

The comparison uses all 134 frozen UIEB v2 validation pairs and the identical
raw-input metric preprocessing. These are relative-to-raw regressions, not
test-set results.

- PSNR: 24 worsened, 110 improved;
  mean delta +3.0150 dB and median
  +3.2767 dB.
- SSIM: 32 worsened, 102 improved;
  mean delta +0.09845 and median
  +0.10510.
- 35 images regressed on at least one metric. Their exact
  original/raw/output/reference metrics remain in
  `training_runs/funie/funie_batchnorm_frozen_e2to4_mps_20260930/comparison_against_original/per_image_metrics_epoch_020.csv`.

| Filename | Δ PSNR vs raw | Δ SSIM vs raw |
| --- | ---: | ---: |
| 725_img_.png | -11.9439 | -0.07458 |
| 809_img_.png | -10.0307 | -0.06625 |
| 832_img_.png | -7.1690 | -0.11684 |
| 727_img_.png | -6.6779 | -0.11240 |
| 29_img_.png | -6.0019 | -0.11008 |
| 920_img_.png | -5.5429 | -0.07792 |
| 569_img_.png | -3.8229 | -0.05047 |
| 738_img_.png | -2.3007 | -0.10497 |
| 763_img_.png | -1.9374 | -0.13762 |
| 732_img_.png | -2.3509 | -0.08372 |
| 721_img_.png | -2.1260 | -0.06264 |
| 767_img_.png | -2.6017 | -0.00194 |
| 5554.png | -1.8126 | -0.06985 |
| 687_img_.png | -2.7970 | +0.03062 |
| 741_img_.png | -1.2447 | -0.07665 |
| 48_img_.png | -1.5457 | -0.03658 |
| 450_img_.png | -1.3850 | -0.00202 |
| 369_img_.png | -0.9569 | -0.03896 |
| 810_img_.png | -1.7967 | +0.05048 |
| 60_img_.png | -0.1810 | -0.08101 |
| 79_img_.png | -0.8333 | -0.00290 |
| 751_img_.png | +0.4248 | -0.08214 |
| 370_img_.png | -0.0190 | -0.03070 |
| 820_img_.png | -0.1412 | -0.00452 |
| 292_img_.png | -0.0156 | +0.01306 |
| 681_img_.png | +0.6373 | -0.02897 |
| 573_img_.png | +0.8807 | -0.02227 |
| 259_img_.png | +0.7969 | -0.00671 |
| 193_img_.png | +1.0129 | -0.01916 |
| 101_img_.png | +1.3785 | -0.02582 |
| 680_img_.png | +1.2703 | -0.01230 |
| 610_img_.png | +1.5196 | -0.03721 |
| 726_img_.png | +1.7854 | -0.01327 |
| 394_img_.png | +2.8575 | -0.04409 |
| 1573.png | +3.5957 | -0.03969 |
