# Experiment Tracker

| Version (lightning logger) | Config / Command                                    | Notes / Metrics (summary)                     |
|----------------------------|------------------------------------------------------|-----------------------------------------------|
| version_0                  | `make train` (initial smoke test, 2 epochs)         | sanity check only                             |
| version_1                  | `make train TRAIN_EPOCHS=15 ...` (baseline run)     | baseline metrics logged in `baseline_run.md`  |
| version_2                  | `make train TRAIN_EPOCHS=15 ...` (baseline repeat)  | best ckpt: `val/macro_f1_species≈0.331`       |
| version_3                  | `make train CONFIG=...consistency.yaml ...`        | hierarchy consistency `λ=0.05`, val F1≈0.326  |
| version_4                  | `make train CONFIG=...consistency.yaml ...`        | λ=0.01; val macro-F1≈0.325 (no gain)          |
| version_5                  | `make train CONFIG=...consistency_high.yaml ...`   | λ=0.1; val macro-F1≈0.321 (still no gain)     |
| version_9                  | `make train CONFIG=...parentgate.yaml ...`         | Parent-gate (concat), val Macro-F1≈0.331      |
| version_10                 | `make train CONFIG=...parentgate.yaml ...`         | Parent-gate (MLP gating), val Macro-F1≈0.309  |
| version_11                 | `make train CONFIG=...classbalanced.yaml ...`     | Class-balanced species loss, val F1≈0.318     |
| version_12                 | `make train CONFIG=...focal.yaml ...`             | Focal loss γ=1.5, val Macro-F1≈0.330          |
| version_13                 | `make train CONFIG=configs/vit_threeheads.yaml ...` | ViT-B/16 backbone, val Macro-F1≈0.213       |
| version_14                 | `make train CONFIG=configs/convnext_threeheads.yaml ...` | ConvNeXt-Tiny, val Macro-F1≈0.361    |
| version_15                 | `make train CONFIG=configs/convnext_threeheads_focal.yaml ...` | ConvNeXt + Focal (γ=1.5), val F1≈0.361 |
| version_16 *(planned)*     | `make train CONFIG=configs/convnext_threeheads_balanced.yaml ...` | ConvNeXt + balanced sampling       |
| version_18                 | `make train CONFIG=configs/convnext_threeheads_balanced.yaml ...` | Balanced sampler run, val macro-F1≈0.369 (family≈0.475, genus≈0.375) |
| version_20 (smoke)         | `make train CONFIG=configs/convnext_threeheads_mixup.yaml --epochs 2` | Mixup/cutmix code-path check, val F1≈0.259 |
| version_21                 | `make train CONFIG=configs/convnext_threeheads_mixup.yaml TRAIN_EPOCHS=30 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0` | Mixup+cutmix full run, val macro-F1≈0.395 (family≈0.48, genus≈0.39), top-1≈0.53 |
| version_22                 | `make train CONFIG=configs/convnext_threeheads_randaugment.yaml ...` | RandAugment + mixup (m=12, n=2): val macro-F1 species≈0.387 (peak 0.396), genus≈0.408, family≈0.492 |
| version_24                 | `make train CONFIG=configs/convnext_threeheads_staged.yaml ...` | Staged fine-tuning (freeze 10 epochs, backbone lr=1e-4): val macro-F1 species≈0.389, genus≈0.381, family≈0.470 |
| version_25                 | `make train CONFIG=configs/convnext_threeheads_randaugment_mid.yaml ...` | Medium RandAugment (n=1, m=9): val macro-F1 species≈0.414, genus≈0.410, family≈0.498 |
| version_26                 | `make train CONFIG=configs/convnext_small_randaugment.yaml ...` | ConvNeXt-Small + medium RandAugment: val macro-F1 species≈0.402, genus≈0.405, family≈0.513 |
| version_27 *(planned)*     | `make train CONFIG=configs/convnext_small_randaugment_long.yaml ...` | ConvNeXt-Small longer schedule (45 epochs, lr=1.5e-4) to chase species macro-F1 ≥0.42 |

> Update this table each time you run a new experiment, linking to the relevant report file.
