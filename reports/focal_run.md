# Focal Loss Experiment (γ = 1.5)

- Config: `configs/resnet50_threeheads_focal.yaml`
- Command: `make train CONFIG=configs/resnet50_threeheads_focal.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Logger version: `logs/lightning/version_16`
- Best checkpoint: `fathomnet-hcls-epoch=13-val/macro_f1_species=0.331.ckpt`

## Validation Metrics
| Metric                  | Baseline (ver_2) | Focal Loss (ver_16) |
|-------------------------|------------------|---------------------|
| Loss                    | 3.10             | 3.38                |
| Species top-1 accuracy  | 0.444            | 0.444               |
| Species macro-F1        | 0.331            | 0.330               |
| Genus top-1 accuracy    | 0.71             | 0.448               |
| Genus macro-F1          | 0.61             | 0.342               |
| Family top-1 accuracy   | 0.86             | 0.540               |
| Family macro-F1         | 0.74             | 0.413               |

**Observation**: Focal loss (γ=1.5) essentially matches the baseline (species Macro-F1 ≈ 0.33). While it slightly rebalances class emphasis, it does not yield a measurable improvement under the current training schedule. Further gains will likely require a stronger backbone or more targeted sampling/augmentation strategies.
