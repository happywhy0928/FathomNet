# Class-Balanced Species Loss

- **Config**: `configs/resnet50_threeheads_classbalanced.yaml` (inverse-frequency weights for species head)
- **Command**: `make train CONFIG=configs/resnet50_threeheads_classbalanced.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- **Logger version**: `logs/lightning/version_11`
- **Best checkpoint**: `fathomnet-hcls-epoch=09-val/macro_f1_species=0.329.ckpt`

## Validation Metrics
| Metric                  | Baseline (ver_2) | Class-balanced (ver_11) |
|-------------------------|------------------|-------------------------|
| Loss                    | 3.10             | 3.55                    |
| Species top-1 accuracy  | 0.444            | 0.428                   |
| Species macro-F1        | 0.331            | 0.318                   |
| Genus top-1 accuracy    | 0.71             | 0.437                   |
| Genus macro-F1          | 0.61             | 0.328                   |
| Family top-1 accuracy   | 0.86             | 0.536                   |
| Family macro-F1         | 0.74             | 0.416                   |

**Observation**: Applying inverse-frequency species weights does not improve performance; macro-F1 even drops slightly (0.318). The species loss becomes dominated by rare classes, hurting overall accuracy. This suggests that more sophisticated balancing (e.g., focal loss or class-aware sampling) is needed.
