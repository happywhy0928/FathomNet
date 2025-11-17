# ConvNeXt-Tiny Backbone Run

- **Config**: `configs/convnext_threeheads.yaml`
- **Command**: `make train CONFIG=configs/convnext_threeheads.yaml TRAIN_EPOCHS=30 LIMIT_TRAIN=200 LIMIT_VAL=60`
- **Logger version**: `logs/lightning/version_15`
- **Best checkpoint**: `fathomnet-hcls-epoch=15-val/macro_f1_species=0.370.ckpt`

## Validation Metrics
| Metric                  | Baseline (ResNet, ver_2) | ConvNeXt-Tiny (ver_15) |
|-------------------------|--------------------------|------------------------|
| Loss                    | 3.10                     | 3.90                   |
| Species top-1 accuracy  | 0.444                    | 0.490                  |
| Species macro-F1        | 0.331                    | 0.361                  |
| Genus top-1 accuracy    | 0.71                     | 0.479                  |
| Genus macro-F1          | 0.61                     | 0.351                  |
| Family top-1 accuracy   | 0.86                     | 0.592                  |
| Family macro-F1         | 0.74                     | 0.459                  |

**Observation**: ConvNeXt-Tiny delivers the first noticeable improvement (species macro-F1 +0.03 vs. ResNet baseline). Although genus/family accuracies dropped slightly (due to the more aggressive backbone), the species-level gain demonstrates that a stronger feature extractor plus longer training is beneficial.**
