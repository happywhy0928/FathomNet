# ConvNeXt + Focal Loss (γ = 1.5)

- **Config**: `configs/convnext_threeheads_focal.yaml`
- **Command**: `make train CONFIG=configs/convnext_threeheads_focal.yaml TRAIN_EPOCHS=30 LIMIT_TRAIN=200 LIMIT_VAL=60`
- **Logger version**: `logs/lightning/version_16`
- **Best checkpoint**: `fathomnet-hcls-epoch=15-val/macro_f1_species=0.370.ckpt`

## Validation Metrics
| Metric                  | ConvNeXt baseline (ver_14) | ConvNeXt + Focal (ver_16) |
|-------------------------|----------------------------|---------------------------|
| Loss                    | 3.90                       | 3.90                      |
| Species top-1 accuracy  | 0.49                       | 0.49                      |
| Species macro-F1        | 0.361                      | 0.361                     |
| Genus macro-F1          | 0.351                      | 0.351                     |
| Family macro-F1         | 0.459                      | 0.459                     |

**Observation**: Adding focal loss to ConvNeXt does not change the metrics; species Macro-F1 stays at ~0.36. With a stronger backbone, class reweighting seems unnecessary—further improvements likely need better sampling/augmentation rather than heavier losses.
