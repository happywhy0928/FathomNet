# Consistency Regularization Run (λ = 0.05)

- **Config**: `configs/resnet50_threeheads_consistency.yaml`
- **Command**: `make train CONFIG=configs/resnet50_threeheads_consistency.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- **Logger version**: `logs/lightning/version_3`
- **Best checkpoint**: `fathomnet-hcls-epoch=14-val/macro_f1_species=0.326.ckpt`
- **Training notes**:
  - Loss curves similar to baseline; training loss reached ~1.25 by epoch 14.
  - No stability issues; KL penalty weight set to 0.05.

## Validation Metrics
| Metric                     | Baseline (ver_2) | Consistency (ver_3) |
|----------------------------|------------------|---------------------|
| Loss                       | 3.10             | 3.10                |
| Species top-1 accuracy     | 0.444            | 0.441               |
| Species macro-F1           | 0.331            | 0.326               |
| Genus top-1 accuracy       | 0.71             | 0.69                |
| Genus macro-F1             | 0.61             | 0.60                |
| Family top-1 accuracy      | 0.86             | 0.85                |
| Family macro-F1            | 0.74             | 0.73                |

**Conclusion**: λ=0.05 consistency loss keeps metrics roughly unchanged compared to the baseline. As next steps, we plan to (a) tune λ (0.01 / 0.1) to see if lighter or stronger regularization helps, and (b) explore parent-gate/attention modules if regularization alone remains ineffective.
