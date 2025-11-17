# Parent-Gate Experiment (learnable gating)

- **Config**: `configs/resnet50_threeheads_parentgate.yaml` (`parent_gate=true`, `parent_gate_type=mlp`)
- **Command**: `make train CONFIG=configs/resnet50_threeheads_parentgate.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- **Logger version**: `logs/lightning/version_10`
- **Best checkpoint**: `fathomnet-hcls-epoch=11-val/macro_f1_species=0.319.ckpt`

## Validation Metrics
| Metric                  | Baseline (ver_2) | Parent-gate (ver_10) |
|-------------------------|------------------|----------------------|
| Loss                    | 3.10             | 3.46                 |
| Species top-1 accuracy  | 0.444            | 0.422                |
| Species macro-F1        | 0.331            | 0.309                |
| Genus top-1 accuracy    | 0.71             | 0.438                |
| Genus macro-F1          | 0.61             | 0.329                |
| Family top-1 accuracy   | 0.86             | 0.533                |
| Family macro-F1         | 0.74             | 0.410                |

**Observation**: The learnable parent gate significantly underperforms the baseline (macro-F1 0.309). The additional gating MLP appears to distort the representations without providing helpful structure. This indicates that simply gating backbone features by parent probabilities is not effective; future variants should explore attention mechanisms or temperature scaling with careful normalization.
