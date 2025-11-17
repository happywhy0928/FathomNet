# ConvNeXt-Tiny + Medium RandAugment + Mixup

- Config: `configs/convnext_threeheads_randaugment_mid.yaml`
- Motivation: retain the balanced sampler + mixup/cutmix recipe (version_21) but add a milder RandAugment (num_ops=1, magnitude=9) to gain diversity without the over-regularization seen at magnitude 12.
- Command:
  ```bash
  make train CONFIG=configs/convnext_threeheads_randaugment_mid.yaml \
            TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: completed (version_25).

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.40 by adding moderate invariances. |
| Genus   | Benefit from smoother species decisions; expect ≈0.41. |
| Family  | Stay near 0.49 (should not degrade like high-magnitude run). |

## Results (35 epochs, version_25)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.498 (peak 0.500) | 0.410 (peak 0.413) | 0.414 | Best ckpt: `logs/lightning/version_25/checkpoints/fathomnet-hcls-epoch=25-val/macro_f1_species=0.414.ckpt`, species top-1 ≈0.536 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- Species macro-F1 reached 0.414 (+0.019 vs. mixup-only baseline) while genus≈0.41 and family≈0.50, our best overall metrics so far.
- Training remained stable; moderate RandAugment seems to improve rare species without the drop seen at magnitude 12.
- Next: consider running this config longer (45 epochs) or scaling backbone (ConvNeXt-S) using the same augmentation recipe.
