# ConvNeXt-Tiny + Medium RandAugment + Mixup

- Config: `configs/convnext_threeheads_randaugment_mid.yaml`
- Motivation: retain the balanced sampler + mixup/cutmix recipe (version_21) but add a milder RandAugment (num_ops=1, magnitude=9) to gain diversity without the over-regularization seen at magnitude 12.
- Command:
  ```bash
  make train CONFIG=configs/convnext_threeheads_randaugment_mid.yaml \
            TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: pending.

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.40 by adding moderate invariances. |
| Genus   | Benefit from smoother species decisions; expect ≈0.41. |
| Family  | Stay near 0.49 (should not degrade like high-magnitude run). |

## Results (fill after run)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | _TBD_           | _TBD_          | _TBD_            |       |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Add observations after training (e.g., convergence speed, whether augmentations help rare classes).
