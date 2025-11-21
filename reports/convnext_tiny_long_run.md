# ConvNeXt-Tiny + Medium RandAugment (Longer Schedule)

- Config: `configs/convnext_threeheads_randaugment_mid_long.yaml`
- Motivation: our best run so far (`version_25`) used ConvNeXt-Tiny + RandAugment (n=1, m=9) for 35 epochs with lr=3e-4. This experiment extends the same recipe to 45 epochs and slightly lowers the LR (2.5e-4) to see if species macro-F1 can exceed 0.42 without changing backbone size.
- Command:
  ```bash
  make train CONFIG=configs/convnext_threeheads_randaugment_mid_long.yaml \
            TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: pending.

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.42 (beyond 0.414 from version_25). |
| Genus   | ≥0.42 due to more stable features with longer schedule. |
| Family  | ≈0.50 since balanced sampler + mixup already helps rare classes. |

## Results (fill after run)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | _TBD_           | _TBD_          | _TBD_            |       |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Add observations once complete (training time, whether extra epochs help or hurt).
