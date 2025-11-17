# ConvNeXt-Small + Longer Training

- Config: `configs/convnext_small_randaugment_long.yaml`
- Idea: keep balanced sampler + mixup/cutmix + RandAugment(n=1, m=9) but extend ConvNeXt-S training to 45 epochs and lower LR to 1.5e-4 to see if larger backbone benefits from extra convergence time.
- Command:
  ```bash
  make train CONFIG=configs/convnext_small_randaugment_long.yaml \
            TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: pending.

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.42 (surpass ConvNeXt-Tiny mid run 0.414). |
| Genus   | ≥0.41 due to more stable features over longer schedule. |
| Family  | Around 0.52–0.53 leveraging larger backbone capacity. |

## Results (fill after run)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | _TBD_           | _TBD_          | _TBD_            |       |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Document training time, stability, and whether the longer schedule helps species F1.
