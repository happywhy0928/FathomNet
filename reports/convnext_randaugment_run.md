# ConvNeXt-Tiny + RandAugment + Mixup

- Config: `configs/convnext_threeheads_randaugment.yaml`
- Motivation: strong appearance augmentations (RandAugment with magnitude 12) stacked on top of balanced sampling and mixup/cutmix should further generalize rare classes.
- Command (full run):
  ```bash
  make train CONFIG=configs/convnext_threeheads_randaugment.yaml \
            TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: pending (not yet executed).

## Hypothesis

| Level   | Expected Impact |
|---------|-----------------|
| Species | Extra augmentations should push macro-F1 beyond 0.40 by exposing the model to diverse colors/shapes. |
| Genus   | Benefits from improved species invariance; target +0.02 macro-F1 compared to mixup-only run. |
| Family  | Should stay ≥0.48, possibly reach 0.50 with better generalized features. |

## Results (35 epochs, version_22)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.492           | 0.408          | 0.387            | Best ckpt: `logs/lightning/version_22/checkpoints/fathomnet-hcls-epoch=22-val/macro_f1_species=0.396.ckpt`, species top-1 ≈0.53 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- Genus/family macro-F1 improved slightly vs. mixup-only run (0.39→0.41 genus, 0.48→0.49 family), but species macro-F1 dipped to 0.387 (peak 0.396) indicating RandAugment may be too aggressive.
- Training loss plateaued around 3.0 with occasional spikes, suggesting these augmentations introduce heavy color/shape shifts. Consider reducing magnitude to 9 or limiting ops to color-only transforms.
- Follow-up: try a medium RandAugment setting (num_ops=1, magnitude=9) **or** staged fine-tuning (freeze 10 epochs, then unfreeze) to push species macro-F1 above 0.40.
