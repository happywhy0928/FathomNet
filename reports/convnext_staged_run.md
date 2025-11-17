# ConvNeXt-Tiny + Staged Fine-Tuning

- Config: `configs/convnext_threeheads_staged.yaml`
- Idea: keep balanced sampling + mixup/cutmix, but freeze the ConvNeXt backbone for the first 10 epochs so the heads adapt, then unfreeze with a smaller backbone LR (1e-4) for the remaining epochs. Goal is better stability on species while preserving genus/family gains.
- Command:
  ```bash
  make train CONFIG=configs/convnext_threeheads_staged.yaml \
            TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: completed (version_24).

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Avoid abrupt feature drift, target macro-F1 ≥0.40. |
| Genus   | Should match or exceed 0.41 due to more stable backbone. |
| Family  | Remains around ≥0.49; slower LR should prevent over-regularization. |

## Results (45 epochs, version_24)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.470 (peak 0.487) | 0.381 (peak 0.388) | 0.389 | Best ckpt: `logs/lightning/version_24/checkpoints/fathomnet-hcls-epoch=24-val/macro_f1_species=0.389.ckpt`, species top-1 ≈0.515 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- Species macro-F1 plateaued at ≈0.389, not beating the mixup baseline (0.395). Freezing the backbone delayed convergence but did not yield higher accuracy.
- Genus peaked at ≈0.388, family at ≈0.487 — both lower than RandAugment’s family boost or mixup run.
- Suggest reverting to mixup baseline as the “best so far” and focus next iteration on medium-strength augmentations or alternate backbones.
