# ConvNeXt-Small + Medium RandAugment + Mixup

- Config: `configs/convnext_small_randaugment.yaml`
- Motivation: scale up from ConvNeXt-Tiny to ConvNeXt-Small while keeping the proven recipe (balanced sampler + mixup/cutmix + RandAugment n=1, m=9) to test if higher capacity yields >0.42 macro-F1.
- Command:
  ```bash
  make train CONFIG=configs/convnext_small_randaugment.yaml \
            TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: completed (version_26).

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.42; larger backbone should capture richer textures. |
| Genus   | Expect ≥0.42 thanks to improved species features. |
| Family  | Stay ≈0.50 or higher due to better high-level semantics. |

## Results (35 epochs, version_26)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.513 (peak 0.521) | 0.405 (peak 0.409) | 0.402 | Best ckpt: `logs/lightning/version_26/checkpoints/fathomnet-hcls-epoch=22-val/macro_f1_species=0.402.ckpt`, species top-1 ≈0.538 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- ConvNeXt-S boosted family macro-F1 to ≈0.51 (new high) but species macro-F1 only reached 0.402, slightly below the ConvNeXt-Tiny + RandAugment mid run (0.414). Genus stayed around 0.405.
- Training took ~32 minutes total on M4 Pro; stable but no clear species gain. Might require longer training (45 epochs) or tuned LR to fully exploit larger capacity.
- Next: either extend epochs with cosine warm restarts or sweep LR/weight decay to push species F1 beyond 0.42.
