# ConvNeXt-Small + Longer Training

- Config: `configs/convnext_small_randaugment_long.yaml`
- Idea: keep balanced sampler + mixup/cutmix + RandAugment(n=1, m=9) but extend ConvNeXt-S to 45 epochs with a lower LR (1.5e-4) to see if the larger backbone benefits from extra convergence time.
- Command:
  ```bash
  make train CONFIG=configs/convnext_small_randaugment_long.yaml \
            TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Status: completed (version_27).

## Expectations

| Level   | Expected Impact |
|---------|-----------------|
| Species | Target macro-F1 ≥0.42 (surpass ConvNeXt-Tiny mid run 0.414). |
| Genus   | ≥0.41 due to more stable features. |
| Family  | ≈0.52 leveraging ConvNeXt-S capacity. |

## Results (45 epochs, version_27)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.485 (peak 0.492) | 0.391 (peak 0.399) | 0.387 | Best ckpt: `logs/lightning/version_27/checkpoints/fathomnet-hcls-epoch=25-val/macro_f1_species=0.387.ckpt`, species top-1 ≈0.528 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- Longer training with a lower LR did not improve species performance (0.387 vs. 0.402 from the 35-epoch run). Family/genus also dropped slightly compared to version_26.
- Training time increased (~40 minutes) but appeared stable; likely hitting capacity/regularization limits rather than insufficient epochs.
- Conclusion: ConvNeXt-Tiny with medium RandAugment remains best (species 0.414). Further ConvNeXt-S gains might require architectural changes or fine-grained LR schedules instead of simply training longer.
