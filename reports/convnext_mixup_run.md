# ConvNeXt-Tiny + Balanced Sampler + Mixup/Cutmix

- Config: `configs/convnext_threeheads_mixup.yaml`
- Goal: retain the balanced sampler (helps rare species) while adding mixup/cutmix regularization to boost macro-F1 across family/genus/species.
- Command (full run, record in `docs/COMMAND_LOG.md`):  
  ```bash
  make train CONFIG=configs/convnext_threeheads_mixup.yaml \
            TRAIN_EPOCHS=30 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
  ```
- Smoke test: `version_20` (2 epochs) confirmed the code path works; val macro-F1 (species) ≈0.259 so results are not representative.

## Expected impact

| Level   | Rationale                                                                                     |
|---------|------------------------------------------------------------------------------------------------|
| Species | Mixup/cutmix should reduce overfitting on small classes → higher macro-F1.                     |
| Genus   | Softer decision boundaries should transfer improvements upward; expect +0.01–0.02 macro-F1.   |
| Family  | Benefit mainly through balanced sampler + mixup smoothing; expect stability vs. baseline.     |

## Results (30 epochs, version_21)

| Split | Macro-F1 family | Macro-F1 genus | Macro-F1 species | Notes |
|-------|-----------------|----------------|------------------|-------|
| Val   | 0.480           | 0.390          | 0.395            | Best ckpt: `logs/lightning/version_21/checkpoints/fathomnet-hcls-epoch=21-val/macro_f1_species=0.399.ckpt`, species top-1 ≈0.53 |
| Test  | _TBD_           | _TBD_          | _TBD_            |       |

Observations:
- Mixup + cutmix + balanced sampler lifted species macro-F1 from 0.369 (previous ConvNeXt baseline) to ≈0.395 (+0.026 absolute), and genus/family also improved to ≈0.39/0.48.
- Training remained stable on MPS (loss curve smooth, no NaNs). Slightly higher val loss than balanced run because mixup soft targets slow convergence but deliver better macro metrics.
- Next ideas: stronger color/RandAug augmentations or staged fine-tuning (freeze first 10 epochs, unfreeze later) to push species >0.42.
