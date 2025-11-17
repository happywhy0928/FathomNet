# Consistency Weight Tuning (λ = 0.01 vs. 0.1)

## Run 1: λ = 0.01 (logger version_4)
- Config: `configs/resnet50_threeheads_consistency.yaml`
- Command: `make train CONFIG=...consistency.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Best checkpoint: `logs/lightning/version_4/checkpoints/fathomnet-hcls-epoch=09-val/macro_f1_species=0.325.ckpt`
- Metrics (val):
  - Species: loss 3.05, top-1 0.428, macro-F1 0.325
  - Genus: top-1 ≈0.69, macro-F1 ≈0.59
  - Family: top-1 ≈0.84, macro-F1 ≈0.72
- Observation: Slightly worse than baseline (macro-F1 0.331) and similar to λ=0.05.

## Run 2: λ = 0.1 (logger version_5)
- Config: `configs/resnet50_threeheads_consistency_high.yaml`
- Command: `make train CONFIG=...consistency_high.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Best checkpoint: `logs/lightning/version_5/checkpoints/fathomnet-hcls-epoch=09-val/macro_f1_species=0.321.ckpt`
- Metrics (val):
  - Species: loss 2.92, top-1 0.429, macro-F1 0.321
  - Genus: top-1 ≈0.70, macro-F1 ≈0.59
  - Family: top-1 ≈0.84, macro-F1 ≈0.72
- Observation: Stronger penalty also fails to improve over baseline; species macro-F1 drops slightly to 0.321. Consistency alone is insufficient—move on to parent-gate/attention ideas.
