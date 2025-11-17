# ViT-B/16 Backbone Experiment (planned)

- Config: `configs/vit_threeheads.yaml` (backbone = `vit_base_patch16_224`, training epochs = 30, batch_size = 32).
- Command: `make train CONFIG=configs/vit_threeheads.yaml TRAIN_EPOCHS=30 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0`
- Rationale: use a stronger transformer backbone and longer training schedule to see if macro-F1 improves beyond 0.33.
- Status: pending execution.

After running, record metrics + logger version here and in `reports/experiments.md`.
