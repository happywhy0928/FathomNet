# Automatic Mixed Precision (AMP) Experiment

- Config: `configs/resnet50_threeheads_amp.yaml` (`training.use_amp = true`)
- Description: run the baseline model with mixed precision to allow higher throughput and explore whether longer training (more epochs under same time) or better numerics improve macro-F1.
- Command: `make train CONFIG=configs/resnet50_threeheads_amp.yaml TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Status: pending execution.

After running, record metrics (species/genus/family) and compare to baseline in this file and in `reports/experiments.md`.
