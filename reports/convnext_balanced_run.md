# ConvNeXt-Tiny + Balanced Sampling (planned)

- Config: `configs/convnext_threeheads_balanced.yaml`
- Description: apply class-balanced sampling (inverse-frequency) when drawing training batches to boost rare species.
- Command: `make train CONFIG=configs/convnext_threeheads_balanced.yaml TRAIN_EPOCHS=30 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Status: pending execution.
