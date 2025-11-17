# Baseline Run (ResNet50 + Three Heads)

- Config: `configs/resnet50_threeheads.yaml`
- Command: `make train TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60`
- Dataset: 8,600 images (69 species), splits 6,880 / 860 / 860
- Label coverage: genus 64 labels, family 34 labels (81% from WoRMS)

## Key Metrics (best checkpoint: `logs/lightning/version_2/checkpoints/fathomnet-hcls-epoch=13-val/macro_f1_species=0.331.ckpt`)

| Metric (val)            | Value |
|-------------------------|-------|
| Loss                    | 3.10  |
| Species top-1 accuracy  | 0.44  |
| Species macro-F1        | 0.331 |
| Genus top-1 accuracy    | ~0.71 |
| Genus macro-F1          | ~0.61 |
| Family top-1 accuracy   | ~0.86 (valid subset) |
| Family macro-F1         | ~0.74 |
| Epochs                  | 15    |

Notes:
- Training loss dropped to ~1.27 by epoch 14; species top-1 rose to ~0.72 in training.
- Validation metrics stabilized around epoch 13; the checkpoint above is used as baseline reference.
- MacOS MPS accelerator with pin_memory warning (expected, harmless).

Next steps:
1. Implement hierarchy-consistency / parent-gate regularization.
2. Re-run training with identical settings and compare against this baseline.
3. Track results here for each experiment stage.
