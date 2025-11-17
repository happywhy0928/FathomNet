# Command Log / Usage Guide

This file records the key shell commands run so far and serves as a quick how-to for reproducing the workflow.

```bash
# 1) Install project dependencies (ran outside a virtual env this time)
python3 -m pip install -r requirements.txt

# 2) Download all FathomNet imagery referenced in the COCO manifests
make download_real

# 3) Normalize metadata, build splits + label encoders
# (use PREP_ARGS=--skip-download if images already exist)
make prep_data

# 4) (Optional) Rebuild taxonomy map and rerun ingestion manually
python scripts/build_taxonomy_map.py
python -m src.data.kaggle_ingest --in_csv data/splits/all_from_coco.csv \
    --img_root data/kaggle/images --out_dir data/splits \
    --min_per_class 50 --seed 2025 --save-encoders

# 5) Run the Lightning training CLI (baseline)
# short smoke test (2 epochs, default)
make train
# longer baseline run (example: 15 epochs on limited batches)
make train TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60

# 6) Consistency regularization experiment
make train CONFIG=configs/resnet50_threeheads_consistency.yaml \\
          TRAIN_EPOCHS=15 LIMIT_TRAIN=200 LIMIT_VAL=60

# 7) ConvNeXt-Tiny + balanced sampling + mixup/cutmix (full run)
make train CONFIG=configs/convnext_threeheads_mixup.yaml \\
          TRAIN_EPOCHS=30 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0

# 8) ConvNeXt-Tiny + RandAugment + mixup/cutmix (planned stronger aug)
make train CONFIG=configs/convnext_threeheads_randaugment.yaml \\
          TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0

# 9) ConvNeXt-Tiny + staged fine-tuning (freeze 10 epochs, LR split)
make train CONFIG=configs/convnext_threeheads_staged.yaml \\
          TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0

# 10) ConvNeXt-Tiny + medium RandAugment + mixup/cutmix
make train CONFIG=configs/convnext_threeheads_randaugment_mid.yaml \\
          TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0

# 11) ConvNeXt-Small + medium RandAugment + mixup/cutmix
make train CONFIG=configs/convnext_small_randaugment.yaml \\
          TRAIN_EPOCHS=35 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0

# 12) ConvNeXt-Small + medium RandAugment + longer schedule
make train CONFIG=configs/convnext_small_randaugment_long.yaml \\
          TRAIN_EPOCHS=45 LIMIT_TRAIN=1.0 LIMIT_VAL=1.0
```

> Tip: For fresh setups, create/activate a virtual environment first, then run Step 1.
