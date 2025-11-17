"""Lightning training CLI for hierarchy-consistent models."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.dm import HierarchyDataModule
from .lightning_module import HierarchyLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-head classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--limit-train",
        type=float,
        default=None,
        help="Train batches (int for count, 0-1 float for fraction; default from config)",
    )
    parser.add_argument(
        "--limit-val",
        type=float,
        default=None,
        help="Val batches (int for count, 0-1 float for fraction; default from config)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = yaml.safe_load(config_path.read_text())

    seed_everything(cfg.get("seed", 2025), workers=True)

    paths = cfg.get("paths", {})
    splits_dir = Path(paths.get("splits_dir", "data/splits"))
    if not splits_dir.is_absolute():
        splits_dir = repo_root / splits_dir

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    max_epochs = args.epochs or train_cfg.get("epochs", 30)
    datamodule = HierarchyDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        img_size=train_cfg.get("img_size", 224),
        batch_size=train_cfg.get("batch_size", 64),
        num_workers=train_cfg.get("num_workers", 8),
        sample_mode=train_cfg.get("sample_mode"),
        train_augment=train_cfg.get("augment"),
    )

    backbone_lr = train_cfg.get("backbone_lr")
    backbone_lr = float(backbone_lr) if backbone_lr is not None else None
    model = HierarchyLightningModule(
        backbone=model_cfg.get("backbone", "resnet50"),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        freeze_backbone_epochs=train_cfg.get("freeze_backbone_epochs", 0),
        lr=float(train_cfg.get("lr", 3e-4)),
        backbone_lr=backbone_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        max_epochs=max_epochs,
        w_family=loss_cfg.get("w_family", 0.2),
        w_genus=loss_cfg.get("w_genus", 0.3),
        w_species=loss_cfg.get("w_species", 1.0),
        lambda_consistency=loss_cfg.get("lambda_consistency", 0.0),
        hierarchy_csv=train_csv,
        parent_gate=model_cfg.get("parent_gate", False),
        parent_gate_type=model_cfg.get("parent_gate_type"),
        class_weight_mode=model_cfg.get("class_weight_mode"),
        use_mixup=train_cfg.get("use_mixup", False),
        mixup_alpha=float(train_cfg.get("mixup_alpha", 0.4)),
        cutmix_alpha=float(train_cfg.get("cutmix_alpha", 0.0)),
        mixup_prob=float(train_cfg.get("mixup_prob", 1.0)),
    )

    logs_dir = Path(paths.get("logs_dir", "logs"))
    if not logs_dir.is_absolute():
        logs_dir = repo_root / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/macro_f1_species",
        mode="max",
        filename="fathomnet-hcls-{epoch:02d}-{val/macro_f1_species:.3f}",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=str(logs_dir), name="lightning")

    limit_train = train_cfg.get("limit_train", 1.0)
    limit_val = train_cfg.get("limit_val", 1.0)
    trainer = Trainer(
        accelerator="auto",  # CPU, CUDA, or MPS (Apple Silicon)
        precision="16-mixed" if train_cfg.get("use_amp", False) else "32",
        max_epochs=max_epochs,
        limit_train_batches=limit_train if args.limit_train is None else args.limit_train,
        limit_val_batches=limit_val if args.limit_val is None else args.limit_val,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=5,
    )

    trainer.fit(model, datamodule=datamodule)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
