"""Lightning module with ResNet backbone and three classification heads."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import timm
import torch
from lightning.pytorch import LightningModule
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta
from torchmetrics import Accuracy, F1Score

from src.data.dataset import load_label_encoders


class ParentGate(nn.Module):
    """Learns a gating vector from parent-level probabilities."""

    def __init__(self, parent_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(parent_dim, feature_dim)

    def forward(self, parent_probs: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.linear(parent_probs))
        return gate


class HierarchyLightningModule(LightningModule):
    def __init__(
        self,
        *,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int = 0,
        lr: float = 3e-4,
        backbone_lr: Optional[float] = None,
        weight_decay: float = 0.05,
        max_epochs: int = 30,
        w_family: float = 0.2,
        w_genus: float = 0.3,
        w_species: float = 1.0,
        lambda_consistency: float = 0.0,
        hierarchy_csv: str | Path | None = None,
        parent_gate: bool = False,
        parent_gate_type: str | None = None,
        class_weight_mode: Dict[str, str] | None = None,
        focal_gamma: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 0.0,
        mixup_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        encoders = load_label_encoders()
        self.num_family = max(len(encoders["family"]), 1)
        self.num_genus = len(encoders["genus"])
        self.num_species = len(encoders["species"])
        if self.num_species == 0:
            raise RuntimeError("Species encoder is empty. Run prep_data first.")

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.freeze_backbone_epochs = max(int(freeze_backbone_epochs), 0)
        self._backbone_frozen = freeze_backbone or self.freeze_backbone_epochs > 0
        if self._backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise AttributeError("Backbone does not expose num_features")

        self.family_head = nn.Linear(feature_dim, self.num_family)
        self.parent_gate = bool(parent_gate)
        self.parent_gate_type = parent_gate_type or "mlp"
        genus_input_dim = feature_dim + (self.num_family if self.parent_gate and self.parent_gate_type == "concat" else 0)
        species_input_dim = feature_dim + (self.num_genus if self.parent_gate and self.parent_gate_type == "concat" else 0)
        self.genus_head = nn.Linear(genus_input_dim, self.num_genus)
        self.species_head = nn.Linear(species_input_dim, self.num_species)
        if self.parent_gate and self.parent_gate_type != "concat":
            self.genus_gate = ParentGate(self.num_family, feature_dim)
            self.species_gate = ParentGate(self.num_genus, feature_dim)
        else:
            self.genus_gate = None
            self.species_gate = None

        self.family_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.genus_ce = nn.CrossEntropyLoss()
        self.class_weight_mode = class_weight_mode or {}
        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.w_family = w_family
        self.w_genus = w_genus
        self.w_species = w_species
        self.lambda_consistency = lambda_consistency
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_species)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_species)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_species, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_species, average="macro")

        self.train_genus_acc = Accuracy(task="multiclass", num_classes=self.num_genus)
        self.val_genus_acc = Accuracy(task="multiclass", num_classes=self.num_genus)
        self.train_genus_f1 = F1Score(task="multiclass", num_classes=self.num_genus, average="macro")
        self.val_genus_f1 = F1Score(task="multiclass", num_classes=self.num_genus, average="macro")

        self.train_family_acc = Accuracy(task="multiclass", num_classes=self.num_family)
        self.val_family_acc = Accuracy(task="multiclass", num_classes=self.num_family)
        self.train_family_f1 = F1Score(task="multiclass", num_classes=self.num_family, average="macro")
        self.val_family_f1 = F1Score(task="multiclass", num_classes=self.num_family, average="macro")

        if hierarchy_csv:
            hierarchy_csv = Path(hierarchy_csv)
            species_to_genus, genus_to_family = self._build_hierarchy_matrices(
                hierarchy_csv, encoders
            )
        else:
            species_to_genus = torch.zeros(self.num_species, self.num_genus)
            genus_to_family = torch.zeros(self.num_genus, self.num_family)
        self.register_buffer("species_to_genus", species_to_genus)
        self.register_buffer("genus_to_family", genus_to_family)

        if hierarchy_csv and self.class_weight_mode.get("species") == "balanced":
            species_weight = self._compute_class_weights(hierarchy_csv, encoders["species"])
        else:
            species_weight = torch.ones(self.num_species, dtype=torch.float32)
        self.register_buffer("species_weight", species_weight)
        self.focal_gamma = focal_gamma
        self.species_ce = nn.CrossEntropyLoss(weight=self.species_weight, reduction="none")

    def _build_hierarchy_matrices(
        self, csv_path: str | Path, encoders: Dict[str, Dict[str, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(csv_path, usecols=["species", "genus", "family"]).drop_duplicates("species")
        species_map = encoders["species"]
        genus_map = encoders["genus"]
        family_map = encoders["family"]

        species_to_genus = torch.zeros(len(species_map), len(genus_map), dtype=torch.float32)
        genus_to_family = torch.zeros(len(genus_map), max(len(family_map), 1), dtype=torch.float32)

        for _, row in df.iterrows():
            s = row["species"]
            g = row["genus"]
            f = row.get("family")
            if s not in species_map or g not in genus_map:
                continue
            species_to_genus[species_map[s], genus_map[g]] = 1.0
            if f and f in family_map:
                genus_to_family[genus_map[g], family_map[f]] = 1.0

        # normalize rows to avoid zero division
        for mat in (species_to_genus, genus_to_family):
            for idx in range(mat.shape[0]):
                row_sum = mat[idx].sum()
                if row_sum > 0:
                    mat[idx] = mat[idx] / row_sum
        return species_to_genus, genus_to_family

    def _compute_class_weights(self, csv_path: Path, encoder: Dict[str, int]) -> torch.Tensor:
        df = pd.read_csv(csv_path)
        counts = df["species"].value_counts()
        weights = torch.ones(len(encoder), dtype=torch.float32)
        for species, count in counts.items():
            idx = encoder.get(species)
            if idx is not None and count > 0:
                weights[idx] = 1.0 / count
        weights = weights * (len(encoder) / weights.sum().clamp(min=1e-8))
        return weights

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        features = self.backbone(images)
        family_logits = self.family_head(features)

        genus_features = features
        if self.parent_gate:
            family_probs = torch.softmax(family_logits, dim=1)
            if self.parent_gate_type == "concat":
                genus_features = torch.cat([features, family_probs], dim=1)
            else:
                genus_features = features * (1 + self.genus_gate(family_probs))

        genus_logits = self.genus_head(genus_features)

        species_features = features
        if self.parent_gate:
            genus_probs = torch.softmax(genus_logits, dim=1)
            if self.parent_gate_type == "concat":
                species_features = torch.cat([features, genus_probs], dim=1)
            else:
                species_features = features * (1 + self.species_gate(genus_probs))
        species_logits = self.species_head(species_features)

        return {
            "family": family_logits,
            "genus": genus_logits,
            "species": species_logits,
        }

    def _compute_loss(self, logits: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        family_target = batch["y_family"]
        if family_target.dtype.is_floating_point:
            mask = batch.get("y_family_mask")
            losses["loss_family"] = self._soft_cross_entropy(logits["family"], family_target, mask)
        else:
            losses["loss_family"] = self.family_ce(logits["family"], family_target.long())

        genus_target = batch["y_genus"]
        if genus_target.dtype.is_floating_point:
            losses["loss_genus"] = self._soft_cross_entropy(logits["genus"], genus_target)
        else:
            losses["loss_genus"] = self.genus_ce(logits["genus"], genus_target.long())

        species_logits = logits["species"]
        species_target = batch["y_species"]
        if species_target.dtype.is_floating_point:
            losses["loss_species"] = self._soft_cross_entropy(species_logits, species_target)
        else:
            y_species = species_target.long()
            ce_species = self.species_ce(species_logits, y_species)
            if self.focal_gamma > 0:
                probs = torch.softmax(species_logits, dim=1)
                pt = probs.gather(dim=1, index=y_species.view(-1, 1)).clamp(min=1e-8)
                focal_weight = (1 - pt) ** self.focal_gamma
                losses["loss_species"] = (focal_weight.view(-1) * ce_species).mean()
            else:
                losses["loss_species"] = ce_species.mean()

        total = (
            self.w_family * losses["loss_family"]
            + self.w_genus * losses["loss_genus"]
            + self.w_species * losses["loss_species"]
        )
        losses["loss"] = total

        if self.lambda_consistency > 0:
            species_probs = torch.softmax(logits["species"], dim=1)
            genus_probs = torch.softmax(logits["genus"], dim=1)
            family_probs = torch.softmax(logits["family"], dim=1)

            genus_from_species = torch.clamp(species_probs @ self.species_to_genus, min=1e-8)
            family_from_genus = torch.clamp(genus_probs @ self.genus_to_family, min=1e-8)
            genus_consistency = F.kl_div(
                genus_probs.log(), genus_from_species, reduction="batchmean"
            )
            family_consistency = F.kl_div(
                family_probs.log(), family_from_genus, reduction="batchmean"
            )
            losses["loss_consistency"] = genus_consistency + family_consistency
            losses["loss"] = losses["loss"] + self.lambda_consistency * losses["loss_consistency"]
        else:
            losses["loss_consistency"] = torch.zeros(1, device=self.device)
        return losses

    def _soft_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1)
        if mask is not None:
            mask = mask.bool()
            if not mask.any():
                return torch.zeros(1, device=logits.device)
            loss = loss[mask]
        return loss.mean()

    def _sample_lambda(self, alpha: float) -> float:
        if alpha <= 0:
            return 1.0
        lam = float(Beta(alpha, alpha).sample().item())
        return float(min(max(lam, 1e-3), 1.0 - 1e-3))

    def _rand_bbox(self, size: torch.Size, lam: float, device: torch.device) -> tuple[int, int, int, int]:
        _, _, height, width = size
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = max(1, int(width * cut_ratio))
        cut_h = max(1, int(height * cut_ratio))

        cx = torch.randint(0, width, (1,), device=device).item()
        cy = torch.randint(0, height, (1,), device=device).item()

        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(bbx1 + cut_w, width)
        bby2 = min(bby1 + cut_h, height)
        return bbx1, bby1, bbx2, bby2

    def _mix_labels(
        self,
        labels: torch.Tensor,
        perm: torch.Tensor,
        lam: torch.Tensor,
        num_classes: int,
        allow_ignore: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = labels.size(0)
        device = labels.device
        labels_long = labels.long()
        one_hot = torch.zeros(batch_size, num_classes, device=device)
        if allow_ignore:
            valid = labels_long >= 0
            if valid.any():
                one_hot[valid] = F.one_hot(labels_long[valid], num_classes=num_classes).float()
        else:
            valid = torch.ones(batch_size, dtype=torch.bool, device=device)
            one_hot = F.one_hot(labels_long, num_classes=num_classes).float()
        perm_one_hot = one_hot[perm]
        lam = lam.view(-1, 1)
        mixed = lam * one_hot + (1 - lam) * perm_one_hot
        mask = (valid | valid[perm]) if allow_ignore else None
        return mixed, mask

    def _apply_mixup(self, batch: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> None:
        if torch.rand(1).item() > self.mixup_prob:
            return

        images = batch["image"]
        batch_size = images.size(0)
        if batch_size < 2:
            return

        device = images.device
        perm = torch.randperm(batch_size, device=device)

        use_cutmix = self.cutmix_alpha > 0 and torch.rand(1).item() < 0.5
        if use_cutmix:
            lam_value = self._sample_lambda(self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam_value, device)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
            area = (bbx2 - bbx1) * (bby2 - bby1)
            lam_value = 1.0 - float(area) / float(images.size(2) * images.size(3))
        else:
            lam_value = self._sample_lambda(self.mixup_alpha)
            if lam_value >= 0.999:
                return
            images = lam_value * images + (1.0 - lam_value) * images[perm]

        batch["image"] = images
        lam_tensor = torch.full((batch_size,), lam_value, device=device)

        species_soft, _ = self._mix_labels(targets["species"], perm, lam_tensor, self.num_species)
        genus_soft, _ = self._mix_labels(targets["genus"], perm, lam_tensor, self.num_genus)
        family_soft, family_mask = self._mix_labels(
            targets["family"], perm, lam_tensor, self.num_family, allow_ignore=True
        )
        batch["y_species"] = species_soft
        batch["y_genus"] = genus_soft
        batch["y_family"] = family_soft
        if family_mask is not None:
            batch["y_family_mask"] = family_mask

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        metric_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        logits = self.forward(batch["image"])
        losses = self._compute_loss(logits, batch)
        self.log(f"{stage}/loss", losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{stage}/loss_family", losses["loss_family"], on_epoch=True)
        self.log(f"{stage}/loss_genus", losses["loss_genus"], on_epoch=True)
        self.log(f"{stage}/loss_species", losses["loss_species"], on_epoch=True)

        logits_species = logits["species"]
        logits_genus = logits["genus"]
        logits_family = logits["family"]
        if metric_targets is None:
            metric_targets = {
                "species": batch["y_species"],
                "genus": batch["y_genus"],
                "family": batch["y_family"],
            }
        y_species = metric_targets["species"].long()
        y_genus = metric_targets["genus"].long()
        y_family = metric_targets["family"].long()

        metric_acc = self.train_acc if stage == "train" else self.val_acc
        metric_f1 = self.train_f1 if stage == "train" else self.val_f1
        acc = metric_acc(logits_species, y_species)
        f1 = metric_f1(logits_species, y_species)
        self.log(f"{stage}/acc_species_top1", acc, prog_bar=True, on_epoch=True, on_step=stage == "train")
        self.log(f"{stage}/macro_f1_species", f1, prog_bar=False, on_epoch=True)

        genus_acc_metric = self.train_genus_acc if stage == "train" else self.val_genus_acc
        genus_f1_metric = self.train_genus_f1 if stage == "train" else self.val_genus_f1
        self.log(f"{stage}/acc_genus_top1", genus_acc_metric(logits_genus, y_genus), on_epoch=True)
        self.log(f"{stage}/macro_f1_genus", genus_f1_metric(logits_genus, y_genus), on_epoch=True)

        family_acc_metric = self.train_family_acc if stage == "train" else self.val_family_acc
        family_f1_metric = self.train_family_f1 if stage == "train" else self.val_family_f1
        valid_family = y_family >= 0
        if valid_family.any():
            self.log(
                f"{stage}/acc_family_top1",
                family_acc_metric(logits_family[valid_family], y_family[valid_family]),
                on_epoch=True,
            )
            self.log(
                f"{stage}/macro_f1_family",
                family_f1_metric(logits_family[valid_family], y_family[valid_family]),
                on_epoch=True,
            )
        return losses["loss"]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        metric_targets = {
            "species": batch["y_species"].clone(),
            "genus": batch["y_genus"].clone(),
            "family": batch["y_family"].clone(),
        }
        if self.use_mixup:
            self._apply_mixup(batch, metric_targets)
        return self._shared_step(batch, "train", metric_targets)

    def on_train_epoch_start(self) -> None:  # pragma: no cover - lightning hook
        if self._backbone_frozen and self.current_epoch >= self.freeze_backbone_epochs:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self._backbone_frozen = False
            self.print(f"Unfroze backbone at epoch {self.current_epoch}")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        head_params = list(self.family_head.parameters()) + list(self.genus_head.parameters()) + list(
            self.species_head.parameters()
        )
        if self.genus_gate:
            head_params += list(self.genus_gate.parameters())
        if self.species_gate:
            head_params += list(self.species_gate.parameters())
        backbone_params = list(self.backbone.parameters())
        param_groups = [
            {"params": backbone_params, "lr": self.backbone_lr},
            {"params": head_params, "lr": self.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


__all__ = ["HierarchyLightningModule"]
