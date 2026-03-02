"""
NeuroLens Training Script.

Trains BayesViT with Geodesic Variational Attention on RFMiD dataset.
Designed to run on Google Colab (T4 GPU) or CPU (slower).

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml training.epochs=10
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import wandb
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurolens.models.bayes_vit import BayesViT
from neurolens.training.losses.elbo_loss import ELBOLoss, CyclicalBetaScheduler
from neurolens.inference.uncertainty.decomposer import UncertaintyDecomposer
from neurolens.inference.conformal.predictor import ConformalPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neurolens.train")


class Trainer:
    """
    Full training loop for BayesViT with ELBO loss.

    Features:
        - Cyclical KL annealing to prevent posterior collapse
        - Gradient clipping for training stability
        - Best model checkpoint saving (by validation AUC)
        - W&B logging of all metrics
        - Conformal calibration after training
    """

    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")

        # ── Model ─────────────────────────────────────────────────────────
        self.model = BayesViT(
            num_classes=cfg.model.num_classes,
            num_trainable_blocks=cfg.model.num_trainable_blocks,
            img_size=cfg.model.img_size,
            use_riemannian=cfg.model.use_riemannian,
            pretrained=cfg.model.pretrained,
            device=self.device,
        ).to(self.device)

        # ── Loss & Optimizer ───────────────────────────────────────────────
        self.loss_fn = ELBOLoss(
            n_classes=cfg.model.num_classes,
            gamma=cfg.training.gamma_ece,
        )
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=1e-6,
        )
        self.beta_scheduler = CyclicalBetaScheduler(
            n_epochs=cfg.training.epochs,
            n_cycles=cfg.training.n_cycles_kl,
        )

        # ── Utilities ──────────────────────────────────────────────────────
        self.decomposer = UncertaintyDecomposer()
        self.conformal = ConformalPredictor(alpha=cfg.inference.alpha)
        self.best_auc = 0.0
        self.best_epoch = 0

        os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)

    def train_epoch(
        self, loader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        beta = self.beta_scheduler.get_beta(epoch)

        total_loss = total_ce = total_kl = total_ece = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits, kl = self.model(images)

            losses = self.loss_fn(
                logits, labels, kl,
                beta=beta,
                batch_size=images.size(0),
            )
            losses["total"].backward()

            # Gradient clipping for stable Bayesian training
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_ce += losses["ce"].item()
            total_kl += losses["kl"].item()
            total_ece += losses["ece"].item()

            probs = torch.softmax(logits.detach(), dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if batch_idx % self.cfg.logging.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                    f"loss={losses['total'].item():.4f} "
                    f"ce={losses['ce'].item():.4f} "
                    f"kl={losses['kl'].item():.4f} "
                    f"beta={beta:.3f}"
                )

        n = len(loader)
        metrics = {
            "train/loss": total_loss / n,
            "train/ce": total_ce / n,
            "train/kl": total_kl / n,
            "train/ece": total_ece / n,
            "train/beta": beta,
        }

        # Compute AUC if multi-class
        try:
            probs_arr = np.array(all_probs)
            labels_arr = np.array(all_labels)
            if len(np.unique(labels_arr)) > 1:
                auc = roc_auc_score(
                    labels_arr, probs_arr,
                    multi_class="ovr", average="macro"
                )
                metrics["train/auc"] = auc
        except Exception:
            pass

        return metrics

    @torch.no_grad()
    def validate(self, loader, epoch: int, split: str = "val") -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        all_epistemic, all_aleatoric = [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Standard forward pass for loss
            logits, kl = self.model(images)
            losses = self.loss_fn(logits, labels, kl, beta=1.0)
            total_loss += losses["total"].item()

            # MC uncertainty for a subset (every 5th batch to save time)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        preds_arr = np.array(all_preds)

        metrics = {f"{split}/loss": total_loss / len(loader)}

        try:
            if len(np.unique(labels_arr)) > 1:
                auc = roc_auc_score(
                    labels_arr, probs_arr,
                    multi_class="ovr", average="macro"
                )
                metrics[f"{split}/auc"] = auc

            f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
            acc = float(np.mean(preds_arr == labels_arr))
            metrics[f"{split}/f1"] = f1
            metrics[f"{split}/acc"] = acc
        except Exception as e:
            logger.warning(f"Metric computation failed: {e}")

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best") -> None:
        path = os.path.join(
            self.cfg.logging.checkpoint_dir, f"neurolens_{tag}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "cfg": OmegaConf.to_container(self.cfg),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def run(self, train_loader, val_loader, cal_loader=None) -> None:
        """Main training loop."""
        # Init W&B
        wandb.init(
            project=self.cfg.logging.project,
            entity=self.cfg.logging.entity,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            name=f"bayesvit_gva_run",
        )
        wandb.watch(self.model, log="gradients", log_freq=100)

        logger.info(f"Starting training for {self.cfg.training.epochs} epochs")

        for epoch in range(self.cfg.training.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            # Validate
            val_metrics = self.validate(val_loader, epoch, split="val")
            # LR step
            self.scheduler.step()

            # Combine and log
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["lr"] = self.scheduler.get_last_lr()[0]
            wandb.log(all_metrics, step=epoch)

            # Save best checkpoint
            val_auc = val_metrics.get("val/auc", val_metrics.get("val/acc", 0.0))
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, all_metrics, tag="best")

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, all_metrics, tag=f"epoch_{epoch+1}")

            logger.info(
                f"Epoch {epoch}/{self.cfg.training.epochs} | "
                f"val_auc={val_auc:.4f} | best={self.best_auc:.4f} (ep {self.best_epoch})"
            )

        # Post-training: calibrate conformal predictor
        if cal_loader is not None:
            logger.info("Calibrating conformal predictor...")
            self._calibrate_conformal(cal_loader)

        wandb.finish()
        logger.info(f"Training complete. Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")

    def _calibrate_conformal(self, cal_loader) -> None:
        """Run conformal calibration on held-out calibration set."""
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for images, labels in cal_loader:
                images = images.to(self.device)
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu())
                all_labels.append(labels)

        probs_tensor = torch.cat(all_probs, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        q_hat = self.conformal.calibrate(probs_tensor, labels_tensor)

        # Save conformal threshold with checkpoint
        conformal_path = os.path.join(
            self.cfg.logging.checkpoint_dir, "conformal_q_hat.pt"
        )
        torch.save({"q_hat": q_hat, "alpha": self.cfg.inference.alpha}, conformal_path)
        logger.info(f"Conformal q_hat={q_hat:.4f} saved to {conformal_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NeuroLens BayesViT")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.epochs:
        cfg.training.epochs = args.epochs

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataloaders
    from neurolens.data.loaders.rfmid_dataset import make_dataloaders
    csv_path = os.path.join(cfg.data.data_dir, "RFMiD_Training_Labels.csv")
    img_dir = os.path.join(cfg.data.data_dir, "Training")

    loaders = make_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=cfg.model.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )

    trainer = Trainer(cfg)
    trainer.run(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cal_loader=loaders["val"],  # Use val set for conformal calibration
    )


if __name__ == "__main__":
    main()
