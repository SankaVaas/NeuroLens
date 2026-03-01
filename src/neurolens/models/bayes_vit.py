"""
BayesViT — Bayesian Vision Transformer with Geodesic Variational Attention.

Wraps a pretrained DINO ViT-Small backbone and replaces its attention layers
with GeodesicVariationalAttention modules. Lower backbone layers are frozen;
upper layers are fine-tuned with Bayesian attention heads.

Usage:
    model = BayesViT(num_classes=4, num_trainable_blocks=4)
    out, kl = model(images)   # kl used in ELBO loss
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BayesViT(nn.Module):
    """
    Bayesian ViT with Geodesic Variational Attention heads.

    Args:
        num_classes: Number of disease categories. Default 4 (AD, PD, MCI, Healthy).
        num_trainable_blocks: Number of transformer blocks (from the top)
            to replace with GVA layers. Remaining blocks are frozen.
        img_size: Input image size. Default 224.
        use_riemannian: Enable Riemannian GVA. Set False for ablation.
        pretrained: Load DINO pretrained weights. Default True.
        device: Compute device.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_trainable_blocks: int = 4,
        img_size: int = 224,
        use_riemannian: bool = True,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_trainable_blocks = num_trainable_blocks
        self.device = device or torch.device("cpu")

        # ── Load pretrained ViT-Small backbone ────────────────────────────
        self.backbone = self._load_backbone(pretrained, img_size)
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-Small

        # ── Freeze lower blocks ───────────────────────────────────────────
        self._freeze_backbone(num_trainable_blocks)

        # ── Replace top attention blocks with GVA ─────────────────────────
        self._replace_attention_with_gva(num_trainable_blocks, use_riemannian)

        # ── Classification head ───────────────────────────────────────────
        from neurolens.models.bayesian.variational_linear import VariationalLinear

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        logger.info(
            f"BayesViT initialized: {num_classes} classes, "
            f"{num_trainable_blocks} GVA blocks, Riemannian={use_riemannian}"
        )

    def _load_backbone(self, pretrained: bool, img_size: int) -> nn.Module:
        """Load DINO ViT-Small via timm."""
        try:
            import timm
            backbone = timm.create_model(
                "vit_small_patch8_224.dino",
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0,  # Remove classification head
            )
            return backbone
        except Exception as e:
            logger.error(f"Failed to load backbone: {e}")
            raise

    def _freeze_backbone(self, num_trainable_blocks: int) -> None:
        """Freeze all backbone parameters, then unfreeze top N blocks."""
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top N transformer blocks
        total_blocks = len(self.backbone.blocks)
        trainable_start = total_blocks - num_trainable_blocks

        for i, block in enumerate(self.backbone.blocks):
            if i >= trainable_start:
                for param in block.parameters():
                    param.requires_grad = True

        # Always train norm layer
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        n_frozen = sum(
            1 for p in self.backbone.parameters() if not p.requires_grad
        )
        n_trainable = sum(
            1 for p in self.backbone.parameters() if p.requires_grad
        )
        logger.info(f"Backbone: {n_frozen} frozen, {n_trainable} trainable params")

    def _replace_attention_with_gva(
        self, num_trainable_blocks: int, use_riemannian: bool
    ) -> None:
        """Replace top N attention blocks with GeodesicVariationalAttention."""
        from neurolens.models.manifold.geodesic_attention import GeodesicVariationalAttention

        total_blocks = len(self.backbone.blocks)
        trainable_start = total_blocks - num_trainable_blocks

        self.gva_layers = nn.ModuleList()

        for i, block in enumerate(self.backbone.blocks):
            if i >= trainable_start:
                gva = GeodesicVariationalAttention(
                    dim=self.embed_dim,
                    num_heads=block.attn.num_heads,
                    dropout=0.0,
                    use_riemannian=use_riemannian,
                )
                self.gva_layers.append(gva)
                # Store reference for forward pass
                block._gva = gva
                block._use_gva = True
            else:
                block._use_gva = False

        logger.info(f"Replaced {len(self.gva_layers)} attention blocks with GVA")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BayesViT.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            logits: Class logits (B, num_classes).
            kl_total: Total KL divergence across all GVA layers (scalar).
        """
        kl_total = torch.tensor(0.0, device=x.device)

        # ── Patch embedding + positional encoding ─────────────────────────
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # ── Transformer blocks ────────────────────────────────────────────
        for block in self.backbone.blocks:
            if getattr(block, "_use_gva", False):
                # GVA block: Bayesian attention + residual
                normed = block.norm1(x)
                attn_out, kl = block._gva(normed)
                x = x + block.drop_path(attn_out)
                kl_total = kl_total + kl
                # Standard MLP sublayer
                x = x + block.drop_path(block.mlp(block.norm2(x)))
            else:
                # Standard frozen block
                x = block(x)

        x = self.backbone.norm(x)

        # ── CLS token → classifier ────────────────────────────────────────
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        return logits, kl_total

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 20
    ) -> dict:
        """
        Monte Carlo inference for uncertainty decomposition.

        Args:
            x: Input images (B, C, H, W).
            n_samples: Number of MC samples. Default 20.

        Returns:
            Dictionary with keys: mean_probs, predictive_entropy,
            epistemic_uncertainty, aleatoric_uncertainty, kl_mean.
        """
        self.train()  # Enable stochastic forward passes

        all_probs = []
        all_kl = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, kl = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)
                all_kl.append(kl.item())

        self.eval()

        probs_stack = torch.stack(all_probs, dim=0)  # (T, B, C)
        mean_probs = probs_stack.mean(dim=0)         # (B, C)

        # Predictive entropy H[y|x] = -sum p(y|x) log p(y|x)
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)

        # Aleatoric: E_w[H[y|x,w]] = mean entropy per sample
        per_sample_entropy = -(probs_stack * torch.log(probs_stack + 1e-8)).sum(dim=-1)
        aleatoric = per_sample_entropy.mean(dim=0)

        # Epistemic = Predictive - Aleatoric (Mutual Information)
        epistemic = pred_entropy - aleatoric

        return {
            "mean_probs": mean_probs,
            "predictive_entropy": pred_entropy,
            "epistemic_uncertainty": epistemic.clamp(min=0.0),
            "aleatoric_uncertainty": aleatoric,
            "kl_mean": sum(all_kl) / len(all_kl),
        }
