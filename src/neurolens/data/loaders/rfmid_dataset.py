"""
RFMiD Dataset Loader for NeuroLens.

RFMiD 2.0 — Retinal Fundus Multi-Disease Image Dataset
Source: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid

Label Mapping for NeuroLens (4-class neurological screening):
    We map RFMiD disease labels to our 4 target classes:
    0 = Alzheimer's / Neurodegeneration indicators
    1 = Parkinson's indicators
    2 = MCI (Mild Cognitive Impairment) indicators
    3 = Healthy / Normal

    NOTE: RFMiD does not have direct AD/PD labels — this is a research
    assumption based on the retinal biomarker literature. The primary
    RFMiD experiment uses Disease_Risk as binary target and the full
    46-class set for pretraining. Neurological mapping is applied in
    the fine-tuning phase using UK Biobank crosswalk.

Usage:
    dataset = RFMiDDataset(
        csv_path="data/raw/rfmid/RFMiD_Training_Labels.csv",
        img_dir="data/raw/rfmid/Training/",
        transform=get_train_transforms()
    )
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# ── Disease Risk label column in RFMiD CSV
DISEASE_RISK_COL = "Disease_Risk"
ID_COL = "ID"


class RFMiDDataset(Dataset):
    """
    PyTorch Dataset for RFMiD 2.0 retinal fundus images.

    Args:
        csv_path: Path to RFMiD labels CSV file.
        img_dir: Directory containing fundus image files.
        transform: Albumentations transform pipeline.
        target_col: Column to use as label. Default "Disease_Risk" (binary).
            Set to "neuro_class" for 4-class neurological mapping.
        apply_clahe: Apply CLAHE preprocessing. Default True.
        green_channel: Apply green channel enhancement. Default True.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform=None,
        target_col: str = DISEASE_RISK_COL,
        apply_clahe: bool = True,
        green_channel: bool = True,
    ) -> None:
        self.img_dir = img_dir
        self.transform = transform
        self.target_col = target_col
        self.apply_clahe = apply_clahe
        self.green_channel = green_channel

        # Load and validate CSV
        self.df = pd.read_csv(csv_path)
        self._validate_csv()
        self._build_neuro_labels()

        logger.info(
            f"RFMiDDataset: {len(self.df)} samples loaded from {csv_path}"
        )
        logger.info(f"Label distribution:\n{self.df[target_col].value_counts()}")

    def _validate_csv(self) -> None:
        """Check required columns exist."""
        required = [ID_COL, DISEASE_RISK_COL]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {list(self.df.columns)}")

    def _build_neuro_labels(self) -> None:
        """
        Build 4-class neurological label from RFMiD columns.

        Mapping strategy (literature-informed, for research purposes):
            Healthy (Disease_Risk=0) → class 3
            DR/Diabetic markers     → class 2 (MCI proxy — vascular overlap)
            Glaucoma indicators     → class 1 (Parkinson's proxy — RNFL loss)
            Macular degeneration    → class 0 (AD proxy — drusen, amyloid overlap)
            Other disease           → class 2 (MCI proxy)

        This is a research approximation. Real neurological labels
        require UK Biobank crosswalk (implemented in fine-tuning phase).
        """
        neuro = []
        for _, row in self.df.iterrows():
            if row[DISEASE_RISK_COL] == 0:
                neuro.append(3)  # Healthy
            elif "MH" in self.df.columns and row.get("MH", 0) == 1:
                neuro.append(0)  # Macular hole → AD proxy
            elif "TSLN" in self.df.columns and row.get("TSLN", 0) == 1:
                neuro.append(1)  # Tessellation → PD proxy (RNFL related)
            elif "DN" in self.df.columns and row.get("DN", 0) == 1:
                neuro.append(2)  # Drusen → MCI proxy
            else:
                neuro.append(2)  # All other diseases → MCI proxy

        self.df["neuro_class"] = neuro

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_id = int(row[ID_COL])

        # Try common image extensions
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(
                f"Image {img_id} not found in {self.img_dir}"
            )

        # Load and preprocess
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.apply_clahe:
            import cv2
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = int(row[self.target_col])
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced training.

        Returns:
            Tensor of shape (n_classes,) with per-class weights.
        """
        counts = self.df[self.target_col].value_counts().sort_index()
        n_classes = len(counts)
        weights = torch.zeros(n_classes)
        total = len(self.df)
        for cls, count in counts.items():
            weights[int(cls)] = total / (n_classes * count)
        return weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Build WeightedRandomSampler for balanced batch sampling.
        Oversamples minority classes to create balanced batches.
        """
        class_weights = self.get_class_weights()
        sample_weights = [
            class_weights[int(label)].item()
            for label in self.df[self.target_col]
        ]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def make_dataloaders(
    csv_path: str,
    img_dir: str,
    img_size: int = 224,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    target_col: str = DISEASE_RISK_COL,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from RFMiD dataset.

    Args:
        csv_path: Path to RFMiD labels CSV.
        img_dir: Directory containing images.
        img_size: Image resize dimension.
        batch_size: Batch size (keep <=16 on Colab CPU, <=32 on T4 GPU).
        val_split: Fraction for validation set.
        test_split: Fraction for test set.
        target_col: Label column name.
        num_workers: DataLoader workers (0 for Windows, 2 for Colab).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: 'train', 'val', 'test'.
    """
    from neurolens.data.preprocessing.fundus_transforms import (
        get_train_transforms,
        get_val_transforms,
    )

    # Load full CSV and split
    df = pd.read_csv(csv_path)
    np.random.seed(seed)
    idx = np.random.permutation(len(df))

    n_test = int(len(df) * test_split)
    n_val = int(len(df) * val_split)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # Save splits for reproducibility
    splits_dir = os.path.join(os.path.dirname(csv_path), "..", "..", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    np.save(os.path.join(splits_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(splits_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(splits_dir, "test_idx.npy"), test_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Save split CSVs temporarily
    import tempfile
    train_csv = os.path.join(tempfile.gettempdir(), "rfmid_train.csv")
    val_csv = os.path.join(tempfile.gettempdir(), "rfmid_val.csv")
    test_csv = os.path.join(tempfile.gettempdir(), "rfmid_test.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_dataset = RFMiDDataset(train_csv, img_dir, get_train_transforms(img_size), target_col)
    val_dataset = RFMiDDataset(val_csv, img_dir, get_val_transforms(img_size), target_col)
    test_dataset = RFMiDDataset(test_csv, img_dir, get_val_transforms(img_size), target_col)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_dataset.get_weighted_sampler(),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Splits — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return {"train": train_loader, "val": val_loader, "test": test_loader}
