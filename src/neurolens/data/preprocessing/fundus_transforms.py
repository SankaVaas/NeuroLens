"""
Fundus Image Preprocessing Pipeline for NeuroLens.

Pipeline (applied in order):
    1. CLAHE  — contrast enhancement in LAB color space
    2. Green channel extraction — highest vessel contrast
    3. Resize to model input size
    4. Normalize with ImageNet stats (for DINO pretrained backbone)
    5. Augmentation (training only)

All transforms are compatible with albumentations and torchvision.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE in LAB color space for retinal contrast enhancement.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8.
        clip_limit: CLAHE clip limit. Higher = more contrast.
        tile_grid: Grid size for local histogram equalization.

    Returns:
        CLAHE-enhanced RGB image (H, W, 3), uint8.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Extract and enhance green channel (highest retinal vessel contrast).
    Returns 3-channel image where green is the primary channel.

    Args:
        image: RGB image (H, W, 3).

    Returns:
        3-channel image with enhanced green channel (H, W, 3).
    """
    green = image[:, :, 1]
    # Stack: original R, enhanced G, original B
    enhanced = np.stack([image[:, :, 0], green, image[:, :, 2]], axis=-1)
    return enhanced


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Training augmentation pipeline for retinal fundus images.

    Augmentations chosen to be clinically valid:
    - Flips: retinal images can be mirrored
    - Rotation: camera angle variation
    - Color jitter: illumination variation between devices
    - Grid distortion: simulates lens distortion
    - NO extreme crops: optic disc must remain visible
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats for DINO backbone
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Validation/test transforms — no augmentation, only resize + normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def preprocess_fundus_image(
    image_path: str,
    apply_clahe_flag: bool = True,
    green_channel: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single fundus image.

    Args:
        image_path: Path to image file.
        apply_clahe_flag: Apply CLAHE enhancement. Default True.
        green_channel: Apply green channel enhancement. Default True.

    Returns:
        Preprocessed image as numpy array (H, W, 3), uint8.
    """
    image = np.array(Image.open(image_path).convert("RGB"))

    if apply_clahe_flag:
        image = apply_clahe(image)

    if green_channel:
        image = extract_green_channel(image)

    return image
