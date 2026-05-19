"""
Training Entry Point for BraTS 2023 with Swin UNETR
===================================================
This script initializes the Swin UNETR 3D model, sets up the BraTS 2023 dataset
(GLI, MEN, or PED), and starts the training loop with hybrid Dice + Focal loss.

Usage:
    python train_swin_unetr.py --challenge PED --fold 0 --data_dir ./dataset/PED_Data
"""

import os
import argparse
import torch
import torch.nn as nn
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from models.swin_unetr_3d import SwinUNETR3D
from model_schedule.train import train
from losses.loss_functions import DiceLoss, FocalLoss

def main():
    parser = argparse.ArgumentParser(description="BraTS 2023 Swin UNETR Training")
    parser.add_argument("--challenge", type=str, default="PED", choices=["GLI", "MEN", "PED"], help="Sub-challenge to train on")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold index (0-4)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the challenge training data directory")
    parser.add_argument("--split_path", type=str, help="Path to the split JSON file")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    
    args = parser.parse_args()

    # 1. Define Transforms (Augmentations)
    # Note: RandFlipd and RandRotated are essential for 3D volumetric medical images.
    train_transform = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotated(keys=["image", "label"], prob=0.5, range_x=0.2, range_y=0.2, range_z=0.2),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
    ])

    # 2. Initialize Model
    # ROI size (192, 192, 128) is optimized for BraTS 2023 PED/MEN
    model = SwinUNETR3D(
        img_size=(192, 192, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48
    )

    # 3. Define Hybrid Loss
    # Combining Dice (for overlap) and Focal (for class imbalance) is standard for BraTS
    loss_functions = [
        DiceLoss(),
        FocalLoss()
    ]
    loss_weights = [1.0, 1.0]  # Balanced starting weights

    # 4. Start Training
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f"Starting Swin UNETR training for {args.challenge} (Fold {args.fold})...")
    
    # We call the core train function from model_schedule.train
    # It handles device detection, optimizer setup, and the epoch loop.
    train(
        data_dir=args.data_dir,
        model=model,
        loss_functions=loss_functions,
        loss_weights=loss_weights,
        init_lr=args.lr,
        max_epoch=args.epochs,
        training_regions='overlapping', # BraTS 2023 evaluation regions
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        transform=train_transform
    )

if __name__ == "__main__":
    main()
