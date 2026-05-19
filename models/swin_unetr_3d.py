"""
Swin UNETR (3D) Architecture
============================
A state-of-the-art transformer-based 3D medical image segmentation model.
Wraps MONAI's SwinUNETR implementation with defaults optimized for BraTS 2023.

Reference:
    Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation
    of Medical Images", arXiv:2201.01266.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETR3D(nn.Module):
    """
    Swin UNETR 3D implementation optimized for BraTS 2023.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Input volume size, default (192, 192, 128).
    in_channels : int
        Number of input modalities (4 for BraTS).
    out_channels : int
        Number of output channels (3 for TC, WT, ET).
    feature_size : int
        Dimension of the embedding space, default 48.
    """

    def __init__(
        self,
        img_size=(192, 192, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    ):
        super().__init__()

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 4, H, W, D)
        Returns:
            Tensor of shape (B, 3, H, W, D)
        """
        # Ensure input is float32
        x = x.float()
        logits = self.model(x)
        
        return logits

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return f"SwinUNETR_3D (feature_size={self.model.feature_size}, params={num_params:,})"
