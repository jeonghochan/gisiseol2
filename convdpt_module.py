"""
ConvDPT module for T-3DGS transient modeling
Based on DPT architecture for covariance parameter prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dino_utils import DPT_Head, pad_and_unpad


class ConvDPT(nn.Module):
    """
    ConvDPT matching the implementation in `utils/transient_utils.py`.

    This version expects a `feature_extractor` argument that exposes
    `extract(image)` returning patch tokens with shape [B, Th*Tw, C].
    The forward signature is `forward(image, feature_extractor)` and it
    returns a tensor of shape [B, 3, H, W] (or when given stacked GT+Render
    inputs, [2*B,3,H,W]).
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = DPT_Head(self.feature_dim)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    @pad_and_unpad
    def forward(self, image: torch.Tensor, feature_extractor):
        """Forward using external feature_extractor.extract(image).

        Args:
            image: [B,3,H,W]
            feature_extractor: object with `extract(image)` -> [B, Th*Tw, C]

        Returns:
            covariations: [B, 3, H, W]
        """
        width, height = image.shape[2:]

        # Extract DINO patch tokens
        features = feature_extractor.extract(image)  # [B, Th*Tw, C]

        # Reshape to [B, C, Th, Tw] where Th = width//14, Tw = height//14
        features = features.reshape(-1, width // 14, height // 14, self.feature_dim).permute(0, 3, 1, 2)

        logits = self.head(features)
        logits = F.interpolate(logits, size=(width, height), mode='bilinear', align_corners=False)

        sigma = self.softplus(logits[:, :2])
        rho = self.tanh(logits[:, 2:3])
        covariations = torch.cat([sigma, rho], dim=1)
        return covariations