"""
ConvDPT module for T-3DGS transient modeling
Based on DPT architecture for covariance parameter prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDPT(nn.Module):
    """
    Convolutional DPT for predicting covariance parameters (σ₁, σ₂, ρ)
    for both GT and rendered images in T-3DGS framework.
    """
    
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim

        # Feature fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # Input: GT+Render RGB (6 channels)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Feature processing from DINO features (input channels = 2*feature_dim)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(2 * feature_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
        )

        # Combined feature processing
        self.combined_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # 128 + 128 = 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output heads for covariance parameters
        # GT parameters: σ₁, σ₂, ρ
        self.gt_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),  # 3 channels: σ₁, σ₂, ρ
        )

        # Render parameters: s₁, s₂, r  
        self.render_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),  # 3 channels: s₁, s₂, r
        )
        
    def forward(self, rgb_input: torch.Tensor, dino_extractor):
        """
        Args:
            rgb_input: [2*B, 3, H, W] - concatenated GT and rendered images
            dino_extractor: DINO feature extractor
            
        Returns:
            tuple: (gt_params, render_params)
                gt_params: [σ₁, σ₂, ρ] each [B, H, W]
                render_params: [s₁, s₂, r] each [B, H, W]
        """
        B, C, H, W = rgb_input.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        B = B // 2  # Since input is concatenated GT+Render
        
        # Split input
        gt_rgb = rgb_input[:B]      # [B, 3, H, W] 
        render_rgb = rgb_input[B:]  # [B, 3, H, W]
        
        # Concatenate for joint processing
        joint_rgb = torch.cat([gt_rgb, render_rgb], dim=1)  # [B, 6, H, W]
        
        # RGB feature extraction
        rgb_feat = self.fusion_conv(joint_rgb)  # [B, 128, H, W]
        
        # DINO feature extraction
        with torch.no_grad():
            dino_tokens_gt, Th, Tw, _, _ = dino_extractor.extract_tokens(gt_rgb)
            dino_tokens_render, _, _, _, _ = dino_extractor.extract_tokens(render_rgb)
        
        # Combine DINO features
        dino_combined = torch.cat([dino_tokens_gt, dino_tokens_render], dim=-1)  # [B, Th*Tw, 2*C]
        dino_feat = dino_combined.view(B, Th, Tw, -1).permute(0, 3, 1, 2)  # [B, 2*C, Th, Tw]
        
        # Resize DINO features to match RGB resolution
        patch_size = dino_extractor.patch_size
        Hpad, Wpad = Th * patch_size, Tw * patch_size
        dino_feat_up = F.interpolate(dino_feat, size=(Hpad, Wpad), mode='bilinear', align_corners=False)
        dino_feat_up = dino_feat_up[:, :, :H, :W]  # Crop to original size
        
        # Project DINO features
        dino_feat_proj = self.feat_proj(dino_feat_up)  # [B, 128, H, W]
        
        # Combine RGB and DINO features
        combined_feat = torch.cat([rgb_feat, dino_feat_proj], dim=1)  # [B, 256, H, W]
        combined_feat = self.combined_conv(combined_feat)  # [B, 128, H, W]
        
        # Predict covariance parameters
        gt_params = self.gt_head(combined_feat)      # [B, 3, H, W]
        render_params = self.render_head(combined_feat)  # [B, 3, H, W]
        
        # Apply activations to ensure valid parameter ranges
        # σ₁, σ₂, s₁, s₂ > 0 (use softplus)
        # ρ, r ∈ (-1, 1) (use tanh) 
        gt_sigma1 = F.softplus(gt_params[:, 0]) + 1e-6      # [B, H, W]
        gt_sigma2 = F.softplus(gt_params[:, 1]) + 1e-6      # [B, H, W]
        gt_rho = torch.tanh(gt_params[:, 2]) * 0.99         # [B, H, W], slight < 1 for numerical stability
        
        render_s1 = F.softplus(render_params[:, 0]) + 1e-6  # [B, H, W]
        render_s2 = F.softplus(render_params[:, 1]) + 1e-6  # [B, H, W] 
        render_r = torch.tanh(render_params[:, 2]) * 0.99   # [B, H, W]
        
        return ([gt_sigma1, gt_sigma2, gt_rho], [render_s1, render_s2, render_r])