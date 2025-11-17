import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import shutil
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
import argparse
import yaml
import viser
import nerfview
from datasets.colmap import Dataset, Parser, ClutterDataset, SemanticParser
from datasets.traj import generate_interpolated_path, get_ordered_poses
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    get_positional_encodings,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
    SpotLessModule,
)

from gsplat.rendering import rasterization
from dino_utils import DinoFeatureExtractor, DPT_Head, DinoUpsampleHead, ConvDPT
# from transient_utils import ConvDPT



@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 8
    # Normalize the axis and world view
    normalize: bool = True
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Train and test image name keywords
    train_keyword: str = "clutter"#clutter
    test_keyword: str = "extra"#extra
    # Enable semantic feature based training
    semantics: bool = True
    # Enable clustering of semantic features
    cluster: bool = False
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    min_steps: int = 7000
    max_steps: int = 30000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7000, 30000])
                                  
    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.0
    # Loss types: l1, robust
    loss_type: str = "robust"
    # Robust loss percentile for threshold
    robust_percentile: float = 0.7
    # enable alpha scheduling
    schedule: bool = True
    # alpha sampling schedule rate (higher more robust)
    schedule_beta: float = -3e-3
    # Thresholds for mlp mask supervision
    lower_bound: float = 0.5
    upper_bound: float = 0.9
    # bin size for the error hist for robust threshold
    bin_size: int = 10000

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float =  0.0002 
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    # Stop refining GSs after this iteration
    refine_start_iter: int = 100
    refine_stop_iter: int = 20000 # tag
    # Reset opacities every this steps
    reset_every: int = 300000
    # Refine GSs every this steps
    refine_every: int = 100
    # Reset SH specular coefficients once
    reset_sh: int = 8002
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Use utilization-based pruning (UBP) for compression: xection 4.2.3 https://arxiv.org/pdf/2406.20055
    ubp: bool = True
    # Threshold for UBP
    ubp_thresh: float = 1e-14
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    #revised
    #mask directory
    mask_dir: Optional[str] = None
    #someday to change 
    # seed : int = 42
    pseudo_gt_dir: Optional[str] = None  # temporary directory for pseudo GT

    # Weight for MLP mask loss
    mlp_gt_lambda: float = 0.1
    # Weight for DINO feature loss
    dino_loss_flag: bool = True
    dino_loss_lambda: float = 0.2 #tag
    

    #parameter for self-ensemble-revised-1013
    # ---------------- Self-Ensemble / Pseudo-view Co-Reg --------------
    # mask 모드(동적마스크 겹침 비율) 파라미터
    uap_dyn_overlap_frac: float = 0.15      # splat 면적 대비 '동적(=0)' 픽셀 비율 임계 (0.20 -> 0.15로 더 민감하게)
    # 공통: 섭동 크기 스케줄
    uap_noise_init: float = 0.12            # 초기 섭동 강도 (0.08 -> 0.12로 증가)
    uap_noise_final: float = 0.03           # 최종 섭동 강도 (0.02 -> 0.03로 증가)
    uap_noise_anneal_end: int = 20000      # 이 step까지 선형 점감


    se_coreg_enable: bool = True
    se_coreg_start_iter: int = 7000
    se_coreg_lambda: float = 0.1 #0.5            # render↔render loss 계수
    se_pseudo_K: int = 2                    # pseudo view 개수
    se_coreg_refine_every: int = 100    # self-ensemble 
    # fallback jitter (cameras.PseudoCamera 미사용시)
    se_jitter_deg: float = 1.0              # yaw/pitch/roll 표준편차(도)
    se_jitter_trans: float = 0.01           # 번역 표준편차(장면스케일 비율)
    #--------------------------------------------------------------
    
    #revised-1015
    #-----------------for opacity based pruning---
    gaussian_visibility_thresh: float = 0.05# Threshold for Gaussian visibility ratio (0.08 -> 0.15로 증가해서 더 많이 pruning) 
    start_pseudo_view_iter: int = 7000 # Start opacity-based pruning earlier (7000 -> 5000)

    #revised-1016 - Advanced Floater Detection
    #-----------------for advanced floater detection algorithms---
    enable_depth_floater_detection: bool = True    # Depth-based floater detection
    depth_floater_thresh: float = 0.005                # Depth inconsistency threshold
    # Ghost/Transparency Artifact Fix
    #-----------------for handling under-reconstructed dynamic regions---
    dynamic_prune_start_iter: int = 15000          # When to start aggressive pruning
    enable_dynamic_inpainting: bool = True          # Fill dynamic regions with nearby background Gaussians
    #--------------------------------------------------------------
    #revised-1017
    # Transient parameters
    transient_training_enable: bool = False    # on/off
    transient_loss_start_iter: int = 7000     # iteration부터 main loss에 transient loss 적용
    KL_threshold: float = 0.5           # KL divergence threshold for masking
    schedule_beta: float = -3e-3        # Alpha sampling schedule rate
    #revised-1028
    # Minimum age (in training steps) before a newly created Gaussian can be
    # pruned. Helps avoid immediate prune after duplication/splitting.
    min_age_before_prune: int = 150

  

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers

#revised-0919
# ---- (NEW) mask weight per-fragment / per-gaussian ----
# binary_mask: [1,H,W,1] with 1=static, 0=dynamic
# Convert to BCHW for grid_sample
# means2d positions are in normalized [-1,1] screen space (same space grads use)
# We'll sample mask at those positions as weights in [0,1].
def sample_mask_at_means2d(means2d_xy: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    """
    means2d_xy: [..., 2] in [-1,1]
    returns: [...] in [0,1]
    """
    shp = means2d_xy.shape
    grid = means2d_xy.reshape(1, -1, 1, 2)           # [1, M, 1, 2]
    m_bchw = binary_mask.permute(0, 3, 1, 2)  # [1,1,H,W]
    w = F.grid_sample(m_bchw, grid, align_corners=True)  # [1,1,M,1]
    w = w.view(-1)                                   # [M]
    return w.reshape(*shp[:-1])                      # [...]

class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"


        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        #revised-0910
        self.mask_dir = cfg.mask_dir
        self.mask_dict = {}
        if self.mask_dir:
            src_root = Path(self.mask_dir)
            for mfile in src_root.rglob("*.*"):
                if mfile.is_file():
                    key = mfile.stem.replace("_extra", "").replace("_clutter", "")
                    self.mask_dict[key] = mfile

        #revised-0917
        self.pseudo_gt_dir = cfg.pseudo_gt_dir
        self.pseudo_gt_dict = {}
        if self.pseudo_gt_dir:
            src_root = Path(self.pseudo_gt_dir)
            for pfile in src_root.rglob("*.*"):
                if pfile.is_file():
                    key = pfile.stem.replace("_pseudo_gt", "")
                    self.pseudo_gt_dict[key] = pfile
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.semantics:
            self.parser = SemanticParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize,
                load_keyword=cfg.train_keyword,
                cluster=cfg.cluster,
            )
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize,
                test_every=cfg.test_every,
            )

        self.trainset = ClutterDataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            train_keyword=cfg.train_keyword,
            test_keyword=cfg.test_keyword,
            semantics=cfg.semantics,
        )


        self.valset = ClutterDataset(
            self.parser,
            split="test",
            train_keyword=cfg.train_keyword,
            test_keyword=cfg.test_keyword,
            semantics=False,
        )
        for idx in range(len(self.parser.image_names)):
            self.parser.image_names[idx] = self.parser.image_names[idx].replace("_clutter", "").replace("_extra", "")


        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
        self.spotless_optimizers = []
        self.mlp_spotless = cfg.semantics and not cfg.cluster
        if self.mlp_spotless:
            # currently using positional encoding of order 20 (4*20 = 80)
            self.spotless_module = SpotLessModule(
                num_classes=1, num_features=self.trainset[0]["semantics"].shape[0] + 80
            ).cuda()


            self.spotless_optimizers = [
                torch.optim.Adam(
                    self.spotless_module.parameters(),
                    lr=1e-3,
                )
            ]
            # self.spotless_loss = lambda p, minimum, maximum: torch.mean(
            #     torch.nn.ReLU()(p - minimum) + torch.nn.ReLU()(maximum - p))

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
#revised-1017
        # T-3DGS transient model initialization
        
        # Initialize DINO extractor and head before transient model
        self.dino_extractor = DinoFeatureExtractor(getattr(cfg, "dino_version", "dinov2_vits14"))
        self.dino_head = DinoUpsampleHead(self.dino_extractor.dino_model.embed_dim).to(self.device)

        self.transient_model = None
        self.transient_optimizer = None
        self.transient_scheduler = None

        # transient model initialization
        self.transient_model = ConvDPT(self.dino_extractor.dino_model.embed_dim).to(self.device)
        self.transient_model.train()
        self.transient_optimizer = torch.optim.Adam(self.transient_model.parameters(), lr=5e-3)
        self.transient_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.transient_optimizer, cfg.max_steps, 2e-4
        )
#------------------------------------------------------------------------------------------------------------
        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.float), #revised-0919/int2float
            "hist_err": torch.zeros((cfg.bin_size,)),
            "avg_err": 1.0,
            "lower_err": 0.0,
            "upper_err": 1.0,
            "sqrgrad": torch.zeros(n_gauss, device=self.device),
            "w_static": torch.zeros(n_gauss, device=self.device, dtype=torch.float), #revised-0921
            "w_dynamic": torch.zeros(n_gauss, device=self.device, dtype=torch.float),
            # Birth step per Gaussian: used to protect newly created Gaussians from immediate pruning. # revised-1028
            # dtype long to store training step indices.
            "birth_steps": torch.zeros(n_gauss, device=self.device, dtype=torch.long),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        #revised -1013
        packed = kwargs.pop("packed", self.cfg.packed)
        
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities= opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            # packed=self.cfg.packed,
            packed = packed,
            absgrad=self.cfg.absgrad,
            ubp=self.cfg.ubp,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        # # Debug: print image size and sample sizes of up to 10 splatted Gaussians
        # try:
        #     W = int(width)
        #     H = int(height)
        #     # determine candidate gaussian ids
        #     if "gaussian_ids" in info and info["gaussian_ids"] is not None:
        #         try:
        #             unique_ids = torch.unique(info["gaussian_ids"].long())
        #         except Exception:
        #             unique_ids = torch.arange(len(means), device=means.device)
        #     else:
        #         unique_ids = torch.arange(len(means), device=means.device)

        #     num_to_show = min(10, unique_ids.numel())
        #     sample_ids = unique_ids[:num_to_show].cpu().numpy().tolist()

        #     # try to extract radii (image-space) if available, otherwise fall back to world-space scales
        #     radii_info = info.get("radii", None)
        #     sizes = []
        #     if radii_info is not None and isinstance(radii_info, torch.Tensor):
        #         r = radii_info
        #         if r.dim() == 2:
        #             # [C, N] -> take max across channels
        #             per_gauss = r.max(dim=0).values
        #             for idx in sample_ids:
        #                 sizes.append(float(per_gauss[idx].cpu().item()))
        #         elif r.dim() == 1 and "gaussian_ids" in info:
        #             # per-fragment radii: aggregate by gaussian id
        #             gid = info["gaussian_ids"].long().view(-1)
        #             per_frag = r.view(-1)
        #             per_gauss = torch.zeros(len(means), device=means.device)
        #             per_gauss.index_add_(0, gid, per_frag)
        #             counts = torch.bincount(gid, minlength=len(means)).float().clamp_min(1.0)
        #             per_gauss = per_gauss / counts
        #             for idx in sample_ids:
        #                 sizes.append(float(per_gauss[idx].cpu().item()))
        #         else:
        #             for idx in sample_ids:
        #                 sizes.append(float(r[idx].cpu().item() if idx < r.numel() else 0.0))
        #     else:
        #         # fallback: use world-space scales (max of 3 axes)
        #         for idx in sample_ids:
        #             s = scales[idx].max().cpu().item()
        #             sizes.append(float(s))

        #     print(f"[DEBUG] Image WxH: {W}x{H}. Showing {num_to_show} Gaussian sizes (pixels or fallback world-units) for IDs {sample_ids}: {sizes}")
        # except Exception as e:
        #     print(f"[DEBUG] Failed to print gaussian sizes: {e}")
        # # Debug: print image size and sample sizes of up to 10 splatted Gaussians
        # try:
        #     W = int(width)
        #     H = int(height)
        #     # determine candidate gaussian ids
        #     if "gaussian_ids" in info and info["gaussian_ids"] is not None:
        #         try:
        #             unique_ids = torch.unique(info["gaussian_ids"].long())
        #         except Exception:
        #             unique_ids = torch.arange(len(means), device=means.device)
        #     else:
        #         unique_ids = torch.arange(len(means), device=means.device)

        #     num_to_show = min(10, unique_ids.numel())
        #     sample_ids = unique_ids[:num_to_show].cpu().numpy().tolist()

        #     # try to extract radii (image-space) if available, otherwise fall back to world-space scales
        #     radii_info = info.get("radii", None)
        #     sizes = []
        #     if radii_info is not None and isinstance(radii_info, torch.Tensor):
        #         r = radii_info
        #         if r.dim() == 2:
        #             # [C, N] -> take max across channels
        #             per_gauss = r.max(dim=0).values
        #             for idx in sample_ids:
        #                 sizes.append(float(per_gauss[idx].cpu().item()))
        #         elif r.dim() == 1 and "gaussian_ids" in info:
        #             # per-fragment radii: aggregate by gaussian id
        #             gid = info["gaussian_ids"].long().view(-1)
        #             per_frag = r.view(-1)
        #             per_gauss = torch.zeros(len(means), device=means.device)
        #             per_gauss.index_add_(0, gid, per_frag)
        #             counts = torch.bincount(gid, minlength=len(means)).float().clamp_min(1.0)
        #             per_gauss = per_gauss / counts
        #             for idx in sample_ids:
        #                 sizes.append(float(per_gauss[idx].cpu().item()))
        #         else:
        #             for idx in sample_ids:
        #                 sizes.append(float(r[idx].cpu().item() if idx < r.numel() else 0.0))
        #     else:
        #         # fallback: use world-space scales (max of 3 axes)
        #         for idx in sample_ids:
        #             s = scales[idx].max().cpu().item()
        #             sizes.append(float(s))

        #     print(f"[DEBUG] Image WxH: {W}x{H}. Showing {num_to_show} Gaussian sizes (pixels or fallback world-units) for IDs {sample_ids}: {sizes}")
        # except Exception as e:
        #     print(f"[DEBUG] Failed to print gaussian sizes: {e}")
        # means2d = info.get("means2d", None)
        # if means2d is not None and isinstance(means2d, torch.Tensor):
        #     # 만약 means2d가 normalized [-1,1]이라면:
        #     m = means2d.detach().cpu()  # shape [nnz, 2] 혹은 [N,2]
        #     # 보장: 2차원 배열로 변환
        #     try:
        #         m = m.reshape(-1, 2)
        #     except Exception:
        #         # 만약 이미 numpy라면 변환
        #         m = m
        #     # 예: normalized -> pixel coords (NumPy로 계산해 scalar int 확보)
        #     m_np = m.numpy()
        #     pix_x = ((m_np[:, 0] + 1.0) / 2.0 * (W - 1)).astype(int)
        #     pix_y = ((1.0 - (m_np[:, 1] + 1.0) / 2.0) * (H - 1)).astype(int)  # y 방향 주의
        #     # radii per gaussian (per_gauss 계산 필요). 안전하게 정의해 둠
        #     try:
        #         _per_gauss = None
        #         if "per_gauss" in locals():
        #             _per_gauss = per_gauss.detach().cpu().numpy()
        #         else:
        #             # fallback: use world-space scales (max of 3 axes)
        #             _per_gauss = scales.max(dim=-1).values.detach().cpu().numpy()
        #     except Exception:
        #         _per_gauss = None

        #     import cv2
        #     # Use the rendered image (render_colors) for visualization
        #     try:
        #         im = (render_colors[0].detach().cpu().numpy() * 255).astype(np.uint8)  # [H,W,3]
        #     except Exception:
        #         # fallback if render_colors is not as expected
        #         im = np.zeros((H, W, 3), dtype=np.uint8)

        #     # Save the raw rendered image for diagnosis
        #     try:
        #         raw_path = os.path.join(getattr(self.cfg, 'result_dir', '.') or '.', 'debug_render.png')
        #         cv2.imwrite(raw_path, im)
        #         print(f"[DEBUG] wrote raw render to {os.path.abspath(raw_path)}")
        #     except Exception as ee:
        #         print(f"[DEBUG] failed to write raw render: {ee}")
        #     H_img, W_img = im.shape[0], im.shape[1]
        #     for i, (x, y) in enumerate(zip(pix_x[:50], pix_y[:50])):  # 상위 50개만
        #         try:
        #             r = int(_per_gauss[i]) if (_per_gauss is not None and i < len(_per_gauss)) else 5
        #         except Exception:
        #             r = 5
        #         # 안전하게 int로 변환하고 이미지 경계 내로 클램프
        #         cx = int(x) if isinstance(x, (int, np.integer)) else int(x.item()) if hasattr(x, 'item') else int(x)
        #         cy = int(y) if isinstance(y, (int, np.integer)) else int(y.item()) if hasattr(y, 'item') else int(y)
        #         cx = max(0, min(W_img - 1, cx))
        #         cy = max(0, min(H_img - 1, cy))
        #         rr = max(1, int(r))
        #         # draw a filled circle with thicker border for visibility
        #         cv2.circle(im, (cx, cy), rr, (0, 255, 0), -1)
        #         cv2.circle(im, (cx, cy), rr + 1, (0, 0, 0), 1)
        #     # write to tmp (platform neutral path if possible)
        #     try:
        #         # ensure image is H x W x 3
        #         if im.ndim == 2:
        #             im_to_save = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        #         elif im.shape[2] == 4:
        #             im_to_save = im[:, :, :3]
        #         else:
        #             im_to_save = im

        #         out_dir = getattr(self.cfg, 'result_dir', '.') or '.'
        #         out_path = os.path.join(out_dir, f"debug_overlay.png")
        #         ok = cv2.imwrite(out_path, im_to_save)
        #         print(f"[DEBUG] wrote overlay to {os.path.abspath(out_path)} success={ok}")
        #     except Exception as ee:
        #         print(f"[DEBUG] failed to write overlay: {ee}")
        #     # Print diagnostic info about radii/means2d to help root-cause the all-zero sizes
        #     try:
        #         print('[DEBUG] info keys:', list(info.keys()))
        #         if 'radii' in info and isinstance(info['radii'], torch.Tensor):
        #             r = info['radii']
        #             try:
        #                 print('[DEBUG] radii shape:', r.shape, 'min/max:', float(r.min()), float(r.max()))
        #             except Exception:
        #                 print('[DEBUG] radii sample (flatten):', r.view(-1)[:20])
        #         if 'gaussian_ids' in info and isinstance(info['gaussian_ids'], torch.Tensor):
        #             gid = info['gaussian_ids'].long().view(-1)
        #             print('[DEBUG] gaussian_ids shape:', gid.shape, 'unique count sample:', torch.unique(gid)[:20])
        #         if 'means2d' in info and isinstance(info['means2d'], torch.Tensor):
        #             m = info['means2d'].detach().cpu()
        #             try:
        #                 print('[DEBUG] means2d sample:', m.reshape(-1,2)[:10])
        #             except Exception:
        #                 print('[DEBUG] means2d sample (raw):', m[:10])
        #         # world-space scales
        #         try:
        #             print('[DEBUG] world scales sample:', torch.exp(self.splats['scales']).max(dim=-1).values[:10])
        #         except Exception:
        #             pass
        #     except Exception as e:
        #         print('[DEBUG] failed to print diagnostics:', e)


        return render_colors, render_alphas, info
    
    def robust_mask(
        self, error_per_pixel: torch.Tensor, loss_threshold: float
    ) -> torch.Tensor:
        epsilon = 1e-3
        error_per_pixel = error_per_pixel.mean(axis=-1, keepdims=True)
        error_per_pixel = error_per_pixel.squeeze(-1).unsqueeze(0)
        is_inlier_pixel = (error_per_pixel < loss_threshold).float()
        window_size = 3
        channel = 1
        window = torch.ones((1, 1, window_size, window_size), dtype=torch.float) / (
            window_size * window_size
        )
        if error_per_pixel.is_cuda:
            window = window.cuda(error_per_pixel.get_device())
        window = window.type_as(error_per_pixel)
        has_inlier_neighbors = F.conv2d(
            is_inlier_pixel, window, padding=window_size // 2, groups=channel
        )
        has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()
        is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > epsilon).float()
        pred_mask = is_inlier_pixel.squeeze(0).unsqueeze(-1)
        return pred_mask

    def robust_cluster_mask(self, inlier_sf, semantics):
        inlier_sf = inlier_sf.squeeze(-1).unsqueeze(0)
        cluster_size = torch.sum(
            semantics, axis=[-1, -2], keepdims=True, dtype=torch.float
        )
        inlier_cluster_size = torch.sum(
            inlier_sf * semantics, axis=[-1, -2], keepdims=True, dtype=torch.float
        )
        cluster_inlier_percentage = (inlier_cluster_size / cluster_size).float()
        is_inlier_cluster = (cluster_inlier_percentage > 0.5).float()
        inlier_sf = torch.sum(
            semantics * is_inlier_cluster, axis=1, keepdims=True, dtype=torch.float
        )
        pred_mask = inlier_sf.squeeze(0).unsqueeze(-1)
        return pred_mask

    #revised
    def get_ssim_lambda(self, step:int) -> float:
                cfg = self.cfg
                # ramp_end = int(0.5 * cfg.max_steps) 
                # t = min(step / max(ramp_end, 1), 1.0)
                # wt = 0.5 - 0.5 * math.cos(math.pi * t) 
                # # return wt * cfg.ssim_lambda
                # return wt

                cfg = self.cfg
                ramp_start = cfg.min_steps
                ramp_end = int(0.5 * cfg.max_steps)
                if step < ramp_start:
                    return 0.0
                t = min((step - ramp_start) / max(ramp_end - ramp_start, 1), 1.0)
                wt = 0.5 - 0.5 * math.cos(math.pi * t)
                return wt

    
    
    #revised
    def masked_ssim(self, img1, img2, mask, window_size=11, C1=0.01**2, C2=0.03**2):
        """
        img1,img2: [B,3,H,W] in [0,1]
        mask     : [B,1,H,W] with 1=valid(background), 0=ignore(dynamic)
        """
        B, C, H, W = img1.shape
        # per-channel uniform window
        win = torch.ones((C,1,window_size,window_size), device=img1.device) / (window_size**2)

        def filt(x):  # groups=C conv
            return F.conv2d(x, win, padding=window_size//2, groups=C)

        m  = mask.repeat(1, C, 1, 1)  # [B,1,H,W] is expanded to [B,3,H,W]
        M  = torch.clamp(filt(m), min=1e-6)

        mu1 = filt(img1 * m) / M
        mu2 = filt(img2 * m) / M
        sigma1_sq = filt((img1*m)**2) / M - mu1**2
        sigma2_sq = filt((img2*m)**2) / M - mu2**2
        sigma12   = filt((img1*img2*m)) / M - mu1*mu2

        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = (ssim_map * M).sum(dim=(1,2,3)) / (M.sum(dim=(1,2,3)) + 1e-8)
        return ssim.mean()   # scalar

    def masked_psnr(self, pred, gt, mask, max_val=1.0):
        """
        to only calculate activated pixels with psnr, cannot use pre-built psnr func
        because there can be include 0 in denominator
        """
        m = mask.expand_as(pred) 
        mse = ((pred - gt)**2 * m).sum(dim=(1,2,3)) / (m.sum(dim=(1,2,3)) + 1e-8)
        psnr = 10.0 * torch.log10((max_val**2) / (mse + 1e-12))
        return psnr.mean()


# revised-1013
    def _uap_noise_ratio(self, step: int) -> float:
        """
        섭동 강도 스케줄: uap_noise_init -> uap_noise_final 로 선형 점감.
        """
        T = max(self.cfg.uap_noise_anneal_end, 1)
        t = min(step / T, 1.0)
        # print("uap noise ration fucn works")
        return self.cfg.uap_noise_init + (self.cfg.uap_noise_final - self.cfg.uap_noise_init) * t
    

    @torch.no_grad()
    def _make_pseudo_views(self, camtoworlds: Tensor, Ks: Tensor, K: int) -> Tuple[Tensor, Tensor]:
        """
        소각도 SE(3) jitter로 fallback 생성
        반환: (C2W[K,4,4], K[K,3,3])
        """
        device = camtoworlds.device
        c2w0 = camtoworlds.repeat(K, 1, 1).clone()
        rad = math.pi / 180.0
        ang = torch.randn(K, 3, device=device) * (self.cfg.se_jitter_deg * rad)
        trans = torch.randn(K, 3, device=device) * (self.cfg.se_jitter_trans * self.scene_scale)

        def Rxyz(ax, ay, az):
            Rx = torch.tensor([[1, 0, 0], [0, torch.cos(ax), -torch.sin(ax)], [0, torch.sin(ax), torch.cos(ax)]], device=device)
            Ry = torch.tensor([[torch.cos(ay), 0, torch.sin(ay)], [0, 1, 0], [-torch.sin(ay), 0, torch.cos(ay)]], device=device)
            Rz = torch.tensor([[torch.cos(az), -torch.sin(az), 0], [torch.sin(az), torch.cos(az), 0], [0, 0, 1]], device=device)
            return Rz @ Ry @ Rx

        for i in range(K):
            R = Rxyz(*ang[i])
            c2w0[i, :3, :3] = R @ c2w0[i, :3, :3]
            c2w0[i, :3, 3] = c2w0[i, :3, 3] + trans[i]
      
        # print("make pseudo view with manual jittering works")
        return c2w0, Ks.repeat(K, 1, 1)

    @torch.no_grad()
    def _clone_splats(self) -> torch.nn.ParameterDict:
        """Δ-브랜치용 임시 파라미터 사본(optimizer 연결 없음)."""
        clone = torch.nn.ParameterDict()
        for k, v in self.splats.items():
            clone[k] = torch.nn.Parameter(v.detach().clone(), requires_grad=True if k in ["means3d","scales","quats","opacities","shN","sh0","colors","features"] else False)
        return clone

    from contextlib import contextmanager
    @contextmanager
    def _temporary_splats(self, temp_splats: torch.nn.ParameterDict):
        """일시적으로 self.splats를 교체해 렌더링."""
        orig = self.splats
        try:
            self.splats = temp_splats
            yield
        finally:
            self.splats = orig


    
    # gpt made function
    @torch.no_grad()
    def _find_reset_gaussians_by_mask(
        self,
        dyn_mask_bhw: torch.Tensor,  # [B,1,H,W] or [1,H,W]
        info: dict,
        cameras=None,                # list[int] or tensor of indices/extrinsics
        overlap_thresh: float = None,
    ) -> torch.Tensor:
        dev = dyn_mask_bhw.device
        if overlap_thresh is None:
            overlap_thresh = getattr(self.cfg, "uap_dyn_overlap_frac", 0.15)

        N = len(self.splats["means3d"])
        reset_mask = torch.zeros(N, dtype=torch.bool, device=dev)

        # ---- unpack packed outputs ----
        gaussian_ids = info.get("gaussian_ids", None)      # [nnz]
        means2d     = info.get("means2d", None)            # [nnz,2]
        camera_ids  = info.get("camera_ids", None)         # [nnz] (optional)
        H = int(info.get("height",  dyn_mask_bhw.shape[-2]))
        W = int(info.get("width",   dyn_mask_bhw.shape[-1]))

        # ---- normalize dyn mask to [B,1,H,W] & build dynamic=1 ----
        if dyn_mask_bhw.dim() == 3:
            dyn_mask_bhw = dyn_mask_bhw.unsqueeze(0)  # [1,1,H,W] or [1,H,W] -> [1,1,H,W]
        if dyn_mask_bhw.shape[1] != 1:                # [B,H,W] -> [B,1,H,W]
            dyn_mask_bhw = dyn_mask_bhw.unsqueeze(1)
        # binary: 1=static, 0=dynamic  -> dynamic=1
        dyn_mask = (dyn_mask_bhw < 0.5).float()       # [B,1,H,W]
        B = dyn_mask.shape[0]

        # ---- quick sanity prints ----
        print(f"[SE-COREG DEBUG] packed nnz={None if gaussian_ids is None else int(gaussian_ids.numel())}, "
            f"HxW={H}x{W}, B={B}, overlap_thresh={overlap_thresh}")

        if (gaussian_ids is None) or (means2d is None) or (gaussian_ids.numel() == 0):
            print("[SE-COREG WARN] missing packed fields → skip reset")
            return reset_mask

        # ---- CAMERA FILTERING ----
        # Determine which camera indices we should keep for this call
        # 1) if 'cameras' provided as indices: use them
        # 2) else if B==1 and camera_ids present: keep dominant cam (mode)
        # 3) else: keep all
        keep_mask = torch.ones_like(gaussian_ids, dtype=torch.bool)
        if camera_ids is not None:
            if cameras is not None:
                if torch.is_tensor(cameras) and cameras.dtype in (torch.int32, torch.int64):
                    cam_set = set(cameras.view(-1).tolist())
                elif isinstance(cameras, (list, tuple)) and len(cameras) > 0 and isinstance(cameras[0], int):
                    cam_set = set(cameras)
                else:
                    # cameras might be extrinsics; fallback to "current batch views"
                    cam_set = set()
            else:
                cam_set = set()

            if len(cam_set) > 0:
                keep_mask = torch.isin(camera_ids, torch.tensor(sorted(cam_set), device=dev))
            elif B == 1:
                # choose dominant camera in fragments as the "current view"
                # (this matches single-view mask most of the time)
                uniq, counts = torch.unique(camera_ids, return_counts=True)
                dominant = uniq[torch.argmax(counts)].item()
                keep_mask = (camera_ids == dominant)
                print(f"[SE-COREG DEBUG] dominant_camera={dominant} (kept {int(keep_mask.sum())} frags)")
            else:
                print("[SE-COREG DEBUG] multi-view batch without 'cameras' indices: keeping all fragments")

        # filter packed arrays
        gid_kept = gaussian_ids[keep_mask]                 # [nnz_kept]
        m2d_kept = means2d[keep_mask]                      # [nnz_kept, 2]
        nnz_kept = int(gid_kept.numel())
        if nnz_kept == 0:
            print("[SE-COREG DEBUG] no fragments kept after camera filter")
            return reset_mask

        # ---- COORDINATE SYSTEM HEURISTIC ----
        # auto-detect whether means2d are normalized or pixel
        mabs = m2d_kept.abs().max().item()
        if mabs <= 2.5:
            # assume [-1,1] normalized
            xs = (m2d_kept[:, 0] * 0.5 + 0.5) * (W - 1)
            ys = (m2d_kept[:, 1] * 0.5 + 0.5) * (H - 1)
            coord_mode = "norm"
        else:
            # assume already pixel coordinates
            xs = m2d_kept[:, 0]
            ys = m2d_kept[:, 1]
            coord_mode = "pixel"

        # clamp & integer sample
        xi = xs.clamp(0, W - 1).round().long()
        yi = ys.clamp(0, H - 1).round().long()

        # ---- pick correct mask view ----
        # If we selected dominant camera (B==1), use mask[0]; else best-effort use mask[0]
        dyn_view = dyn_mask[0, 0]  # [H,W]
        dyn_vals = dyn_view[yi, xi].float()  # [nnz_kept], 1 if dynamic region, else 0

        # ---- aggregate per Gaussian ----
        total_counts = torch.bincount(gid_kept, minlength=N).float()
        dyn_counts   = torch.bincount(gid_kept, weights=dyn_vals, minlength=N).float()
        frac = dyn_counts / (total_counts + 1e-6)
        over = frac > overlap_thresh
        reset_mask |= over

        # ---- DEBUG STATS ----
        touched = (total_counts > 0).sum().item()
        dyn_frag_ratio = (dyn_vals > 0.5).float().mean().item() if nnz_kept > 0 else 0.0
        print(f"[SE-COREG DEBUG] coord_mode={coord_mode}, kept_frags={nnz_kept}, "
            f"touched_gauss={touched}, dyn_frag_ratio={dyn_frag_ratio:.4f}, "
            f"max_frac={frac.max().item():.4f}, mean_frac={(frac[total_counts>0].mean().item() if touched>0 else 0.0):.4f}, "
            f"num_over={over.sum().item()}")

        return reset_mask





    @torch.no_grad()
    def _apply_reset_attributes(self, reset_mask: torch.Tensor, noise_ratio: float):
        """
        선택된 가우시안들에 경량 섭동을 인플레이스로 적용.
        - means3d: scene-scale 기반 소노이즈
        - scales : log-space 소노이즈
        - shN    : (존재 시) 하이주파 감쇄
        - opacities: 상한 클램프(폭주 방지)
        """
        if reset_mask.sum() == 0:
            return
        dev = self.device
        idx = torch.where(reset_mask)[0]
        n = idx.numel()
        # 위치/스케일 섭동
        self.splats["means3d"].data[idx] += torch.randn(n,3,device=dev) * (self.scene_scale * noise_ratio)
        self.splats["scales"].data[idx]  += torch.clamp(torch.randn(n,3,device=dev) * 0.1 * noise_ratio, -0.5, 0.5)
        # SH 하이주파 감쇄
        if "shN" in self.splats:
            self.splats["shN"].data[idx] *= 0.0
        # opacity 상한 클램프
        if "opacities" in self.splats:
            mx = torch.logit(torch.tensor(self.cfg.init_opa, device=dev)).item()
            self.splats["opacities"].data[idx] = torch.clamp(self.splats["opacities"].data[idx], max=mx)
#-------------------------------------------------------------------------------------------------------------------------
#revised-1015
    @torch.no_grad()
    def _decay_opacity(self, mask: torch.Tensor, factor: float):
        """
        Multiply opacity α by `factor` for gaussians where mask==True.
        Safe in-place update; keeps optimizer state.
        """
        if mask is None or mask.numel() == 0:
            return
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.any():
            opa = self.splats["opacities"].data
            alpha = torch.sigmoid(opa)
            alpha[mask] = torch.clamp(alpha[mask] * factor, 1e-6, 1 - 1e-6)
            self.splats["opacities"].data = torch.logit(alpha)
#revised-1016-------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _detect_depth_inconsistent_gaussians(
        self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int, 
        dyn_mask: Tensor, depth_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Depth-based floater detection:
        렌더링된 깊이와 동적 마스크를 비교하여 배경 앞에 떠있는 floater를 감지
        
        Args:
            camtoworlds: [1, 4, 4]
            Ks: [1, 3, 3]
            width, height: image dimensions
            dyn_mask: [1,H,W] or [1,H,W,1], 1=static, 0=dynamic
            depth_threshold: 깊이 불일치 임계값 (normalized depth 기준)
        
        Returns:
            [N] boolean mask indicating floaters
        """
        device = self.device
        N = len(self.splats["means3d"])
        
        # Render depth map
        renders, _, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+ED",
            packed=True
        )
        
        if renders.shape[-1] < 4:
            return torch.zeros(N, dtype=torch.bool, device=device)
        
        depth_map = renders[..., 3:4]  # [1, H, W, 1]
        
        # Normalize depth
        valid_depth = depth_map[depth_map > 0]
        if valid_depth.numel() == 0:
            return torch.zeros(N, dtype=torch.bool, device=device)
        
        depth_min, depth_max = valid_depth.min(), valid_depth.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        
        # dyn_mask 차원 맞추기
        if dyn_mask.dim() == 4:
            dyn_mask = dyn_mask.squeeze(-1)  # [1,H,W]
        
        # 동적 영역(0)에서의 깊이 통계
        dyn_region = (dyn_mask[0] < 0.5)  # [H,W]
        stat_region = (dyn_mask[0] >= 0.5)  # [H,W]
        
        if not dyn_region.any() or not stat_region.any():
            return torch.zeros(N, dtype=torch.bool, device=device)
        
        # 주변 정적 영역의 평균 깊이 계산 (morphological dilation)
        kernel_size = 15
        dyn_dilated = F.max_pool2d(
            dyn_region.float().unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=kernel_size//2
        ).squeeze()
        
        # 동적 영역 주변의 정적 영역
        border_region = (dyn_dilated > 0.5) & stat_region
        
        if border_region.any():
            border_depth_mean = depth_normalized[0, border_region].mean()
            
            # 각 Gaussian의 깊이와 비교
            means3d = self.splats["means3d"]  # [N, 3]
            cam_pos = camtoworlds[0, :3, 3]  # [3]
            
            # Gaussian의 카메라로부터의 거리
            gauss_depth = torch.norm(means3d - cam_pos[None, :], dim=1)  # [N]
            gauss_depth_normalized = (gauss_depth - gauss_depth.min()) / (gauss_depth.max() - gauss_depth.min() + 1e-6)
            
            # 동적 마스크 영역에 주로 나타나면서 주변 배경보다 앞에 있는 Gaussian 감지
            if hasattr(self.running_stats, 'w_dynamic') and hasattr(self.running_stats, 'w_static'):
                w_dyn = self.running_stats["w_dynamic"]
                w_stat = self.running_stats["w_static"]
                total_w = w_dyn + w_stat + 1e-6
                dynamic_ratio = w_dyn / total_w
                
                # 동적 영역에 자주 나타나고(>0.5), 배경보다 앞에 있는 Gaussian
                is_in_dynamic = dynamic_ratio > 0.5
                is_in_front = gauss_depth_normalized < (border_depth_mean - depth_threshold)
                
                floater_mask = is_in_dynamic & is_in_front
                return floater_mask
        print("depth inconsistent gaussians detection works")
        return torch.zeros(N, dtype=torch.bool, device=device)

    @torch.no_grad()
    def _inpaint_dynamic_regions(
        self, binary_mask: torch.Tensor, camtoworlds: Tensor, inpaint_strength: float
    ) -> int:
        """
        동적 영역의 "구멍"을 주변 배경 Gaussian으로 채우기 (inpainting)
        
        전략: 동적 영역 경계 근처의 정적 Gaussian을 복제하여 구멍을 메움
        
        Args:
            binary_mask: [1,H,W] or [1,H,W,1], 1=static, 0=dynamic
            camtoworlds: [1, 4, 4]
            inpaint_strength: 복제할 Gaussian 비율
        
        Returns:
            복제된 Gaussian 개수
        """
        device = self.device
        
        # binary_mask 차원 정규화
        if binary_mask.dim() == 4:
            binary_mask = binary_mask.squeeze(-1)  # [1,H,W]
        
        # 동적 영역과 정적 영역 경계 찾기
        dyn_region = (binary_mask[0] < 0.5).float()  # [H,W]
        
        # 동적 영역 경계 (erosion + dilation 차이)
        kernel_size = 7
        dyn_eroded = -F.max_pool2d(
            -dyn_region.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=kernel_size//2
        ).squeeze()
        dyn_dilated = F.max_pool2d(
            dyn_region.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=kernel_size//2
        ).squeeze()
        
        # 경계 영역 (동적 영역 주변)
        border_region = (dyn_dilated > 0.5) & (dyn_eroded < 0.5)
        
        # 카메라 위치
        cam_pos = camtoworlds[0, :3, 3]  # [3]
        means3d = self.splats["means3d"]  # [N, 3]
        
        # 각 Gaussian을 카메라 평면에 투영하여 경계 영역 근처인지 확인
        # (간단한 근사: 3D 거리로 대체)
        
        # 경계 영역의 중심 계산 (world space)
        border_pixels = torch.where(border_region)
        if len(border_pixels[0]) == 0:
            return 0
        
        # 정적 영역의 Gaussian 중 경계 근처에 있는 것 찾기
        if hasattr(self.running_stats, 'w_static') and hasattr(self.running_stats, 'w_dynamic'):
            w_stat = self.running_stats["w_static"]
            w_dyn = self.running_stats["w_dynamic"]
            total_w = w_stat + w_dyn + 1e-6
            static_ratio = w_stat / total_w
            
            # 주로 정적 영역에 나타나는 Gaussian (>80%)
            is_background = static_ratio > 0.6
            
            # 이 중 일부를 복제하여 동적 영역으로 확장
            bg_indices = torch.where(is_background)[0]
            if bg_indices.numel() == 0:
                return 0
            
            # 무작위로 일부 선택
            n_to_copy = int(bg_indices.numel() * inpaint_strength)
            if n_to_copy == 0:
                return 0
            
            perm = torch.randperm(bg_indices.numel(), device=device)
            to_copy = bg_indices[perm[:n_to_copy]]
            
            # 분할(splitting): 선택한 배경 가우시안을 복제하고,
            # 원본과 복제본 모두에 가우시안 노이즈를 주입해 동적 영역으로 확장
            N_before = len(means3d)
            copy_mask = torch.zeros(N_before, dtype=torch.bool, device=device)
            copy_mask[to_copy] = True
            self.refine_duplicate(copy_mask)

            # 새로 생성된 가우시안 인덱스 계산
            N_after = len(self.splats["means3d"])  # 업데이트된 길이
            new_created = N_after - N_before
            if new_created <= 0:
                return 0
            # 일반적으로 refine_duplicate는 선택된 각 항목을 1개씩 복제한다고 가정
            new_indices = torch.arange(N_after - new_created, N_after, device=device)

            # 원본+복제본 모두에 노이즈 주입
            cfg = self.cfg
            idx_all = torch.cat([to_copy, new_indices])

            # 노이즈 스케일(안전한 기본값 제공, cfg에서 덮어쓰기 가능)
            pos_sigma   = float(getattr(cfg, "split_pos_noise", 0.01)) * float(self.scene_scale)
            scale_sigma = float(getattr(cfg, "split_scale_noise", 0.05))
            quat_sigma  = float(getattr(cfg, "split_quat_noise", 0.02))
            alpha_sigma = float(getattr(cfg, "split_alpha_noise", 0.02))
            sh_sigma    = float(getattr(cfg, "split_sh_noise", 0.02))
            feat_sigma  = float(getattr(cfg, "split_feature_noise", 0.01))

            # means3d: 위치 노이즈
            if "means3d" in self.splats:
                m = self.splats["means3d"].data
                m[idx_all] = m[idx_all] + torch.randn_like(m[idx_all]) * pos_sigma

            # scales: log-space 노이즈(과도한 폭주 방지 위해 클램프)
            if "scales" in self.splats:
                s = self.splats["scales"].data
                s[idx_all] = s[idx_all] + torch.randn_like(s[idx_all]) * scale_sigma
                s[idx_all] = torch.clamp(s[idx_all], min=-5.0, max=5.0)  # exp 후 합리적 범위 유지를 위한 로그 스페이스 클램프

            # quats: 소규모 노이즈 후 정규화
            if "quats" in self.splats:
                q = self.splats["quats"].data
                q[idx_all] = q[idx_all] + torch.randn_like(q[idx_all]) * quat_sigma
                q[idx_all] = q[idx_all] / (q[idx_all].norm(dim=-1, keepdim=True) + 1e-8)

            # opacities: 알파 도메인에서 노이즈 → 다시 로그잇으로
            if "opacities" in self.splats:
                opa = self.splats["opacities"].data
                alpha = torch.sigmoid(opa[idx_all])
                alpha = alpha + torch.randn_like(alpha) * alpha_sigma
                alpha = torch.clamp(alpha, 1e-6, 1 - 1e-6)
                opa[idx_all] = torch.logit(alpha)

            # SH 계수 또는 색/특징에 노이즈
            if "sh0" in self.splats:
                sh0 = self.splats["sh0"].data
                sh0[idx_all] = sh0[idx_all] + torch.randn_like(sh0[idx_all]) * sh_sigma
            if "shN" in self.splats:
                shN = self.splats["shN"].data
                shN[idx_all] = shN[idx_all] + torch.randn_like(shN[idx_all]) * sh_sigma
            if "colors" in self.splats:
                cols = self.splats["colors"].data
                cols[idx_all] = cols[idx_all] + torch.randn_like(cols[idx_all]) * sh_sigma
            if "features" in self.splats:
                feats = self.splats["features"].data
                feats[idx_all] = feats[idx_all] + torch.randn_like(feats[idx_all]) * feat_sigma

            # 실제로 분할된 개수 반환(보호적으로 계산)
            return int(min(n_to_copy, new_created))
        
        # If none of the inpainting branches ran (for example if running
        # statistics keys are missing), ensure an integer is returned so the
        # caller can safely compare the result (avoid NoneType > int error).
        return 0

#-------------------------------------------------------------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        device = self.device

        # train용 GT/Render 저장 폴더 revised
        
        train_dir = os.path.join(cfg.result_dir,"train")
        method_train_dir = os.path.join(train_dir, "ours_30000")
        gt_train_dir = os.path.join(method_train_dir, "gt")
        rend_train_dir = os.path.join(method_train_dir, "renders")
        os.makedirs(gt_train_dir, exist_ok=True)
        os.makedirs(rend_train_dir, exist_ok=True)
        
        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            # revised-1028    
            # expose current training step to other methods (refine_*),
            # so newly created Gaussians can record a birth step.
            self.current_step = step

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]
            
            

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            filename = data["filename"]
            if isinstance(filename, list):
                filename = filename[0]
            base = Path(filename).stem
            name = base.replace("_extra", "").replace("_clutter", "") + ".png"
            mask_name = base.replace("_extra", "").replace("_clutter", "")

            #binary mask init - MOVED BEFORE rasterize_splats
            binary_mask = None
            
            if self.mask_dir:
                mpath = self.mask_dict.get(mask_name, None)
                if mpath:
                    # mask load & process
                    mask_img = imageio.imread(mpath)
                    if mask_img.ndim == 3:
                        print("[WARN] mask image has 3 channels, converting to grayscale")
                        # mask_img = mask_img[..., 0]
                    orig_imsize = self.parser.imsize_dict.get(name)
                    if orig_imsize is not None:
                        h0, w0 = orig_imsize
                        factor = self.cfg.data_factor
                        h1, w1 = h0 // factor, w0 // factor
                    else:
                        h1, w1 = height, width

                    mask_img_resized = cv2.resize(mask_img, (w1, h1), interpolation=cv2.INTER_NEAREST)
                    # dilation to enlarge the mask
                    kernel = np.ones((5, 5), np.uint8) 
                    mask_img_resized = cv2.dilate(mask_img_resized, kernel, iterations=1)

                    mask_tensor = torch.from_numpy(mask_img_resized).float() / 255.0
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)
                    # Additional resize if needed to match exact render resolution
                    if mask_tensor.shape[1] != height or mask_tensor.shape[2] != width:
                        mask_tensor = F.interpolate(mask_tensor.permute(0, 3, 1, 2), size=(height, width), mode='nearest')
                        mask_tensor = mask_tensor.permute(0, 2, 3, 1) # [1,H,W,1]
                    binary_mask = 1.0 -  mask_tensor.to(device) # 1 for background, 0 for dynamic object[1,H,W,1]

            # train forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )


            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
           
            colors = torch.clamp(colors, 0., 1.0)

            #revised
            # save train GT/Render images
            gt_np = (pixels.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            rend_np = (colors.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

            # save images
            imageio.imwrite(os.path.join(gt_train_dir,  name), gt_np)
            imageio.imwrite(os.path.join(rend_train_dir, name), rend_np)

            # Copy mask file for saving (if mask was loaded)
            if self.mask_dir and binary_mask is not None:
                mpath = self.mask_dict.get(mask_name, None)
                if mpath:
                    mdst_dir = Path(self.cfg.result_dir) / "train" / "mask"
                    mdst_dir.mkdir(parents=True, exist_ok=True)
                    mdst = mdst_dir / mpath.name
                    shutil.copy(mpath, mdst)

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            info["means2d"].retain_grad()  # used for running stats

            rgb_pred_mask = None 

            # loss
            if cfg.loss_type == "l1":
                rgbloss = F.l1_loss(colors, pixels)
            else:
                # robust loss
                error_per_pixel = torch.abs(colors - pixels)
            
            # revised
            #--------------------------------------------------------------------------------------------------------------------------------------------
            if cfg.dino_loss_flag:
                if not hasattr(self, "dino_extractor"):
                    self.dino_extractor = DinoFeatureExtractor(getattr(cfg, "dino_version", "dinov2_vits14"))
                    # freeze DINO backbone parameters so we do not train the DINO model
                    try:
                        self.dino_extractor.dino_model.eval()
                        for p in self.dino_extractor.dino_model.parameters():
                            p.requires_grad = False
                    except Exception:
                        # defensive: some extractor implementations may differ
                        pass
                if not hasattr(self, "dino_head"):
                    # DINO 토큰(저해상도) -> 픽셀 해상도로 올리는 경량 업샘플 헤드
                    self.dino_head = DinoUpsampleHead(self.dino_extractor.dino_model.embed_dim).to(self.device)
                    # we don't want to train the upsample head either (only splats should be updated)
                    for p in self.dino_head.parameters():
                        p.requires_grad = False
                    self.dino_head.eval()

                # self hard coded control
                mask_adaptation = False

                # 1) DINO 토큰 추출 (GT/Render)
                gt_chw     = pixels.permute(0, 3, 1, 2)    # [B,3,H,W], GT
                render_chw = colors.permute(0, 3, 1, 2)    # [B,3,H,W], 렌더 결과

                if mask_adaptation and (binary_mask is not None):
                    mask_chw = binary_mask.permute(0, 3, 1, 2)   # [B,1,H,W], 마스크
                    render_chw_safe = render_chw * mask_chw + render_chw.detach() * (1.0 - mask_chw) # 마스크 영역은 렌더링 그래디언트가 안 흐르도록 처리

                # Extract GT tokens without gradient to save memory
                with torch.no_grad():
                    ftok,  Th, Tw, _, _  = self.dino_extractor.extract_tokens(gt_chw)

                # Extract render tokens with gradient enabled so dino_loss can backprop to render pixels -> splats
                if mask_adaptation and (binary_mask is not None):
                    fhtok, Th2, Tw2, _, _  = self.dino_extractor.extract_tokens(render_chw_safe)
                else:
                    fhtok, Th2, Tw2, _, _ = self.dino_extractor.extract_tokens(render_chw)

                assert (Th == Th2) and (Tw == Tw2), "DINO token grid mismatch between GT and Render"
                B, _, C = ftok.shape

                f_gt  = ftok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()   # [B,C,Th,Tw]
                f_rnd = fhtok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()  # [B,C,Th,Tw]

                # 2) 업샘플 → 이미지 해상도 정렬
                ps = self.dino_extractor.patch_size
                Hpad, Wpad = Th * ps, Tw * ps

                f_gt_up_pad  = F.interpolate(self.dino_head(f_gt),  size=(Hpad, Wpad), mode="bilinear", align_corners=False)
                f_rnd_up_pad = F.interpolate(self.dino_head(f_rnd), size=(Hpad, Wpad), mode="bilinear", align_corners=False)

                # 오른쪽/아래만 패딩했으므로 좌상단 기준으로 원 해상도(H,W)로 자르기
                H, W = colors.shape[1], colors.shape[2]
                f_gt_up  = f_gt_up_pad[:,  :, :H, :W]   # [B,C,H,W]
                f_rnd_up = f_rnd_up_pad[:, :, :H, :W]   # [B,C,H,W]
                
                # 3) 코사인 불일치(=의미 차이) 맵 -> 배경(inlier) 영역에서만 정합 유도
                cosd = 1.0 - F.cosine_similarity(f_gt_up, f_rnd_up, dim=1, eps=1e-6).unsqueeze(1)  # [B,1,H,W]       
                if mask_adaptation and (binary_mask is not None):
                    dino_feat_loss = (cosd * mask_chw).sum() / (mask_chw.sum() + 1e-8)
                else:
                    dino_feat_loss = (cosd).sum() / (binary_mask.sum() + 1e-8)  # [B,1,H,W]

                dino_loss =  dino_feat_loss
                
            else:
                dino_loss = 0.0
            #--------------------------------------------------------------------------------------------------------------------------------------------
#revised-1017
            # this is not used do to unstable error backpropagation
            # Todo: transient loss implementation

            transient_loss = torch.tensor(0.0, device=device)

            if cfg.transient_training_enable and binary_mask is not None and cfg.transient_loss_start_iter <= step and step % cfg.refine_every ==0:  # transient 활성화 조건
                print("Transient loss computation is activated.")
                # 1) 입력 텐서 준비 (DETACH): ConvDPT만 학습되도록 렌더 경로 차단
                gt_chw_det     = gt_chw.detach()       # [B,3,H,W]
                render_chw_det = render_chw.detach()   # [B,3,H,W]
                transient_input = torch.cat((gt_chw_det, render_chw_det), dim=0)  # [2*B,3,H,W]

                # 2) 공분산 파라미터 예측
                # DINO extractor는 가중치 갱신/그래프 연결 불필요 → no_grad
                with torch.no_grad():
                    _ = self.dino_extractor  # 일부 구현에서 내부 상태 접근을 위해 참조 필요
                transient_out = self.transient_model(transient_input, self.dino_extractor)

                # transient_utils.ConvDPT returns Tensor [2*B, 3, H, W]
                # convdpt_module.ConvDPT returns tuple/list: ([gt_sigma1,gt_sigma2,gt_rho],[s1,s2,r])
                if isinstance(transient_out, torch.Tensor):
                    cov = transient_out
                    B2 = gt_chw_det.shape[0]
                    cov_gt = cov[:B2]
                    cov_render = cov[B2 : B2 * 2]

                    sigma_1 = cov_gt[:, 0]
                    sigma_2 = cov_gt[:, 1]
                    rho     = cov_gt[:, 2]

                    s_1 = cov_render[:, 0]
                    s_2 = cov_render[:, 1]
                    r   = cov_render[:, 2]
                else:
                    transient_maps = transient_out
                    sigma_1 = transient_maps[0][0]  # [B, H, W]
                    sigma_2 = transient_maps[0][1]  # [B, H, W]  
                    rho     = transient_maps[0][2]  # [B, H, W]

                    s_1 = transient_maps[1][0]  # [B, H, W]
                    s_2 = transient_maps[1][1]  # [B, H, W]
                    r   = transient_maps[1][2]  # [B, H, W]
                
                # 수치 안전화 (양쪽 determinant/분산 모두 clamp)
                eps = 1e-6
                det        = torch.clamp(sigma_1.square() * sigma_2.square() * (1.0 - rho.square()), min=eps)
                det_render = torch.clamp(s_1.square()     * s_2.square()     * (1.0 - r.square()  ), min=eps)
 
                 # 4) KL divergence 계산

                # 3) DSSIM 공간 맵 계산 (논문과 동일)
                ssim_map = self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
                dssim_map = (1.0 - ssim_map) / 2.0  # [B, H, W]

                # 4) KL divergence 계산
                KL = ((sigma_1.square() / torch.clamp(s_1.square(), min=eps)
                    + sigma_2.square() / torch.clamp(s_2.square(), min=eps)
                    - rho * r * sigma_1 * sigma_2 / torch.clamp(s_1 * s_2, min=eps)) / torch.clamp(1.0 - rho.square(), min=eps)
                    - torch.log(det / det_render) - 2) * 0.5

                # 5) likelihood (negative log-likelihood)
                # d_cos와 dssim_map은 detach하여 transient 모델만 학습
                likelihood = (1 / torch.clamp(1.0 - rho.square(), min=eps)) * (
                    cosd.squeeze(1).detach().square() / torch.clamp(sigma_1.square(), min=eps) +
                    dssim_map.detach().square() / torch.clamp(sigma_2.square(), min=eps) -
                    2 * cosd.squeeze(1).detach() * dssim_map.detach() * rho / torch.clamp(sigma_1 * sigma_2, min=eps)
                ) + torch.log(det)

                # 6) per-pixel 에너지 맵
                transient_energy = KL + likelihood  # [B, H, W]
                transient_energy = torch.nan_to_num(transient_energy, nan=0.0, posinf=10.0, neginf=0.0)

                # 7) (옵션) KL 기반 마스크 생성 및 photometric 손실에 적용 (논문 방식)
                # KL_threshold, dilate_exp, schedule_beta 등 하이퍼파라미터 필요
                # 예시:
                # transient_mask = transient_energy.clone().detach()
                # transient_mask = dilate_mask(transient_mask, opt_dilate_exp)
                # mask = torch.where(transient_mask < opt_KL_threshold, 1, 0)
                # alpha = np.exp(opt_schedule_beta * np.floor((1 + step) / 1.5))
                # mask = torch.bernoulli(torch.clip(alpha + (1 - alpha) * mask, min=0.0, max=1.0))
                # mask = mask.detach()
                # photometric_loss = torch.mean(diff * mask)

                # 8) 전체 평균으로 스칼라화
                transient_loss = transient_energy.mean()
                transient_loss_weighted = transient_loss


                #revised-1017               
                self.transient_optimizer.zero_grad()
                # 여기서 사용하는 transient_loss는 위에서 DETACH 입력으로 계산된 것
                transient_loss.backward()     # ConvDPT 파라미터에만 gradient
                self.transient_optimizer.step()
                if self.transient_scheduler is not None:
                    self.transient_scheduler.step()
            else:
                transient_loss_weighted = 0.0 


#--------------------------------------------------------------------------------------------------------------------------------------------

            #revised
            # cosine scheduling for ssim_lambda and compose the photometric loss
            if binary_mask is not None:
                curr_ssim_lambda = self.get_ssim_lambda(step)

                # loss definition
                rgbloss = (binary_mask * error_per_pixel).sum() / (binary_mask.sum()*3 + 1e-8)
                # ssim loss
                # ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
                # do not use because it allocates a lot of memories (16GB)
                ssimloss = 1.0 - self.masked_ssim(pixels.permute(0,3,1,2), colors.permute(0,3,1,2), binary_mask.permute(0,3,1,2)) # take all into B C H W
                
            #depth loss (experimental)
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

#revised-1013
            # ---------------- Self-Ensemble: pseudo-view render↔render loss ----------------
            se_coreg = torch.tensor(0.0, device=device)
            # 디버그: se_coreg 진입 조건 확인용 (train 루프 안, 조건문 바로 위)
    
            if (self.cfg.se_coreg_enable and binary_mask is not None and cfg.refine_start_iter < step and step % cfg.se_coreg_refine_every == 0
            ):
            
            # and cfg.se_coreg_start_iter < step
                #: implement pseudo view selection strategy
                # to use spotless-mlp classifier pixelwise model trained with binary mask and use that for pseudo view dynamic mask.


                # (1) Make pseudo views (may return multiple views). Limit to two
                # pv_c2w, pv_K = self._make_pseudo_views(camtoworlds, Ks, self.cfg.se_pseudo_K)

                # # Ensure torch tensors on device
                # if isinstance(pv_c2w, np.ndarray):
                #     pv_c2w_t = torch.from_numpy(pv_c2w).float().to(device)
                # else:
                #     pv_c2w_t = pv_c2w

                # if isinstance(pv_K, np.ndarray):
                #     pv_K_t = torch.from_numpy(pv_K).float().to(device)
                # else:
                #     pv_K_t = pv_K

                # if pv_c2w_t.shape[0] < 2:
                #     se_coreg_loss = 0.0
                # else:
                #     pv_c2w_use = pv_c2w_t[:2]
                #     pv_K_use = pv_K_t[:2]

                    # # convert mask dims safely (used in other parts if needed)
                    # if binary_mask.shape == (1, height, width, 1):
                    #     dyn_mask = binary_mask.squeeze(-1)
                    # else:
                    #     dyn_mask = binary_mask.permute(0, 3, 1, 2).squeeze(1)

                    # # decide which Gaussians to reset using first pseudo view info
                    # with torch.no_grad():
                    #     _, _, info_pv0 = self.rasterize_splats(
                    #         camtoworlds=pv_c2w_use[0:1],
                    #         Ks=pv_K_use[0:1],
                    #         width=width,
                    #         height=height,
                    #         sh_degree=sh_degree_to_use,
                    #         near_plane=cfg.near_plane,
                    #         far_plane=cfg.far_plane,
                    #     )
                    # reset_mask_pv = self._find_reset_gaussians_by_mask(dyn_mask, info_pv0)

                    # # (2) Render sigma (background) under no_grad to avoid storing activations
                    # with self._temporary_splats(self._clone_splats()):
                    #     # fully reset dynamic gaussians for background render
                    #     # self._apply_reset_attributes(reset_mask_pv, 0.0)
                    #     with torch.no_grad():
                    #         sigma_imgs, _, _ = self.rasterize_splats(
                    #             camtoworlds=pv_c2w_use,
                    #             Ks=pv_K_use,
                    #             width=width,
                    #             height=height,
                    #             sh_degree=sh_degree_to_use,
                    #             near_plane=cfg.near_plane,
                    #             far_plane=cfg.far_plane,
                    #         )
                    #         sigma_imgs = torch.clamp(sigma_imgs, 0.0, 1.0)

                    # # (3) Render delta (background + small perturb) with gradients enabled
                    # delta_splats = self._clone_splats()
                    # with self._temporary_splats(delta_splats):
                    #     self._apply_reset_attributes(reset_mask_pv, self._uap_noise_ratio(step))
                    #     delta_imgs, _, _ = self.rasterize_splats(
                    #         camtoworlds=pv_c2w_use,
                    #         Ks=pv_K_use,
                    #         width=width,
                    #         height=height,
                    #         sh_degree=sh_degree_to_use,
                    #         near_plane=cfg.near_plane,
                    #         far_plane=cfg.far_plane,
                    #     )
                    #     delta_imgs = torch.clamp(delta_imgs, 0.0, 1.0)

                    # (4) Compute photometric loss between background-only sigma and perturbed delta.
                    # Use detach on sigma_imgs (was computed with no_grad()) to be explicit.
                    # se_coreg = F.l1_loss(sigma_imgs.detach(), delta_imgs.detach())
                    # se_coreg_loss = se_coreg


                ### already exist camera view with delta perturbation model


                if binary_mask.shape == (1, height, width, 1):
                        dyn_mask = binary_mask.squeeze(-1)
                else:
                    dyn_mask = binary_mask.permute(0, 3, 1, 2).squeeze(1)
                
                # Render sigma (background) under no_grad to avoid storing activations
                # Request packed mode so the renderer fills per-pixel gaussian ids in `info`.
                with self._temporary_splats(self._clone_splats()):
                    with torch.no_grad():
                        _, _, info_sigma = self.rasterize_splats(
                            camtoworlds,
                            Ks,
                            width=width,
                            height=height,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            packed=True,
                        )
                    # renders image use!
                    sigma_imgs = torch.clamp(renders, 0.0, 1.0)
                
                # Use the packed-render info (which includes 'gaussian_ids') to find reset gaussians
                reset_mask_pv = self._find_reset_gaussians_by_mask(dyn_mask, info_sigma, cameras=camtoworlds)

                # Render delta (background + small perturb) with gradients enabled
                delta_splats = self._clone_splats()

                with self._temporary_splats(delta_splats):
                    self._apply_reset_attributes(reset_mask_pv, self._uap_noise_ratio(step))
                    with torch.no_grad():
                        # delta render doesn't strictly need gaussian_ids, but enabling packed
                        # is low-risk and keeps info consistent with the sigma render.
                        delta_imgs, _, _ = self.rasterize_splats(
                            camtoworlds,
                            Ks,
                            width=width,
                            height=height,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            packed=True,
                        )
                    delta_imgs = torch.clamp(delta_imgs, 0.0, 1.0)

                # (4) Compute photometric loss between background-only sigma and perturbed delta.
                # Use detach on sigma_imgs (was computed with no_grad()) to be explicit.
                se_coreg = F.l1_loss(sigma_imgs, delta_imgs.detach())
                se_coreg_loss = se_coreg


            else:
                se_coreg_loss = 0.0

            #need scheduler
            lambda_p = self.get_ssim_lambda(step)
            #total loss
            # loss = rgbloss * (1.0 - curr_ssim_lambda) + (ssimloss * curr_ssim_lambda) + (f * cfg.dino_lambda) + (transient_loss * cfg.transient_lambda) + (self.cfg.se_coreg_lambda * se_coreg_loss)
            # loss = rgbloss * (1.0 - lambda_p) + (lambda_p * se_coreg_loss) + (dino_loss *(lambda_p)) + (transient_loss * lambda_p)
            # loss = ( rgbloss  + dino_loss *cfg.dino_loss_lambda ) * (1.0 - lambda_p)  + (transient_loss * lambda_p)  

                
            # total loss            
            loss = ( rgbloss  + dino_loss *cfg.dino_loss_lambda*0.01 ) + se_coreg_loss *cfg.se_coreg_lambda + transient_loss_weighted * lambda_p
            if step % cfg.refine_every ==0 :
                print (  f"Step {step}: rgbloss={rgbloss.item():.6f}, dino_loss={dino_loss.item() if torch.is_tensor(dino_loss) else dino_loss:.6f}, se_coreg_loss={se_coreg_loss.item() if torch.is_tensor(se_coreg_loss) else se_coreg_loss:.6f} ")

            # loss = ( rgbloss  + dino_loss *cfg.dino_loss_lambda )  + (se_coreg_loss * cfg.se_coreg_lambda)* lambda_p + transient_loss_weighted * lambda_p
# ----------------------------------------------------------------------------------------------------------------------------

            loss.backward()


            desc = f"loss={loss.item():.9f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # tensorboard logging
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                #revised-1013
                if self.cfg.se_coreg_enable:
                    self.writer.add_scalar("train/se_coreg", se_coreg.item() if torch.is_tensor(se_coreg) else se_coreg, step)
                #------------------------------------------------------------------------------------------------------------------------
                self.writer.add_scalar("train/rgbloss", rgbloss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if transient_loss > 0:
                    self.writer.add_scalar("train/transient_loss", transient_loss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()


            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(info, binary_mask, step)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )
                    
#revised-1015---------------------------------------------------------------------------------------------------
                    # # test camera pose들에 대한 pseudo view 생성 및 저장
                    # # Trajectory의 초반과 후반부 카메라 포즈 생성
                    # if cfg.start_pseudo_view_iter > 0 and step >= cfg.start_pseudo_view_iter and step % cfg.refine_every == 0:
                    #     print("Generating trajectory start and end poses for pseudo views...")

                    #     # parser에서 직접 카메라 포즈 가져오기
                    #     K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
                    #     camtoworlds = get_ordered_poses(self.parser.camtoworlds)
                    #     camtoworlds = torch.from_numpy(camtoworlds).float().to(device)  # [N, 4, 4]


                    
                    #     # 전체 카메라 수의 20%를 초반과 후반에서 각각 선택
                    #     n_views = len(camtoworlds)
                    #     n_select = max(1, int(n_views * 0.2))  # 최소 1개는 선택

                    #     # 초반부와 후반부 카메라 포즈 선택
                    #     start_poses = camtoworlds[:n_select]  # 처음 20%
                    #     end_poses = camtoworlds[-n_select:]   # 마지막 20%

                    #     # 선택된 포즈들 결합
                    #     selected_poses = torch.cat([start_poses, end_poses], dim=0)  # [2*n_select, 4, 4]
                    #     selected_Ks = K.unsqueeze(0).repeat(len(selected_poses), 1, 1)  # [2*n_select, 3, 3]

            
                    #     # 각 선택된 포즈에 대해 K개의 pseudo view 생성
                    #     pseudo_camtoworlds, pseudo_Ks = self._make_pseudo_views(
                    #         selected_poses, 
                    #         selected_Ks, 
                    #         self.cfg.se_pseudo_K
                    #     )

                    #     # tag visiblity per gaussians
                    #     # Initialize Gaussian tracking
                    #     num_gaussians = len(self.splats["means3d"])
                    #     gaussian_render_counts = torch.zeros(num_gaussians, device=device)

                    #     # Render each pseudo view
                    #     for i in range(len(pseudo_camtoworlds)):
                    #         # Get camera parameters for this pseudo view
                    #         curr_c2w = pseudo_camtoworlds[i:i+1]  # [1, 4, 4]
                    #         curr_K = pseudo_Ks[i:i+1]  # [1, 3, 3]
                            
                    #         # Render the view
                    #         _, _, info = self.rasterize_splats(
                    #             camtoworlds=curr_c2w,
                    #             Ks=curr_K,
                    #             width=width,
                    #             height=height,
                    #             sh_degree=cfg.sh_degree,
                    #             near_plane=cfg.near_plane,
                    #             far_plane=cfg.far_plane,
                    #             packed=True  # Ensure we get gaussian_ids
                    #         )
                            
                    #         # Track which Gaussians were used in this view
                    #         if "gaussian_ids" in info:
                    #             gaussian_ids = info["gaussian_ids"].long()  # [nnz]
                    #             if gaussian_ids is not None and gaussian_ids.numel() > 0:
                    #                 gaussian_render_counts.index_add_(
                    #                     0, 
                    #                     gaussian_ids, 
                    #                     torch.ones_like(gaussian_ids, dtype=torch.float)
                    #                 )
                    
                    #     # Normalize counts to get percentage of views each Gaussian appears in
                    #     total_views = len(pseudo_camtoworlds)
                    #     gaussian_visibility_ratio = gaussian_render_counts / total_views  # [num_gaussians]
                        
                    #     #if gaussian_visibility_ratio is less than threshold, decay their opacity
                    #     low_visibility_mask = gaussian_visibility_ratio < cfg.gaussian_visibility_thresh
                    #     if low_visibility_mask.any():
                    #         # Use _decay_opacity function with stronger decay 
                    #         self._decay_opacity(low_visibility_mask, factor=0.005)
                    #         print(f"Adjusted opacity for {low_visibility_mask.sum().item()} low-visibility Gaussians")
                    

#---------------------------------------------------------------------------------------------------------------------
                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa                
                    if step > cfg.reset_every:
                        # The official code also implements screen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                        
#revised-1015-1024 ---------------------------------------------------------------------------------------------------
# Prune Gaussians too close to trajectory cameras (매번 다른 궤적 사용)
                    if cfg.start_pseudo_view_iter > 0 and step >= cfg.start_pseudo_view_iter and step % cfg.refine_every == 0:
                        # 1) Get ordered camera poses
                        prune_camtoworlds = get_ordered_poses(self.parser.camtoworlds)
                        
                        # 2) 매번 다른 샘플링 오프셋 사용 (시간에 따라 변화)
                        # step에 따라 샘플링 시작점을 순환 (0~19)
                        offset = (step // cfg.refine_every) % 20
                        
                        # 3) 무작위 섭동 추가 (작은 노이즈로 다양한 궤적 생성)
                        np.random.seed(step)  # step 기반 시드로 재현 가능하지만 매번 다름
                        # 샘플링된 키프레임 선택 (오프셋 적용)
                        keyframe_indices = np.arange(offset, len(prune_camtoworlds), 20)
                        keyframes = prune_camtoworlds[keyframe_indices].copy()
 
                        prune_camtoworlds = generate_interpolated_path(
                            keyframes, 40, spline_degree=1, smoothness=0.3
                        )  # [N, 3, 4]
                        
                        # 5) 무작위 스케일 변화 추가 (0.9~1.2 범위)
                        scale_variation = 0.9 + 0.3 * np.random.rand()  # 매번 다른 스케일
                        
                        # 6) Add homogeneous coordinates
                        prune_camtoworlds = np.concatenate(
                            [
                                prune_camtoworlds,
                                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(prune_camtoworlds), axis=0),
                            ],
                            axis=1,
                        )  # [N, 4, 4]
                        # Choose up to K trajectory cameras (reduces randomness vs single random camera)
                        K = min(8, len(prune_camtoworlds)) if len(prune_camtoworlds) > 0 else 0
                        if K == 0:
                            # nothing to do
                            continue

                        # pick K indices spaced across the path for diversity
                        if len(prune_camtoworlds) <= K:
                            sampled = prune_camtoworlds
                        else:
                            # evenly spaced selection + small random offset
                            idxs = np.linspace(0, len(prune_camtoworlds) - 1, K, dtype=int)
                            sampled = prune_camtoworlds[idxs]

                        # 7) Apply dynamic scale to sampled cameras
                        scale_factor = 1.1 * scale_variation
                        sampled = sampled * np.reshape(
                            np.array([[scale_factor, scale_factor, scale_factor, 1.0]]), (1, 4, 1)
                        )

                        # 8) Convert sampled cameras to torch tensor
                        prune_camtoworlds_t = torch.from_numpy(sampled).float().to(device)  # [K,4,4]

                        # Compute minimum distance from each Gaussian to any sampled camera
                        means3d = self.splats["means3d"]  # [num_gaussians, 3]
                        cam_positions = prune_camtoworlds_t[:, :3, 3]  # [K,3]
                        # torch.cdist returns [G, K]
                        dists = torch.cdist(means3d, cam_positions)  # [G, K]
                        min_dists = dists.min(dim=1)[0]

                        is_2close = min_dists < (0.1 * self.scene_scale)  # threshold configurable
                        is_prune = is_prune | is_2close
                        print(f"  -> [Offset:{offset}, Scale:{scale_variation:.2f}] Sampled {K} cams; marked {is_2close.sum().item()} Gaussians too close to trajectory cameras for pruning")
#-----------------------------------------------------------------------------------------------------
#revised-1016       # ========== 새로운 Floater Detection 알고리즘들 ==========  after dynamic pruen start iter
                    # 1. Depth-based floater detection 
                        # dyn_mask_2d = binary_mask.squeeze(-1) if binary_mask.dim() == 4 else binary_mask
                        # depth_floaters = self._detect_depth_inconsistent_gaussians(
                        #     curr_c2w, curr_K, width, height, dyn_mask_2d, depth_threshold=cfg.depth_floater_thresh #take pseudo view camera to this.
                        # )
                        # if depth_floaters.any():
                        #     is_prune = is_prune | (is_2close & depth_floaters)
                        #     print(f"  -> Marked {depth_floaters.sum().item()} depth-inconsistent floaters")
                    
                    
                    # 2. Inpaint dynamic regions (후기에만, pruning 후)
                    if (cfg.enable_dynamic_inpainting and step >= cfg.dynamic_prune_start_iter):
                        if step % cfg.refine_every == 0:
                            print("[Ghost Fix] Inpainting dynamic regions...")
                            n_inpainted = self._inpaint_dynamic_regions(
                                binary_mask, camtoworlds, inpaint_strength=1.0
                            )
                            if n_inpainted > 0:
                                print(f"  -> Inpainted {n_inpainted} background Gaussians into dynamic regions")
                    # ====================================================
#-----------------------------------------------------------------------------------------------------------
                    
                    if cfg.ubp:
                        not_utilized = self.running_stats["sqrgrad"] < cfg.ubp_thresh
                        is_prune = is_prune | not_utilized
                        #revised-1028
                    # Protect newly created Gaussians from immediate pruning by requiring
                    # a minimum age (in training steps) before they become eligible.
                    if "birth_steps" in self.running_stats and getattr(cfg, "min_age_before_prune", 0) > 0:
                        birth = self.running_stats["birth_steps"]
                        # birth is a tensor of dtype long on the same device as running_stats
                        age = (self.current_step - birth)
                        eligible = age >= int(cfg.min_age_before_prune)
                        # ensure boolean mask on same device/dtype
                        eligible = eligible.to(is_prune.device)
                        is_prune = is_prune & eligible
                    #------------------------------------------------------------

                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)

                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    if cfg.ubp:
                        self.running_stats["sqrgrad"].zero_()
                    self.running_stats["count"].zero_()
                    self.running_stats["w_static"].zero_()
                    self.running_stats["w_dynamic"].zero_()

                if step % cfg.reset_every == 0 and cfg.loss_type != "robust":
                    self.reset_opa(cfg.prune_opa * 2.0)
                if step == cfg.reset_sh and cfg.loss_type == "robust":
                    self.reset_sh()
            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.spotless_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()


            # Save the mask image with gt -renders
            if step > max_steps - 200 and cfg.semantics:
                st_interval = time.time()
                rgb_pred_mask = (binary_mask.repeat(1, 1, 1, 3).clone().detach())

                canvas = (
                    torch.cat([pixels, rgb_pred_mask, colors], dim=2)
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy()
                )
                imname = image_ids.cpu().detach().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/train_{imname}.png",
                    (canvas * 255).astype(np.uint8),
                )
                global_tic += time.time() - st_interval

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step+1}.pt",    #default pt & now revised
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                print(f"[TRANI] Evaluating while traing at step {step}...")
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    #revised-0919
    def update_running_stats(self, info: Dict, binary_mask: torch.Tensor, step: int):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        if cfg.ubp:
            sqrgrads = info["means2d"].sqrgrad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size

        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).float()
            )
            if cfg.ubp:
                self.running_stats["sqrgrad"].index_add_(
                    0, gs_ids, torch.sum(sqrgrads, dim=-1)
                )

        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).float()
            )     
            if cfg.ubp:
                self.running_stats["sqrgrad"].index_add_(
                    0, gs_ids, torch.sum(sqrgrads[sel], dim=-1)
                )


    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        print("[DEBUG] reset opacities IN")
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reset_sh(self, value: float = 0.001):
        """Utility function to reset SH specular coefficients."""
        colors = torch.clamp(
            self.splats["shN"], max=torch.abs(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "shN":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(colors)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None or k.find("err") != -1:
                continue

            # repeats = [2] + [1] * (v.dim() - 1)
            # v_new = v[sel].repeat(repeats)
            # if k == "sqrgrad":
            #     v_new = torch.ones_like(
            #         v_new
            #     )  # the new ones are assumed to have high utilization in the start

            #revised-1028
            # For most stats we repeat the selected entries twice (split -> 2 children)
            if k == "birth_steps":
                # new children get birth step == current_step
                n_children = 2 * len(sel)
                v_new = torch.full((n_children,), fill_value=getattr(self, "current_step", 0), device=self.device, dtype=v.dtype)
            else:
                repeats = [2] + [1] * (v.dim() - 1)
                v_new = v[sel].repeat(repeats)
                if k == "sqrgrad":
                    # the new ones are assumed to have high utilization in the start
                    v_new = torch.ones_like(v_new)
            #---------------------------------------------------------------
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            if k == "sqrgrad":
                # self.running_stats[k] = torch.cat(
                #     (v, torch.ones_like(v[sel]))
                # )  # new ones are assumed to have high utilization

                #revised-1028
                # new ones are assumed to have high utilization
                self.running_stats[k] = torch.cat((v, torch.ones_like(v[sel])))
            elif k == "birth_steps":
                # stamp birth step for newly duplicated gaussians
                n_new = len(sel)
                new_births = torch.full((n_new,), fill_value=getattr(self, "current_step", 0), device=self.device, dtype=v.dtype)
                self.running_stats[k] = torch.cat((v, new_births))
                #---------------------------------------------------------------
            else:
                self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        # test용 GT/Render 저장 폴더 revised
        test_dir = os.path.join(cfg.result_dir, "test")
        method_test_dir = os.path.join(test_dir, "ours_30000")
        gt_test_dir = os.path.join(method_test_dir, "gt")
        rend_test_dir = os.path.join(method_test_dir, "renders")
        os.makedirs(gt_test_dir, exist_ok=True)
        os.makedirs(rend_test_dir, exist_ok=True)

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "MASK_psnr": [], "MASK_ssim": [], "MASK_lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            filename = data["filename"]
            if isinstance(filename, list):
                filename = filename[0]
            base = Path(filename).stem
            name = base.replace("_extra", "").replace("_clutter", "") + ".png"
            test_mask_name = base.replace("_extra", "").replace("_clutter", "")
        

            #revised 0910
            # test mask saved
            test_binary_mask = None

            if self.mask_dir:
                mpath = self.mask_dict.get(test_mask_name, None)
                test_mdst_dir = Path(self.cfg.result_dir) / "test" / "mask"
                test_mdst_dir.mkdir(parents=True, exist_ok=True)
                if mpath:
                    mdst = test_mdst_dir / mpath.name
                    shutil.copy(mpath, mdst)
                else:
                    raise FileNotFoundError(f"Mask not found for '{test_mask_name}' in mask_dir.")

                # mask load and process
                test_mask_img = imageio.imread(mpath)
                if test_mask_img.ndim == 3:
                    print("[WARN] mask image has 3 channels, converting to grayscale")
                    # test_mask_img = test_mask_img[..., 0]

                orig_imsize = self.parser.imsize_dict.get(name)
                if orig_imsize is not None:
                    h0, w0 = orig_imsize
                    factor = self.cfg.data_factor
                    h1, w1 = h0 // factor, w0 // factor
                else:
                    h1, w1 = height, width
                test_mask_img_resized = cv2.resize(test_mask_img, (w1, h1), interpolation=cv2.INTER_NEAREST)

                # dilation 적용
                kernel = np.ones((5, 5), np.uint8)  # 커널 크기를 조절해서 enlarge 정도를 변경
                test_mask_img_resized = cv2.dilate(test_mask_img_resized, kernel, iterations=1)

                test_mask_tensor = torch.from_numpy(test_mask_img_resized).float() / 255.0
                test_mask_tensor = test_mask_tensor.unsqueeze(0).unsqueeze(-1)
                
                if test_mask_tensor.shape[1] != h1 or test_mask_tensor.shape[2] != w1:
                    test_mask_tensor = F.interpolate(test_mask_tensor.permute(0, 3, 1, 2), size=(colors.shape[1], colors.shape[2]), mode='nearest')
                    test_mask_tensor = test_mask_tensor.permute(0, 2, 3, 1)
                    
                test_binary_mask = test_mask_tensor.to(device)

                test_binary_mask = test_binary_mask.permute(0, 3, 1, 2)   # [B,1,H,W]

                test_binary_mask = 1.0 - test_binary_mask  # 1 for background, 0 for foreground
                # print("test_binary_mask unique:", torch.unique(test_binary_mask))
                # print("test_binary_mask mean:", test_binary_mask.mean())
                # print("test_binary_mask shape:", test_binary_mask.shape)


            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # revised
            gt_np = (pixels.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            rend_np = (colors.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            
            imageio.imwrite(os.path.join(gt_test_dir,  name), gt_np)
            imageio.imwrite(os.path.join(rend_test_dir, name), rend_np)
            # print("[DEBUG] TEST gt and rendered images are writtn")

            
            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            

            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))
            
            # test metrics with mask applied
            metrics["MASK_psnr"].append(self.masked_psnr(colors, pixels, test_binary_mask))
            metrics["MASK_ssim"].append(self.masked_ssim(colors, pixels, test_binary_mask).mean())     
            comp_pred = test_binary_mask * colors + (1 - test_binary_mask) * pixels
            comp_gt   = pixels
            metrics["MASK_lpips"].append(self.lpips(comp_pred, comp_gt))



        masked_psnr = torch.stack(metrics["MASK_psnr"]).mean()
        masked_ssim = torch.stack(metrics["MASK_ssim"]).mean()
        masked_lpips = torch.stack(metrics["MASK_lpips"]).mean()

        ellipse_time /= len(valloader)

        print("")
        print("[EVAL] TEST rendered image WITH MASK metrics")
        print(
            f"MASK_PSNR: {masked_psnr.item():.9f}, MASK_SSIM: {masked_ssim.item():.9f}, MASK_LPIPS: {masked_lpips.item():.9f} "
            f"Time: {ellipse_time:.9f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
            )    

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print("")
        print("[EVAL] saving TEST rendered image and test metrics")
        print(
            f"PSNR: {psnr.item():.9f}, SSIM: {ssim.item():.9f}, LPIPS: {lpips.item():.9f} "
            f"Time: {ellipse_time:.9f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

        print("")
        print("[EVAL] saving TRAINING rendered images in eval mode...")
        cfg = self.cfg
        device = self.device
        
        
        # train metrics printing code
        # train용 GT/Render 저장 폴더 revised
        train_dir = os.path.join(cfg.result_dir, "train")
        method_train_dir = os.path.join(train_dir, "ours_30000")
        gt_train_dir = os.path.join(method_train_dir, "gt")
        rend_train_dir = os.path.join(method_train_dir, "renders")
        os.makedirs(gt_train_dir, exist_ok=True)
        os.makedirs(rend_train_dir, exist_ok=True)

        trainloader = torch.utils.data.DataLoader(  self.trainset, batch_size=1, shuffle=False, num_workers=1 )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        for i, data in enumerate(trainloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()


            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # revised
            gt_np = (pixels.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            rend_np = (colors.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            
            # ─── 원본 파일명으로 저장하기 ─────────────────────────
            filename = data["filename"]
            if isinstance(filename, list):
                filename = filename[0]
            base = Path(filename).stem
            name = base.replace("_extra", "").replace("_clutter", "") + ".png"
            
            imageio.imwrite(os.path.join(gt_train_dir,  name), gt_np)
            imageio.imwrite(os.path.join(rend_train_dir, name), rend_np)
            
            
            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print("")
        print("[EVAL] train rendered image save and train metrics")
        print(
            f"PSNR: {psnr.item():.9f}, SSIM: {ssim.item():.9f}, LPIPS: {lpips.item():.9f} "
            f"Time: {ellipse_time:.9f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
    

    @torch.no_grad()
    def render_traj(self, step: int ):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        camtoworlds = get_ordered_poses(self.parser.camtoworlds)

        camtoworlds = generate_interpolated_path(
            camtoworlds[::20].copy(), 40, spline_degree=1, smoothness=0.3
        )  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]
        camtoworlds = camtoworlds * np.reshape([1.1, 1.1, 1, 1], (1, 4, 1))

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED", #hard coded for depth rendering
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # # write images
            # canvas = torch.cat(
            #     [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            # )

            # canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)

            #-----------------revised-1002-------------------------------
            colors_np = (colors.cpu().numpy() * 255).astype(np.uint8)        # [H, W, 3]
            depth_f32 = depths.squeeze(-1).cpu().numpy().astype(np.float32)  # [H, W], 

            import matplotlib.cm as cm
            cmap = cm.get_cmap('plasma')   # 'jet','magma','plasma','viridis' 
            depth_color = (cmap(depth_f32)[:, :, :3] * 255).astype(np.uint8)  # RGBA -> RGB

            canvas = np.concatenate(
                [colors_np, depth_color],
                axis=0 if width > height else 1
            )
            #-------------------------------------------------------------
            
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.gif", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.gif")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(10)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
