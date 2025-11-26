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

#for feature map save
# import imageio.v2 as imageio
# from sklearn.decomposition import PCA


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
    refine_start_iter: int = 500 #100
    refine_stop_iter: int = 15000 # tag
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
    ubp: bool = False
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
    seed : int = 42
    pseudo_gt_dir: Optional[str] = None  # temporary directory for pseudo GT

    # Weight for MLP mask loss
    mlp_gt_lambda: float = 0.1
    # Weight for DINO feature loss
    dino_loss_flag: bool = True
    dino_loss_lambda: float = 0.2 #tag
    mask_adaptation: bool = False
    

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
    se_coreg_lambda: float = 0.1
    se_coreg_refine_every: int = 100    # self-ensemble 

    #revised-1028
    # Minimum age (in training steps) before a newly created Gaussian can be
    # pruned. Helps avoid immediate prune after duplication/splitting.
    min_age_before_prune: int = 150

    #revised11-18
    # Learning rate warmup steps
    lr_warmup_steps: int = 1000

  

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


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(cfg.seed)
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
            self.spotless_loss = lambda p, minimum, maximum: torch.mean(
                torch.nn.ReLU()(p - minimum) + torch.nn.ReLU()(maximum - p))

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

        #revised 11-18
        max_steps = cfg.max_steps
        init_step = 0

        # --- LR 웜업 스케줄러 로직 시작 ---
        schedulers = []
        
        # 웜업 후 실제 감속(decay)이 일어날 스텝 수 계산
        main_decay_steps = max(1, max_steps - cfg.lr_warmup_steps)

        # 1. 메인 옵티마이저 (means3d 등) 스케줄
        optimizer_main = self.optimizers[0]
        
        # 웜업 스케줄러 (예: 0.01배 LR -> 1.0배 LR)
        warmup_scheduler_main = torch.optim.lr_scheduler.LinearLR(
            optimizer_main,
            start_factor=0.01, # 웜업 시작 시 LR 배율
            end_factor=1.0,
            total_iters=cfg.lr_warmup_steps
        )
        
        # 메인 감속 스케줄러 (기존 ExponentialLR)
        decay_scheduler_main = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_main, 
            gamma=0.01 ** (1.0 / main_decay_steps) # 웜업 이후 스텝에 대해서만 감속
        )
        
        # 두 스케줄러를 순차적으로 연결
        sequential_scheduler_main = torch.optim.lr_scheduler.SequentialLR(
            optimizer_main,
            schedulers=[warmup_scheduler_main, decay_scheduler_main],
            milestones=[cfg.lr_warmup_steps] # 웜업 스텝 이후에 decay_scheduler_main으로 전환
        )
        schedulers.append(sequential_scheduler_main)

        # 2. 포즈 옵티마이저 (pose_opt) 스케줄 (활성화된 경우)
        if cfg.pose_opt:
            optimizer_pose = self.pose_optimizers[0]
            
            warmup_scheduler_pose = torch.optim.lr_scheduler.LinearLR(
                optimizer_pose,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.lr_warmup_steps
            )
            
            decay_scheduler_pose = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_pose, 
                gamma=0.01 ** (1.0 / main_decay_steps) # 동일한 스텝 수 적용
            )
            
            sequential_scheduler_pose = torch.optim.lr_scheduler.SequentialLR(
                optimizer_pose,
                schedulers=[warmup_scheduler_pose, decay_scheduler_pose],
                milestones=[cfg.lr_warmup_steps]
            )
            schedulers.append(sequential_scheduler_pose)
        # --- LR 웜업 스케줄러 로직 끝 ---

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

            renders, alphas, info = self.rasterize_splats(
            # train forward
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
                pred_mask = self.robust_mask(
                    error_per_pixel, self.running_stats["avg_err"]
                )
                if cfg.semantics:
                    sf = data["semantics"].to(device)
                    if cfg.cluster:
                        # cluster the semantic feature and mask based on cluster voting
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="nearest",
                        )(sf).squeeze(0)
                        pred_mask = self.robust_cluster_mask(pred_mask, semantics=sf)
                    else:
                        # use spotless mlp to predict the mask
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="bilinear",
                        )(sf).squeeze(0)
                        pos_enc = get_positional_encodings(
                            colors.shape[1], colors.shape[2], 20
                        ).permute((2, 0, 1))
                        sf = torch.cat([sf, pos_enc], dim=0)
                        sf_flat = sf.reshape(sf.shape[0], -1).permute((1, 0))
                        self.spotless_module.eval()
                        pred_mask_up = self.spotless_module(sf_flat)
                        pred_mask = pred_mask_up.reshape(
                            1, colors.shape[1], colors.shape[2], 1
                        )
                        # calculate lower and upper bound masks for spotless mlp loss
                        lower_mask = self.robust_mask(
                            error_per_pixel, self.running_stats["lower_err"]
                        )
                        upper_mask = self.robust_mask(
                            error_per_pixel, self.running_stats["upper_err"]
                        )
                log_pred_mask = pred_mask.clone()
                if cfg.schedule:
                    # schedule sampling of the mask based on alpha
                    alpha = np.exp(cfg.schedule_beta * np.floor((1 + step) / 1.5))
                    pred_mask = torch.bernoulli(
                        torch.clip(
                            alpha + (1 - alpha) * pred_mask.clone().detach(),
                            min=0.0,
                            max=1.0,
                        )
                    )
                rgbloss = (pred_mask.clone().detach() * error_per_pixel).mean()
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            # binary_mask: 연속 확률(가중치) 유지. 임계(threshold) 제거.
            # 이후 동적/정적 판별이 필요한 곳(예: inpainting)은 연속값에 대해 <0.5 비교를 그대로 사용.
            binary_mask = pred_mask.clone().detach().permute(0, 3, 1, 2)  # [B,1,H,W]
            
            # revised
            #--------------------------------------------------------------------------------------------------------------------------------------------
            if cfg.dino_loss_flag:
                if not hasattr(self, "dino_extractor"):
                    self.dino_extractor = DinoFeatureExtractor(getattr(cfg, "dino_version", "dinov2_vits14"))
                if not hasattr(self, "dino_head"):
                    # DINO 토큰(저해상도) -> 픽셀 해상도로 올리는 경량 업샘플 헤드
                    self.dino_head = DinoUpsampleHead(self.dino_extractor.dino_model.embed_dim).to(self.device)

                # self hard coded control
                mask_adaptation = cfg.mask_adaptation

                # 1) DINO 토큰 추출 (GT/Render)
                gt_chw     = pixels.permute(0, 3, 1, 2)    # [B,3,H,W], GT
                render_chw = colors.permute(0, 3, 1, 2)    # [B,3,H,W], 렌더 결과

                # The DINO model and upsampling head should be in eval mode and have gradients disabled.
                # This is handled in the DinoFeatureExtractor constructor.
                self.dino_extractor.dino_model.eval()
                self.dino_head.eval()

                # Prepare inputs for feature extraction
                input_gt = gt_chw
                input_render = render_chw

                # Extract GT features with no_grad for efficiency
                with torch.no_grad():
                    ftok, Th, Tw, _, _ = self.dino_extractor.extract_tokens(input_gt)

                # Extract render features with gradients enabled to flow back to splats
                fhtok, Th2, Tw2, _, _ = self.dino_extractor.extract_tokens(input_render)

                assert (Th == Th2) and (Tw == Tw2), "DINO token grid mismatch between GT and Render"
                B, _, C = ftok.shape

                f_gt = ftok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()
                f_rnd = fhtok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()

                # 2) 업샘플 → 이미지 해상도 정렬
                ps = self.dino_extractor.patch_size
                Hpad, Wpad = Th * ps, Tw * ps

                with torch.no_grad():
                    f_gt_up_pad = F.interpolate(self.dino_head(f_gt), size=(Hpad, Wpad), mode="bilinear", align_corners=False)
                f_rnd_up_pad = F.interpolate(self.dino_head(f_rnd), size=(Hpad, Wpad), mode="bilinear", align_corners=False)

                # 오른쪽/아래만 패딩했으므로 좌상단 기준으로 원 해상도(H,W)로 자르기
                H, W = colors.shape[1], colors.shape[2]
                f_gt_up = f_gt_up_pad[:, :, :H, :W]
                f_rnd_up = f_rnd_up_pad[:, :, :H, :W]


                # 3) 코사인 불일치(=의미 차이) 맵 -> 배경(inlier) 영역에서만 정합 유도
                cosd = 1.0 - F.cosine_similarity(f_gt_up, f_rnd_up, dim=1, eps=1e-6).unsqueeze(1)  # [B,1,H,W]

                # dino_feat_loss = (cosd).sum() / (stable_denominator + 1e-8)
                dino_feat_loss = (cosd.sum()) / (binary_mask.sum() + 1e-8)

                dino_loss =  dino_feat_loss
                
            else:
                dino_loss = 0.0

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
            else:
                depthloss = 0.0

#revised-1013
            # ---------------- Self-Ensemble: pseudo-view render↔render loss ----------------
            se_coreg = torch.tensor(0.0, device=device)
      
            if cfg.se_coreg_enable and step > cfg.refine_start_iter and step % cfg.se_coreg_refine_every ==0 :
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
                reset_mask_pv = self._find_reset_gaussians_by_mask(binary_mask, info_sigma, cameras=camtoworlds)

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

            # total loss            
            loss = ( rgbloss  + dino_loss *cfg.dino_loss_lambda*0.01 ) + se_coreg_loss *cfg.se_coreg_lambda
            if step % cfg.refine_every ==0 :
                print (  f"Step {step}: rgbloss={rgbloss.item():.6f}, dino_loss={dino_loss.item() if torch.is_tensor(dino_loss) else dino_loss:.6f}, se_coreg_loss={se_coreg_loss.item() if torch.is_tensor(se_coreg_loss) else se_coreg_loss:.6f} ")

            # loss = ( rgbloss  + dino_loss *cfg.dino_loss_lambda )  + (se_coreg_loss * cfg.se_coreg_lambda)* lambda_p + transient_loss_weighted * lambda_p
# ----------------------------------------------------------------------------------------------------------------------------

            loss.backward()

            if self.mlp_spotless:
                self.spotless_module.train()
                spot_loss = self.spotless_loss(
                    pred_mask_up.flatten(), upper_mask.flatten(), lower_mask.flatten()
                )
                reg = 0.5 * self.spotless_module.get_regularizer()
                spot_loss = spot_loss + reg
                spot_loss.backward()

            # Pass the error histogram for capturing error statistics
            info["err"] = torch.histogram(
                torch.mean(torch.abs(colors - pixels), dim=-1).clone().detach().cpu(),
                bins=cfg.bin_size,
                range=(0.0, 1.0),
            )[0]


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
                # binary_mask는 [B, C, H, W] 형태이므로 [B, H, W, C]로 변환
                # rgb_pred_mask = binary_mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).clone().detach()
                rgb_pred_mask = (
                    (log_pred_mask > 0.5).repeat(1, 1, 1, 3).clone().detach()
                )
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
        metrics = {"psnr": [], "ssim": [], "lpips": []}
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

        ellipse_time /= len(valloader)

   

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