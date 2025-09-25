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
from dino_utils import DinoFeatureExtractor, DPT_Head, DinoUpsampleHead


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
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
                                  
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
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15000
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
    # seed : int = 42
    pseudo_gt_dir: Optional[str] = None  # temporary directory for pseudo GT

    # Weight for MLP mask loss
    # revised-0913
    mlp_gt_lambda: float = 0.1
    # mlp_teacher_mix_start: int = 5000
    # mlp_teacher_mix_end:   int = 15_000
    # mlp_teacher_gt_start:  float = 1.0   # w(0)
    # mlp_teacher_gt_end:    float = 0.5   # w(T)
    # mlp_dice_lambda:       float = 0.2

    train_cutgrad: int = 20000

    reseed_dyn_thr   : float = 0.70
    reseed_sta_thr   : float =  0.80
    reseed_min_count : int =  20
    reseed_knn       : int = 16
    reseed_alpha     : float =  0.02

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
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        mask_bchw: Optional[torch.Tensor] = None,  # [1,1,H,W], 1=static, 0=dynamic # revised-0924
        step: Optional[int] = None, # revised-0924
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        #revised-0924
        cfg = self.cfg
        if step is not None and cfg.refine_start_iter <= step <= cfg.refine_stop_iter:  
            with torch.no_grad():
                gate = None  # [N] in {0,1}

                # # per-view gate from image mask (가우시안의 중심이 동적영역에 있는가!)
                # if mask_bchw is not None:
                #     # 현재 배치가 보통 C=1이므로 0번째 뷰 사용
                #     viewmat = torch.linalg.inv(camtoworlds)[0]  # [4,4]
                #     K = Ks[0]                                   # [3,3]
                #     xyz1 = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)  # [N,4]
                #     xc   = (viewmat @ xyz1.t()).t()[:, :3]      # [N,3] (cam space)
                #     z    = xc[:, 2].clamp_min(1e-6)             # 깊이
                #     hp   = (K @ xc.t()).t()                     # [N,3]
                #     u, v = hp[:, 0] / z, hp[:, 1] / z           # pixel coords
                #     gx   = (u / (width  - 1) * 2) - 1           # [-1,1]
                #     gy   = (v / (height - 1) * 2) - 1           # [-1,1]
                #     grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)  # [1,1,N,2]
                #     m = F.grid_sample(mask_bchw, grid, mode="nearest", align_corners=True).view(-1)  # [N]
                #     m = m * (z > 0).float()                     # 카메라 뒤는 0
                #     gate_view = (m > 0.5).float()               # 1=static, 0=dynamic// current view dynamic/static decision
                #     gate = gate_view

                # (b)dynamic/static decision gate
                dyn = self.running_stats.get("w_dynamic", None)
                sta = self.running_stats.get("w_static",  None)

                # both available
                if dyn is not None and sta is not None:
                    ratio_dyn = dyn.clamp_min(0.0) / (dyn.clamp_min(0.0) + sta.clamp_min(0.0) + 1e-6)
                    thr = float(getattr(cfg, "dyn_gate_thresh", 0.30))
                    gate_stat = (ratio_dyn < thr).float() # 가우시안이 splat된 것이 동적 영역에 많이 포함되는가
                    gate = gate_stat if gate is None else (gate * gate_stat)

                # (c) 최종 적용: α ← α * gate   (이 리포는 α를 라스터라이저에 넘김)
                if gate is not None:
                    opacities = opacities * gate
        

        #revised-0923
        # --- Forward gating: hide dynamic-suspect GS in rendering ---
        # with torch.no_grad():
        #     dyn = self.running_stats.get("w_dynamic", None)
        #     sta = self.running_stats.get("w_static",  None)
        #     if dyn is not None and sta is not None:
        #         ratio_dyn = dyn.clamp_min(0.0) / (dyn.clamp_min(0.0) + sta.clamp_min(0.0) + 1e-6)  # [N]
        #         gate_gauss = (ratio_dyn < 0.3).float()                        # 1: keep, 0: hide
        #         opacities = opacities * gate_gauss                            # [N] × [N] -> hide in forward

        # #----------------------------------------------------------------

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
            packed=self.cfg.packed,
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
                ramp_end = int(0.5 * cfg.max_steps) 
                t = min(step / max(ramp_end, 1), 1.0)
                wt = 0.5 - 0.5 * math.cos(math.pi * t) 
                return wt * cfg.ssim_lambda
    #revised-0918
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
        trainloader_iter = iter(trainloader) #이미지를 하나씩 넘김.

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

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
                    binary_mask = 1.0 -  mask_tensor.to(device) # 1 for background, 0 for dynamic object

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                mask_bchw=binary_mask.permute(0,3,1,2) if binary_mask is not None else None,#revised-0924
                step=step, #revised-0924
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
                        print("this should not be printed.")
                    else:
                        # 4.1.2 part: use spotless mlp to predict the mask
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
                        pred_mask = pred_mask_up.reshape(  # mlp predicted mask
                            1, colors.shape[1], colors.shape[2], 1
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


            #revised-0913-3
            # ------------------------------------------------------------------------------------
            # # 0) lazy init (한 번만 만들기)
            # if not hasattr(self, "dino_extractor"):
            #     self.dino_extractor = DinoFeatureExtractor(getattr(cfg, "dino_version", "dinov2_vits14"))
            # if not hasattr(self, "dino_head"):
            #     # DINO 토큰(저해상도) -> 픽셀 해상도로 올리는 경량 업샘플 헤드
            #     self.dino_head = DinoUpsampleHead(self.dino_extractor.dino_model.embed_dim).to(self.device)

            # # 1) DINO 토큰 추출 (GT/Render)  **no no_grad()**  ← 그래디언트가 colors까지 흘러가야 함
            # gt_chw     = pixels.permute(0, 3, 1, 2)    # [B,3,H,W], GT

            # render_chw = colors.permute(0, 3, 1, 2)    # [B,3,H,W], 렌더 결과
            # mask_chw = binary_mask.permute(0, 3, 1, 2)   # [B,1,H,W], 마스크
            # render_chw_safe = render_chw * mask_chw + render_chw.detach() * (1.0 - mask_chw) # 마스크 영역은 렌더링 그래디언트가 안 흐르도록 처리

            # ftok  = self.dino_extractor.extract_tokens(gt_chw)      # [B,Th*Tw,C] (백본 파라미터는 require_grad=False로 freeze)
            # fhtok = self.dino_extractor.extract_tokens(render_chw)  # [B,Th*Tw,C]

            # with torch.no_grad():  
            #     ftok,  Th, Tw, _, _  = self.dino_extractor.extract_tokens(gt_chw)      
            # fhtok, Th2, Tw2, _, _  = self.dino_extractor.extract_tokens(render_chw_safe)  
            # assert (Th == Th2) and (Tw == Tw2), "DINO token grid mismatch between GT and Render"
            # B, _, C = ftok.shape

            # f_gt  = ftok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()   # [B,C,Th,Tw]
            # f_rnd = fhtok.view(B, Th, Tw, C).permute(0, 3, 1, 2).contiguous()  # [B,C,Th,Tw]


            # # 2) 업샘플 → 이미지 해상도 정렬
            # ps = self.dino_extractor.patch_size
            # Hpad, Wpad = Th * ps, Tw * ps

            # f_gt_up_pad  = F.interpolate(self.dino_head(f_gt),  size=(Hpad, Wpad), mode="bilinear", align_corners=False)
            # f_rnd_up_pad = F.interpolate(self.dino_head(f_rnd), size=(Hpad, Wpad), mode="bilinear", align_corners=False)

            # # 오른쪽/아래만 패딩했으므로 좌상단 기준으로 원 해상도(H,W)로 자르기
            # H, W = colors.shape[1], colors.shape[2]
            #                                       # [B,1,H,W]
            # f_gt_up  = f_gt_up_pad[:,  :, :H, :W]   # [B,C,H,W]
            # f_rnd_up = f_rnd_up_pad[:, :, :H, :W]   # [B,C,H,W]
            
            # # 3) 코사인 불일치(=의미 차이) 맵 -> 배경(inlier) 영역에서만 정합 유도
            # cosd = 1.0 - F.cosine_similarity(f_gt_up, f_rnd_up, dim=1, eps=1e-6).unsqueeze(1)  # [B,1,H,W]
            
            # dino_feat_loss = (cosd * mask_chw).sum() / (mask_chw.sum() + 1e-8)

            # # 4) 램프/가중치 (초기 과구속 방지)
            # lam_dino_max   = float(getattr(cfg, "dino_feat_lambda", 0.1)) 
            # dino_start     = int(getattr(cfg, "dino_feat_start", 0))
            # dino_end       = int(getattr(cfg, "dino_feat_end",   max(1, cfg.max_steps // 3)))
            # if step <= dino_start:
            #     lam_dino = 0.0
            # elif step >= dino_end:
            #     lam_dino = lam_dino_max
            # else:
            #     tau = (step - dino_start) / max(dino_end - dino_start, 1)
            #     lam_dino = lam_dino_max * 0.5 * (1.0 - math.cos(math.pi * tau))  # 코사인 램프 0→lam_dino_max

            # dino_loss = lam_dino * dino_feat_loss

            #revised-0918
            #coscine scheduling for ssim_lambda
            curr_ssim_lambda = self.get_ssim_lambda(step)

            # loss definition
            rgbloss = (binary_mask * error_per_pixel).sum() / (binary_mask.sum()*3 + 1e-8)
            # ssim loss
            # ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
            ssimloss = 1.0 - self.masked_ssim(pixels.permute(0,3,1,2), colors.permute(0,3,1,2), binary_mask.permute(0,3,1,2)) # take all into B C H W
            #total loss
            loss = rgbloss * (1.0 - curr_ssim_lambda) + ssimloss * curr_ssim_lambda


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

            loss.backward()
            

            #revised-0921
            #gaussians gating
            
            # dynamic/foreground gaussians grad pruning (coarse to fine)
            # re_start , re_end = 500, cfg.train_cutgrad
            # if step < re_start:
            #     gate_thres = 1.0
            # elif step > cfg.train_cutgrad:  
            #     gate_thres = 0.1
            # else:
            #     tau = (step - re_start) / max(re_end - re_start, 1)
            #     gate_thres = 1.0 - 0.9 * 0.5 * (1.0 - math.cos(math.pi * tau)) # 1.0 -> 0.1 cosine ramp down
            
            # if step > cfg.train_cutgrad:
            #     cnt = self.running_stats["count"].clamp_min(1e-6)
            #     d = (self.running_stats["w_dynamic"] / cnt)                      # [N]
            #     print("[DEBUG]: count is f{cnt}, d is {d}")
            #     gate = (d < gate_thres).float()        
            #     # gate_col = gate[:, None, None]                               # SH 
            #     # opacities
            #     if self.splats["opacities"].grad is not None:
            #         self.splats["opacities"].grad *= gate
            #     # # SH 색(기본 경로)
            #     # if "sh0" in self.splats and self.splats["sh0"].grad is not None:
            #     #     self.splats["sh0"].grad *= gate_col
            #     # if "shN" in self.splats and self.splats["shN"].grad is not None:
            #     #     self.splats["shN"].grad *= gate_col

            #     if self.splats["means3d"].grad is not None:
            #         self.splats["means3d"].grad *= gate[:, None]
            #     if self.splats["scales"].grad is not None:
            #         self.splats["scales"].grad *= gate[:, None]
            #     if self.splats["quats"].grad is not None:
            #         self.splats["quats"].grad *= gate[:, None]
            #-------------------------------------------------------


            # sls-mlp trainer
            if self.mlp_spotless:
                self.spotless_module.train()

                #revised-0913

                # ablation study1 w cosine scheduling
                # T0, T1 = cfg.mlp_teacher_mix_start, cfg.mlp_teacher_mix_end
                # if step <= T0:    w = cfg.mlp_teacher_gt_start
                # elif step >= T1:  w = cfg.mlp_teacher_gt_end
                # else:
                #     tau = (step - T0) / max(T1 - T0, 1)
                #     w = cfg.mlp_teacher_gt_end + 0.5*(cfg.mlp_teacher_gt_start - cfg.mlp_teacher_gt_end)*(1 + math.cos(math.pi*tau))
                
                # # w = 0.5  #ablation study2 w is 0.5 fixed
                # teacher = torch.clamp(w * binary_mask + (1.0 - w) * pred_mask.detach(), 0.0, 1.0)


                if binary_mask is not None:
                    pred_prob = pred_mask_up.reshape(1, colors.shape[1], colors.shape[2], 1) # pred_mask from mlp--like noise!
                    gt_inlier = binary_mask 

                    # bce = F.binary_cross_entropy(pred_prob, teacher)
                    # bce_dynamic = F.binary_cross_entropy(pred_prob*(1-binary_mask),1-binary_mask)

                    # if cfg.mlp_dice_lambda > 0:
                    #     eps = 1e-6
                    #     inter = (pred_prob * teacher).sum()
                    #     dice = 1.0 - (2*inter + eps) / (pred_prob.sum() + teacher.sum() + eps) # Dice loss means 1 - Dice coeff
                    #     bce = bce + cfg.mlp_dice_lambda * dice

                        # ablation study3 doublemask
                        # dy_inter = (pred_prob * (1-binary_mask)).sum()
                        # dy_dice = 1.0 - (2*dy_inter + eps) / (pred_prob*(1-binary_mask)).sum() + (1-binary_mask).sum() + eps
                        # bce_dynamic = bce_dynamic + cfg.mlp_dice_lambda * dy_dice

                        # bce = bce + bce_dynamic

                    # spot_loss = cfg.mlp_gt_lambda * bce + 0.3 * self.spotless_module.get_regularizer() #mlp_gt_lambda = 1.0 hard coded
                    bce = F.binary_cross_entropy(pred_prob, gt_inlier)
                    spot_loss = cfg.mlp_gt_lambda * bce  # hard coded to 1.0

                reg = 0.3 * self.spotless_module.get_regularizer()
                spot_loss = spot_loss + reg
                spot_loss.backward()

            # Pass the error histogram for capturing error statistics
            #revised-0921 but not used
            # info["err"] = torch.histogram(
            #     torch.mean(torch.abs(colors - pixels), dim=-1).clone().detach().cpu(),
            #     bins=cfg.bin_size,
            #     range=(0.0, 1.0),
            # )[0]
            err_map = torch.mean(torch.abs(colors - pixels), dim=-1)            # [1,H,W]
            m2d     = binary_mask.squeeze(-1)                                   # [1,H,W]
            vals    = err_map[m2d > 0.5].detach().cpu()
            if vals.numel() == 0:
                vals = torch.zeros(1)  # 안전장치
            info["err"] = torch.histogram(vals, bins=cfg.bin_size, range=(0.0, 1.0))[0]

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
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    if cfg.ubp:
                        not_utilized = self.running_stats["sqrgrad"] < cfg.ubp_thresh
                        is_prune = is_prune | not_utilized
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
                rgb_pred_mask = (
                    (log_pred_mask > 0.5).repeat(1, 1, 1, 3).clone().detach() # [1,H,W,3], predicted mask by mlp
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
                self.render_traj(step, binary_mask)#revised-0924

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

        self.running_stats["hist_err"] = (
            0.95 * self.running_stats["hist_err"] + info["err"]
        )
        mid_err = torch.sum(self.running_stats["hist_err"]) * cfg.robust_percentile
        self.running_stats["avg_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= mid_err)[0][
                0
            ]
        ]

        lower_err = torch.sum(self.running_stats["hist_err"]) * cfg.lower_bound
        upper_err = torch.sum(self.running_stats["hist_err"]) * cfg.upper_bound

        self.running_stats["lower_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= lower_err)[
                0
            ][0]
        ]
        self.running_stats["upper_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= upper_err)[
                0
            ][0]
        ]

        #revised-0919
        if cfg.packed:
            
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None

            #not used for low gaussian density
            # sample mask weight per fragment (same length as grads)
            # if step > cfg.train_cutgrad:
            w = sample_mask_at_means2d(info["means2d"].detach(),binary_mask)  # [nnz] in [0,1]
            w_clamped = w.clamp_(0.0, 1.0)
            # #revised-0921
            w_dyn = 1.0 - w
            self.running_stats["w_static"].index_add_(0, gs_ids,w_clamped)
            self.running_stats["w_dynamic"].index_add_(0, gs_ids,w_dyn)
            # self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1)*w_clamped)
            # self.running_stats["count"].index_add_(0, gs_ids, w_clamped)
            # if cfg.ubp:
            #     self.running_stats["sqrgrad"].index_add_(
            #     0, gs_ids, torch.sum(sqrgrads, dim=-1) * w_clamped
            # )
            # else:

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
            # sample weights at means2d [C,N,2] → [C,N]

            #not used for low gaussian density
            # if step > cfg.train_cutgrad:

            w_full = sample_mask_at_means2d(info["means2d"].detach(),binary_mask)  # [C,N]
            w_sel  = w_full[sel].clamp_(0.0, 1.0)                      # [nnz]
            # #revised-0921
            w_dynn = 1.0 - w_sel
            self.running_stats["w_static"].index_add_(0, gs_ids,w_sel)
            self.running_stats["w_dynamic"].index_add_(0, gs_ids,w_dynn)
            # self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1) * w_sel)
            # self.running_stats["count"].index_add_(0, gs_ids, w_sel)
            # if cfg.ubp:
            #     self.running_stats["sqrgrad"].index_add_(
            #         0, gs_ids, torch.sum(sqrgrads[sel], dim=-1) * w_sel
            #     )
            # else:

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
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            if k == "sqrgrad":
                v_new = torch.ones_like(
                    v_new
                )  # the new ones are assumed to have high utilization in the start
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
                self.running_stats[k] = torch.cat(
                    (v, torch.ones_like(v[sel]))
                )  # new ones are assumed to have high utilization
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
                    test_mask_tensor = F.interpolate(test_mask_tensor.permute(0, 3, 1, 2), size=(h1, w1), mode='nearest')
                    test_mask_tensor = test_mask_tensor.permute(0, 2, 3, 1)
                    
                test_binary_mask = test_mask_tensor.to(device)
                test_binary_mask = test_binary_mask.permute(0, 3, 1, 2)   # [B,1,H,W] changed - different from training
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
                mask_bchw=test_binary_mask.permute(0,3,1,2) if test_binary_mask is not None else None, #revised-0924
                step = step, #revised-0924
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
                mask_bchw= test_binary_mask.permute(0,3,1,2)if test_binary_mask is not None else None, #revised-0924
                step = step, #revised-0924
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
    def render_traj(self, step: int, binary_mask: torch.Tensor ):
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
                mask_bchw=binary_mask.permute(0,3,1,2)if binary_mask is not None else None, #revised-0924
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
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
            mask_bchw=None,  # No mask for viewer
            step=None,       # No step for viewer
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
        # runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(10)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
