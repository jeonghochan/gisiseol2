# make ckpt model to ply file extractor
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path
import shutil


def create_sibr_structure(output_dir):
    """Create complete SIBR Viewer compatible directory structure"""
    
    # Create point_cloud/iteration_30000 directory
    point_cloud_dir = os.path.join(output_dir, "point_cloud", "iteration_30000")
    os.makedirs(point_cloud_dir, exist_ok=True)
    
    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create sparse directory (COLMAP format)
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    return point_cloud_dir, images_dir, sparse_dir

def create_dummy_input_ply(output_dir):
    """Create a dummy input.ply file for SIBR compatibility"""
    input_ply_path = os.path.join(output_dir, "input.ply")
    
    # Create a simple PLY with a few points
    dummy_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    with open(input_ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(dummy_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in dummy_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Created dummy input.ply at: {input_ply_path}")

def save_cfg_args(output_dir, source_path="", model_path=""):
    """Save cfg_args file compatible with SIBR Viewer"""
    cfg_args_path = os.path.join(output_dir, "cfg_args")
    
    # Create cfg_args content (Python Namespace string format)
    if not model_path:
        model_path = output_dir
    if not source_path:
        source_path = output_dir
        
    cfg_content = f"Namespace(data_device='cuda', depths='', eval=True, images='images', model_path='{model_path}', resolution=-1, sh_degree=3, source_path='{source_path}', train_test_exp=False, white_background=False)"
    
    with open(cfg_args_path, 'w') as f:
        f.write(cfg_content)
    
    print(f"Saved cfg_args to: {cfg_args_path}")

def save_cameras_json(output_dir):
    """Save cameras.json file (empty array for compatibility)"""
    cameras_path = os.path.join(output_dir, "cameras.json")
    
    # Create basic cameras.json structure
    cameras_data = []  # Empty array - SIBR will use point cloud only
    
    with open(cameras_path, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    
    print(f"Saved cameras.json to: {cameras_path}")


def create_colmap_files(sparse_dir):
    """Create minimal COLMAP files for SIBR compatibility"""
    
    # Create cameras.txt
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 800 600 525.0 525.0 400.0 300.0\n")
    
    # Create images.txt
    images_txt = os.path.join(sparse_dir, "images.txt")
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 dummy.jpg\n")
        f.write("\n")
    
    # Create points3D.txt
    points3d_txt = os.path.join(sparse_dir, "points3D.txt")
    with open(points3d_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("1 0.0 0.0 0.0 255 255 255 0.0 1 0\n")
    
    print(f"Created COLMAP files in: {sparse_dir}")


def save_model2ply(ckpt_path, output_dir, save_config=True, source_path=""):
    """
    Convert .pt checkpoint from spotless_trainer_maskadaptation.py to PLY file.
    This extracts Gaussian Splatting parameters and saves them in PLY format.
    The output will be saved as 'point_cloud.ply' in point_cloud/iteration_30000/ directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create SIBR Viewer compatible directory structure
    point_cloud_dir, images_dir, sparse_dir = create_sibr_structure(output_dir)
    
    # Create full PLY file path (in point_cloud/iteration_30000/)
    ply_path = os.path.join(point_cloud_dir, "point_cloud.ply")
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract splats (Gaussian parameters) from checkpoint
    if 'splats' in checkpoint:
        splats = checkpoint['splats']
    else:
        print("Error: 'splats' key not found in checkpoint")
        sys.exit(1)
    
    # Extract Gaussian parameters
    means3d = splats['means3d'].cpu().numpy()  # [N, 3] positions
    opacities = torch.sigmoid(splats['opacities']).cpu().numpy()  # [N] opacities
    scales = torch.exp(splats['scales']).cpu().numpy()  # [N, 3] scales
    quats = F.normalize(splats['quats'], dim=-1).cpu().numpy()  # [N, 4] quaternions
    
    # Handle Spherical Harmonics coefficients - match garden format (48 total coeffs)
    if 'sh0' in splats and 'shN' in splats:
        # Spherical Harmonics colors - include all coefficients
        sh0 = splats['sh0'].cpu().numpy()  # [N, 1, 3] 
        shN = splats['shN'].cpu().numpy()  # [N, K-1, 3] where K is total SH coefficients
        sh_coeffs = np.concatenate([sh0, shN], axis=1)  # [N, K, 3]
        
        # Pad or truncate to exactly 16 SH coefficients (48 total values: 16*3)
        if sh_coeffs.shape[1] < 16:
            # Pad with zeros
            pad_size = 16 - sh_coeffs.shape[1]
            padding = np.zeros((sh_coeffs.shape[0], pad_size, 3))
            sh_coeffs = np.concatenate([sh_coeffs, padding], axis=1)
        elif sh_coeffs.shape[1] > 16:
            # Truncate
            sh_coeffs = sh_coeffs[:, :16, :]
            
    elif 'colors' in splats:
        # Direct colors - convert to SH format
        colors = torch.sigmoid(splats['colors']).cpu().numpy()  # [N, 3]
        # Create SH coefficients with only DC component
        C0 = 0.28209479177387814
        sh_coeffs = np.zeros((len(colors), 16, 3))  # 16 SH coefficients
        sh_coeffs[:, 0, :] = (colors - 0.5) / C0  # DC component
    else:
        # Default to white if no color information
        colors = np.ones((len(means3d), 3))
        C0 = 0.28209479177387814
        sh_coeffs = np.zeros((len(colors), 16, 3))
        sh_coeffs[:, 0, :] = (colors - 0.5) / C0
    
    # Ensure we have exactly 16 SH coefficients
    assert sh_coeffs.shape[1] == 16, f"Expected 16 SH coefficients, got {sh_coeffs.shape[1]}"
    
    num_sh_coeffs = 16
    num_gaussians = len(means3d)
    print(f"Converting {num_gaussians} Gaussian points to PLY format")
    
    # Create PLY file in binary format (matching garden format)
    import struct
    
    with open(ply_path, 'wb') as f:
        # Write PLY header as text
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {num_gaussians}\n"
        
        # Position properties
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        
        # Normal properties
        header += "property float nx\n"
        header += "property float ny\n"
        header += "property float nz\n"
        
        # Spherical Harmonics coefficients (f_dc and f_rest)
        for i in range(3):  # DC components
            header += f"property float f_dc_{i}\n"
        for i in range(45):  # Rest components (f_rest_0 to f_rest_44)
            header += f"property float f_rest_{i}\n"
        
        # Gaussian properties
        header += "property float opacity\n"
        header += "property float scale_0\n"
        header += "property float scale_1\n"
        header += "property float scale_2\n"
        header += "property float rot_0\n"
        header += "property float rot_1\n"
        header += "property float rot_2\n"
        header += "property float rot_3\n"
        header += "end_header\n"
        
        # Write header as bytes
        f.write(header.encode('ascii'))
        
        # Write vertex data in binary format
        for i in range(num_gaussians):
            # Position (3 floats)
            x, y, z = means3d[i]
            f.write(struct.pack('<fff', float(x), float(y), float(z)))
            
            # Normals (3 floats, zeros for Gaussian splats)
            f.write(struct.pack('<fff', 0.0, 0.0, 0.0))
            
            # SH coefficients
            # DC components (f_dc_0, f_dc_1, f_dc_2)
            for j in range(3):
                f.write(struct.pack('<f', float(sh_coeffs[i, 0, j])))
            
            # Rest components (f_rest_0 to f_rest_44) - 45 values
            for k in range(1, num_sh_coeffs):  # k=1 to 15
                for j in range(3):  # RGB channels
                    f.write(struct.pack('<f', float(sh_coeffs[i, k, j])))
            
            # Gaussian properties (5 floats)
            opacity = opacities[i]
            sx, sy, sz = scales[i]
            qw, qx, qy, qz = quats[i]
            
            f.write(struct.pack('<f', float(opacity)))
            f.write(struct.pack('<fff', float(sx), float(sy), float(sz)))
            f.write(struct.pack('<ffff', float(qw), float(qx), float(qy), float(qz)))
    
    print(f"Successfully saved {num_gaussians} Gaussian points to {ply_path}")
    print(f"PLY file includes 16 SH coefficients per channel (48 total values)")
    print("File format matches garden/point_cloud/iteration_30000/point_cloud.ply (binary format)")

    if save_config:
        # Create input.ply (required by SIBR)
        create_dummy_input_ply(output_dir)
        
        # Create COLMAP files
        create_colmap_files(sparse_dir)
        
        # Save config files with corrected paths
        save_cfg_args(output_dir, source_path or output_dir, output_dir)
        save_cameras_json(output_dir)
        
        print(f"Created complete SIBR Viewer structure:")
        print(f"  - Point cloud: {ply_path}")
        print(f"  - Input mesh: {os.path.join(output_dir, 'input.ply')}")
        print(f"  - COLMAP data: {sparse_dir}")
        print(f"  - Config files: cfg_args, cameras.json")
        print("Compatible with SIBR Viewer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .pt checkpoint to SIBR Viewer compatible PLY file')
    parser.add_argument("-ckpt", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("-o", type=str, required=True, help="Output directory (will save as point_cloud.ply)")
    parser.add_argument("--source_path", type=str, default="", help="Source path for cfg_args (optional)")
    parser.add_argument("--no_config", action='store_true', help="Don't save cfg_args and cameras.json")
    args = parser.parse_args()
    
    ckpt_path = args.ckpt
    output_dir = args.o
    
    # Validate input file
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file {ckpt_path} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Convert checkpoint to PLY
    try:
        save_model2ply(args.ckpt, args.o, save_config=not args.no_config, source_path=args.source_path)
        print(f"Conversion completed successfully!")
        print(f"PLY file saved as: {os.path.join(args.o, 'point_cloud.ply')}")
        if not args.no_config:
            print(f"Config files (cfg_args, cameras.json) also saved to: {args.o}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
