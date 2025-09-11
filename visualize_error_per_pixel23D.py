#visualize error_per_pixel to 3d map
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_residual_surface(npy_path: str, out_dir: str):
    """
    
    
    """
    
    arr = np.load(npy_path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # change to  [H,W] shape

    H, W = arr.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    Z = arr

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Residual Value")

    ax.set_title(os.path.basename(npy_path))
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_zlabel("Residual (error)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(npy_path).replace(".npy", "_surface.png"))
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[Saved] {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", type=str, required=True,
                        help="Directory containing residual .npy files")
    parser.add_argument("--out_dir", type=str, default="residual_plots",
                        help="Directory to save 3D surface plots")
    args = parser.parse_args()
    
    if not args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        
    npy_files = [f for f in os.listdir(args.npy_dir) if f.endswith(".npy")]
    if not npy_files:
        print("No .npy files found in", args.npy_dir)
        return

    for fname in sorted(npy_files):
        npy_path = os.path.join(args.npy_dir, fname)
        plot_residual_surface(npy_path, args.out_dir)

if __name__ == "__main__":
    main()
