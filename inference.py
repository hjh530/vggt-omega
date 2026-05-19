"""Local inference script for VGGT-Omega with batching & world-space point cloud."""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def unproject_depth_to_world(depth, extrinsics, intrinsics):
    """Unproject depth maps to world coordinates.

    Args:
        depth: [N, H, W, 1]
        extrinsics: [N, 3, 4] world-to-camera
        intrinsics: [N, 3, 3]
    Returns:
        world_points: [N, H, W, 3]
    """
    depth = depth[..., 0]
    N, H, W = depth.shape

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = np.broadcast_to(x[None], (N, H, W))
    y = np.broadcast_to(y[None], (N, H, W))

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    cam_points = np.stack([
        (x - cx) / fx * depth,
        (y - cy) / fy * depth,
        depth,
    ], axis=-1)

    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    world_points = np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)), cam_points - t[:, None, None, :])
    return world_points


def main():
    parser = argparse.ArgumentParser(description="VGGT-Omega local inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (.pt)")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--image-resolution", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced",
                        help="Preprocessing mode: balanced or max_size (default: balanced)")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size (default: 16)")
    parser.add_argument("--batch-size", type=int, default=200, help="Images per forward pass (default: 200, max ~24GB VRAM)")
    parser.add_argument("--output", default="output.npz", help="Output .npz file (default: output.npz)")
    args = parser.parse_args()

    for path in args.images:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    all_images = sorted(args.images)
    n_total = len(all_images)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size

    all_extrinsics = []
    all_intrinsics = []
    all_depth = []
    all_depth_conf = []
    all_world_points = []

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_total)
        batch_paths = all_images[start:end]
        n_batch = len(batch_paths)

        print(f"  Batch {batch_idx + 1}/{n_batches}: {n_batch} images")

        images = load_and_preprocess_images(batch_paths, image_resolution=args.image_resolution, mode=args.mode, patch_size=args.patch_size).to("cuda")

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        depth = predictions["depth"].cpu().numpy()
        depth_conf = predictions["depth_conf"].cpu().numpy()
        ext_np = extrinsics.cpu().numpy()
        int_np = intrinsics.cpu().numpy()

        # Unproject depth to world coordinates
        batch_world_points = []
        for i in range(n_batch):
            wp = unproject_depth_to_world(depth[0, i:i+1], ext_np[0, i:i+1], int_np[0, i:i+1])
            batch_world_points.append(wp[0])
        batch_world_points = np.stack(batch_world_points, axis=0)

        all_extrinsics.append(ext_np)
        all_intrinsics.append(int_np)
        all_depth.append(depth)
        all_depth_conf.append(depth_conf)
        all_world_points.append(batch_world_points)

        torch.cuda.empty_cache()

    # Concatenate all batches
    extrinsics = np.concatenate(all_extrinsics, axis=1)
    intrinsics = np.concatenate(all_intrinsics, axis=1)
    depth = np.concatenate(all_depth, axis=1)
    depth_conf = np.concatenate(all_depth_conf, axis=1)
    world_points = np.concatenate(all_world_points, axis=0)

    print(f"Saving results to {args.output}")
    np.savez(
        args.output,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        world_points=world_points,
    )

    print("Done.")
    print(f"  Depth shape:         {depth.shape}")
    print(f"  Depth conf shape:    {depth_conf.shape}")
    print(f"  Extrinsics shape:    {extrinsics.shape}")
    print(f"  Intrinsics shape:    {intrinsics.shape}")
    print(f"  World points shape:  {world_points.shape}")


if __name__ == "__main__":
    main()
