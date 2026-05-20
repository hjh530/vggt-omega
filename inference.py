"""Local inference script for VGGT-Omega."""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def scan_image_dir(path: str) -> list[str]:
    p = os.path.abspath(path)
    if not os.path.isdir(p):
        raise NotADirectoryError(p)
    files = []
    for f in sorted(os.listdir(p)):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(os.path.join(p, f))
    if not files:
        raise FileNotFoundError(f"No images found in {p}")
    return files


def get_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def main():
    parser = argparse.ArgumentParser(description="VGGT-Omega local inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (.pt)")
    parser.add_argument("--image-dir", required=True, help="Directory containing input images")
    parser.add_argument("--image-resolution", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="max_size")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--output-dir", default="output", help="Base output directory (default: output)")
    parser.add_argument("--output", default=None, help="Output .npz path (auto-derived from image-dir if not set)")
    args = parser.parse_args()

    image_paths = scan_image_dir(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    parent = os.path.dirname(os.path.normpath(args.image_dir))
    folder_name = os.path.basename(parent)
    if args.output is None:
        out_dir = os.path.join(args.output_dir, folder_name)
        args.output = os.path.join(out_dir, "predictions.npz")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    n_total = len(image_paths)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size

    all_extrinsics = []
    all_intrinsics = []
    all_depth = []
    all_depth_conf = []

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_total)
        batch_paths = image_paths[start:end]

        images = load_and_preprocess_images(
            batch_paths, image_resolution=args.image_resolution,
            mode=args.mode, patch_size=args.patch_size
        ).to("cuda")

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        all_extrinsics.append(extrinsics.cpu().numpy())
        all_intrinsics.append(intrinsics.cpu().numpy())
        all_depth.append(predictions["depth"].cpu().numpy())
        all_depth_conf.append(predictions["depth_conf"].cpu().numpy())
        torch.cuda.empty_cache()

    extrinsics = np.concatenate(all_extrinsics, axis=1)
    intrinsics = np.concatenate(all_intrinsics, axis=1)
    depth = np.concatenate(all_depth, axis=1)
    depth_conf = np.concatenate(all_depth_conf, axis=1)

    print(f"Saving to {args.output}")
    np.savez(args.output, depth=depth, depth_conf=depth_conf,
             extrinsics=extrinsics, intrinsics=intrinsics)

    print("Done.")
    print(f"  Depth shape:      {depth.shape}")
    print(f"  Extrinsics shape: {extrinsics.shape}")
    print(f"  Intrinsics shape: {intrinsics.shape}")


if __name__ == "__main__":
    main()
