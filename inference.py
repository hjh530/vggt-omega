"""Local inference script for VGGT-Omega."""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def scan_image_dir(path: str) -> list[str]:
    p = os.path.abspath(path)
    if os.path.isdir(os.path.join(p, "images")):
        p = os.path.join(p, "images")
    if not os.path.isdir(p):
        raise NotADirectoryError(p)
    files = [
        os.path.join(p, f) for f in sorted(os.listdir(p))
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {p}")
    return files


def main():
    parser = argparse.ArgumentParser(description="VGGT-Omega local inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (.pt)")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="max_size")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_paths = scan_image_dir(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    folder_name = os.path.basename(os.path.normpath(args.image_dir))
    if args.output is None:
        out_dir = os.path.join(args.output_dir, folder_name)
        args.output = os.path.join(out_dir, "predictions.npz")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    images = load_and_preprocess_images(
        image_paths, image_resolution=args.image_resolution,
        mode=args.mode, patch_size=args.patch_size
    ).to("cuda")

    print(f"Running inference on {len(image_paths)} images ({tuple(images.shape)})")
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

    print(f"Saving to {args.output}")
    np.savez(args.output, depth=depth, depth_conf=depth_conf,
             extrinsics=ext_np, intrinsics=int_np)

    print("Done.")
    print(f"  Depth:      {depth.shape}")
    print(f"  Extrinsics: {ext_np.shape}")
    print(f"  Intrinsics: {int_np.shape}")


if __name__ == "__main__":
    main()
