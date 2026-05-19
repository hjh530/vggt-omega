"""Export VGGT-Omega predictions to COLMAP binary format (cameras.bin, images.bin, points3D.bin)."""

import argparse
import os
import struct

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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


# === COLMAP binary write helpers ===

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
}


def write_next_bytes(fid, data, fmt, endian="<"):
    if isinstance(data, (list, tuple)):
        fid.write(struct.pack(endian + fmt, *data))
    else:
        fid.write(struct.pack(endian + fmt, data))


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for cam in cameras.values():
            model_id = CAMERA_MODEL_IDS[cam["model"]]
            write_next_bytes(fid, [cam["id"], model_id, cam["width"], cam["height"]], "iiQQ")
            for p in cam["params"]:
                write_next_bytes(fid, float(p), "d")


def write_images_binary(images, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for img in images.values():
            write_next_bytes(fid, img["id"], "i")
            write_next_bytes(fid, img["qvec"].tolist(), "dddd")
            write_next_bytes(fid, img["tvec"].tolist(), "ddd")
            write_next_bytes(fid, img["camera_id"], "i")
            name_bytes = img["name"].encode("utf-8")
            for b in name_bytes:
                write_next_bytes(fid, bytes([b]), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img["point3D_ids"]), "Q")
            for xy, p3d_id in zip(img["xys"], img["point3D_ids"]):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_binary(points3D, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for pt in points3D.values():
            write_next_bytes(fid, pt["id"], "Q")
            write_next_bytes(fid, pt["xyz"].tolist(), "ddd")
            write_next_bytes(fid, pt["rgb"].tolist(), "BBB")
            write_next_bytes(fid, pt["error"], "d")
            track_length = len(pt["image_ids"])
            write_next_bytes(fid, track_length, "Q")
            for img_id, pt2d_idx in zip(pt["image_ids"], pt["point2D_idxs"]):
                write_next_bytes(fid, [img_id, pt2d_idx], "ii")


def predictions_to_colmap(predictions_path, image_paths, output_dir, image_width, image_height, conf_thres=3.0, stride=8):
    """Convert VGGT-Omega predictions to COLMAP binary model with point cloud from depth."""

    data = np.load(predictions_path)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    depth = data["depth"]
    depth_conf = data["depth_conf"]
    world_points = data.get("world_points", None)

    # Remove batch dim if present
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if depth.ndim == 5:
        depth = depth[0]
    if depth_conf.ndim == 4:
        depth_conf = depth_conf[0]

    N = extrinsics.shape[0]
    pp_h, pp_w = depth.shape[1:3]

    scale_x = image_width / pp_w
    scale_y = image_height / pp_h

    # Write cameras and images
    cameras = {}
    images = {}

    for i in range(N):
        camera_id = i + 1
        image_id = i + 1

        K = intrinsics[i]
        fx = float(K[0, 0]) * scale_x
        fy = float(K[1, 1]) * scale_y
        cx = float(K[0, 2]) * scale_x
        cy = float(K[1, 2]) * scale_y

        cameras[camera_id] = {
            "id": camera_id,
            "model": "PINHOLE",
            "width": image_width,
            "height": image_height,
            "params": [fx, fy, cx, cy],
        }

        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        qvec = rotmat2qvec(R)

        name = os.path.basename(image_paths[i])

        images[image_id] = {
            "id": image_id,
            "qvec": qvec,
            "tvec": t,
            "camera_id": camera_id,
            "name": name,
            "xys": [],
            "point3D_ids": [],
        }

    # Write points3D from depth
    # Generate world points if not precomputed
    if world_points is None:
        print("Computing world points from depth...")
        world_points = []
        for i in tqdm(range(N), desc="Unprojecting"):
            d = depth[i, ..., 0]
            H, W = d.shape
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            fx = float(intrinsics[i, 0, 0])
            fy = float(intrinsics[i, 1, 1])
            cx = float(intrinsics[i, 0, 2])
            cy = float(intrinsics[i, 1, 2])
            cam_pts = np.stack([
                (x - cx) / fx * d,
                (y - cy) / fy * d,
                d,
            ], axis=-1)  # [H, W, 3]
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            wp = np.einsum("ij,hwj->hwi", R.T, cam_pts - t)
            world_points.append(wp)
        world_points = np.stack(world_points, axis=0)

    # Subsample world points and build points3D + image point2D correspondences
    print(f"Building point cloud (stride={stride})...")
    points3D = {}
    point_id = 0

    for i in tqdm(range(N), desc="Points3D"):
        wp = world_points[i]  # [H, W, 3]
        conf = depth_conf[i]
        H, W = wp.shape[:2]

        # Read original image for color sampling
        img_bgr = cv2.imread(image_paths[i])
        if img_bgr is not None:
            img_bgr = cv2.resize(img_bgr, (W, H))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        img_xys = []
        img_p3d_ids = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if conf[y, x] < conf_thres:
                    continue
                p = wp[y, x]
                if not np.isfinite(p).all():
                    continue

                point_id += 1
                # point2D_idx is the index within this image's xys list
                pt2d_idx = len(img_xys)
                img_xys.append([x, y])
                img_p3d_ids.append(point_id)

                points3D[point_id] = {
                    "id": point_id,
                    "xyz": p.astype(np.float64),
                    "rgb": img_rgb[y, x].astype(np.uint8),
                    "error": 1.0,
                    "image_ids": [i + 1],
                    "point2D_idxs": [pt2d_idx],
                }

        # Update image with its points2D
        images[i + 1]["xys"] = np.array(img_xys, dtype=np.float64) if img_xys else np.zeros((0, 2))
        images[i + 1]["point3D_ids"] = np.array(img_p3d_ids, dtype=np.int64) if img_p3d_ids else np.zeros(0, dtype=np.int64)

    os.makedirs(output_dir, exist_ok=True)
    write_cameras_binary(cameras, os.path.join(output_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(output_dir, "images.bin"))
    write_points3D_binary(points3D, os.path.join(output_dir, "points3D.bin"))

    print(f"Exported: {N} cameras, {N} images, {len(points3D)} points → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export VGGT-Omega predictions to COLMAP format")
    parser.add_argument("--predictions", required=True, help="Path to predictions.npz")
    parser.add_argument("--image-dir", required=True, help="Directory containing input images (same as inference)")
    parser.add_argument("--output", required=True, help="Output directory for COLMAP model")
    parser.add_argument("--conf-thres", type=float, default=3.0, help="Depth confidence threshold (default: 3.0)")
    parser.add_argument("--stride", type=int, default=8, help="Point cloud subsampling stride (default: 8)")
    args = parser.parse_args()

    image_paths = scan_image_dir(args.image_dir)
    width, height = get_image_size(image_paths[0])
    print(f"Found {len(image_paths)} images, original size {width}x{height}")

    os.makedirs(args.output, exist_ok=True)
    predictions_to_colmap(
        args.predictions, image_paths, args.output,
        width, height,
        conf_thres=args.conf_thres, stride=args.stride,
    )


if __name__ == "__main__":
    main()
