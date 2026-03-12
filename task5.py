import argparse
import glob
import json
import os
import sqlite3
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


SCENES = ["barn", "meetingroom", "truck"]


@dataclass
class ColmapModel:
    cameras_txt: Path
    images_txt: Path
    points3d_txt: Path


def camera_intrinsics_from_image(image_path: Path) -> Tuple[float, float, float, float]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image for intrinsics: {image_path}")
    h, w = img.shape
    fx = 0.7 * float(w)
    fy = 0.7 * float(w)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    return fx, fy, cx, cy


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def ensure_frames_for_scene(scene: str, split_a_dir: Path, frames_root: Path, stride: int) -> Path:
    out_dir = frames_root / f"split_a_{scene}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= 2:
        return out_dir

    video_path = split_a_dir / f"split_a_{scene}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing Split A video for scene '{scene}': {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    for idx in range(0, frame_count, max(1, int(stride))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_name = out_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(out_name), frame)
        written += 1
    cap.release()

    if written < 2:
        raise RuntimeError(f"Frame extraction failed for {scene}: only {written} frames written")
    return out_dir


def find_colmap_binary(explicit_path: Optional[str]) -> str:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"COLMAP binary not found at --colmap-bin path: {explicit_path}")
        return str(p)

    which = "where" if os.name == "nt" else "which"
    proc = subprocess.run([which, "colmap"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "COLMAP is not installed or not in PATH. Install COLMAP or pass --colmap-bin <path-to-colmap.exe>."
        )
    line = proc.stdout.strip().splitlines()[0].strip()
    return line


def build_colmap_map(
    colmap_bin: str,
    image_dir: Path,
    scene_out_dir: Path,
    matcher: str,
) -> ColmapModel:
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    db_path = scene_out_dir / "database.db"
    sparse_dir = scene_out_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if len(images) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}")

    fx, fy, cx, cy = camera_intrinsics_from_image(images[0])
    camera_params = f"{fx},{fy},{cx},{cy}"

    if db_path.exists():
        db_path.unlink()

    run_cmd([
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "PINHOLE",
        "--ImageReader.camera_params",
        camera_params,
    ])

    if matcher == "sequential":
        run_cmd([
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
        ])
    else:
        run_cmd([
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
        ])

    run_cmd([
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ])

    candidate_models = sorted([p for p in sparse_dir.glob("*") if p.is_dir()])
    if not candidate_models:
        raise RuntimeError(f"COLMAP mapper did not produce a sparse model for {image_dir}")

    model_dir = candidate_models[0]
    text_dir = scene_out_dir / "model_txt"
    text_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([
        colmap_bin,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ])

    return ColmapModel(
        cameras_txt=text_dir / "cameras.txt",
        images_txt=text_dir / "images.txt",
        points3d_txt=text_dir / "points3D.txt",
    )


def parse_images_txt(path: Path) -> Dict[int, dict]:
    images = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]

        xys_ids = lines[i + 1].split() if i + 1 < len(lines) else []
        points2d = []
        for j in range(0, len(xys_ids) - 2, 3):
            x = float(xys_ids[j])
            y = float(xys_ids[j + 1])
            pid = int(xys_ids[j + 2])
            points2d.append((x, y, pid))

        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "points2d": points2d,
        }
        i += 2
    return images


def parse_points3d_txt(path: Path) -> Dict[int, dict]:
    pts = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.strip().split()
            if len(parts) < 8:
                continue
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            error = float(parts[7])
            track_raw = parts[8:]
            track = []
            for i in range(0, len(track_raw) - 1, 2):
                image_id = int(track_raw[i])
                point2d_idx = int(track_raw[i + 1])
                track.append((image_id, point2d_idx))
            pts[pid] = {"xyz": xyz, "error": error, "track": track}
    return pts


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def load_colmap_descriptors(db_path: Path) -> Dict[int, np.ndarray]:
    if not db_path.exists():
        raise FileNotFoundError(f"COLMAP database missing: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, rows, cols, data FROM descriptors")
    data = {}
    for image_id, rows, cols, blob in cur.fetchall():
        if rows <= 0 or cols <= 0 or blob is None:
            continue
        arr = np.frombuffer(blob, dtype=np.uint8).reshape(rows, cols)
        data[int(image_id)] = arr
    conn.close()
    return data


def build_descriptor_lookup(
    points3d: Dict[int, dict],
    descriptors_by_image: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    descriptor_entries = []
    point_id_entries = []
    point_repr = {}

    for pid, pdata in points3d.items():
        per_point_descs = []
        for image_id, p2d_idx in pdata["track"]:
            desc_img = descriptors_by_image.get(int(image_id))
            if desc_img is None:
                continue
            if 0 <= int(p2d_idx) < desc_img.shape[0]:
                d = desc_img[int(p2d_idx)].astype(np.float32)
                descriptor_entries.append(d)
                point_id_entries.append(int(pid))
                per_point_descs.append(d)

        if per_point_descs:
            point_repr[int(pid)] = np.mean(np.stack(per_point_descs, axis=0), axis=0).astype(np.float32)

    if not descriptor_entries:
        return np.empty((0, 128), dtype=np.float32), np.empty((0,), dtype=np.int64), point_repr

    desc = np.stack(descriptor_entries, axis=0).astype(np.float32)
    point_ids = np.asarray(point_id_entries, dtype=np.int64)
    return desc, point_ids, point_repr


def save_lookup_artifacts(scene_out_dir: Path, points3d: Dict[int, dict], images: Dict[int, dict], desc: np.ndarray, point_ids: np.ndarray) -> None:
    poses = {}
    for image_id, idata in images.items():
        qvec = idata["qvec"]
        tvec = idata["tvec"]
        R = qvec_to_rotmat(qvec)
        C = (-R.T @ tvec.reshape(3, 1)).reshape(-1)
        poses[idata["name"]] = {
            "image_id": int(image_id),
            "qvec": qvec.tolist(),
            "tvec": tvec.tolist(),
            "camera_center": C.tolist(),
        }

    points_json = {str(pid): {"xyz": pdata["xyz"].tolist(), "error": float(pdata["error"])} for pid, pdata in points3d.items()}

    with (scene_out_dir / "poses.json").open("w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    with (scene_out_dir / "points3d.json").open("w", encoding="utf-8") as f:
        json.dump(points_json, f, indent=2)

    np.savez_compressed(
        scene_out_dir / "descriptor_lookup.npz",
        descriptors=desc.astype(np.float32),
        point3d_ids=point_ids.astype(np.int64),
        flann_algorithm=np.array([1], dtype=np.int32),
        flann_trees=np.array([8], dtype=np.int32),
    )


def read_ply_vertices_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY (missing end_header): {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header_txt = b"".join(header_lines).decode("ascii", errors="ignore")
        lines = [ln.strip() for ln in header_txt.splitlines()]

        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_props = []

        for ln in lines:
            if ln.startswith("format "):
                fmt = ln.split()[1]
            elif ln.startswith("element "):
                parts = ln.split()
                in_vertex = (len(parts) >= 3 and parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
            elif ln.startswith("property ") and in_vertex:
                parts = ln.split()
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))

        if fmt is None or vertex_count is None:
            raise RuntimeError(f"Could not parse PLY header: {path}")

        type_map = {
            "char": "b", "int8": "b",
            "uchar": "B", "uint8": "B",
            "short": "h", "int16": "h",
            "ushort": "H", "uint16": "H",
            "int": "i", "int32": "i",
            "uint": "I", "uint32": "I",
            "float": "f", "float32": "f",
            "double": "d", "float64": "d",
        }

        xyz_indices = {}
        for idx, (_, name) in enumerate(vertex_props):
            if name in ("x", "y", "z"):
                xyz_indices[name] = idx

        if not all(k in xyz_indices for k in ("x", "y", "z")):
            raise RuntimeError(f"PLY vertex does not have x/y/z fields: {path}")

        if fmt == "ascii":
            pts = []
            for _ in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(parts) < len(vertex_props):
                    continue
                x = float(parts[xyz_indices["x"]])
                y = float(parts[xyz_indices["y"]])
                z = float(parts[xyz_indices["z"]])
                pts.append([x, y, z])
            return np.asarray(pts, dtype=np.float64)

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise RuntimeError(f"Unsupported PLY format '{fmt}' in {path}")

        endian = "<" if fmt == "binary_little_endian" else ">"
        struct_fmt = endian + "".join(type_map[t] for t, _ in vertex_props)
        row_size = struct.calcsize(struct_fmt)

        pts = []
        for _ in range(vertex_count):
            row = f.read(row_size)
            if len(row) != row_size:
                break
            vals = struct.unpack(struct_fmt, row)
            pts.append([
                float(vals[xyz_indices["x"]]),
                float(vals[xyz_indices["y"]]),
                float(vals[xyz_indices["z"]]),
            ])
        return np.asarray(pts, dtype=np.float64)


def normalize_point_clouds(rec: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rec0 = rec - np.mean(rec, axis=0, keepdims=True)
    gt0 = gt - np.mean(gt, axis=0, keepdims=True)

    rec_scale = float(np.max(np.linalg.norm(rec0, axis=1))) if len(rec0) > 0 else 0.0
    gt_scale = float(np.max(np.linalg.norm(gt0, axis=1))) if len(gt0) > 0 else 0.0
    scale = max(rec_scale, gt_scale, 1e-12)

    return rec0 / scale, gt0 / scale


def chamfer_distance(rec: np.ndarray, gt: np.ndarray) -> float:
    if len(rec) == 0 or len(gt) == 0:
        return float("nan")

    rec_n, gt_n = normalize_point_clouds(rec, gt)
    tree_gt = cKDTree(gt_n)
    tree_rec = cKDTree(rec_n)

    d_rec, _ = tree_gt.query(rec_n, k=1)
    d_gt, _ = tree_rec.query(gt_n, k=1)
    return float(np.mean(d_rec) + np.mean(d_gt))


def save_qualitative_plot(scene_out_dir: Path, rec_pts: np.ndarray, gt_pts: np.ndarray, title: str) -> None:
    if len(rec_pts) == 0 or len(gt_pts) == 0:
        return

    rec_n, gt_n = normalize_point_clouds(rec_pts, gt_pts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    rstep = max(1, len(rec_n) // 12000)
    gstep = max(1, len(gt_n) // 12000)

    ax.scatter(gt_n[::gstep, 0], gt_n[::gstep, 1], gt_n[::gstep, 2], s=1, alpha=0.35, c="gray", label="GT")
    ax.scatter(rec_n[::rstep, 0], rec_n[::rstep, 1], rec_n[::rstep, 2], s=1, alpha=0.6, c="tab:blue", label="COLMAP")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")

    out = scene_out_dir / "qualitative_reconstruction.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def load_custom_metrics(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_with_custom(scene: str, scene_metrics: dict, custom_metrics: Dict[str, dict]) -> dict:
    c = custom_metrics.get(scene, {})

    out = {
        "pose_comparison": "custom pose file not provided",
        "reprojection_error_custom": c.get("mean_reprojection_error"),
        "reprojection_error_colmap": scene_metrics.get("mean_reprojection_error"),
        "num_points_custom": c.get("num_points"),
        "num_points_colmap": scene_metrics.get("num_points"),
        "chamfer_custom": c.get("chamfer_distance"),
        "chamfer_colmap": scene_metrics.get("chamfer_distance"),
    }

    if c.get("poses_path") and os.path.exists(c["poses_path"]):
        out["pose_comparison"] = (
            "custom pose file detected at "
            f"{c['poses_path']} (pose alignment utility not run automatically in this script)"
        )

    return out


def run_scene(
    scene: str,
    args,
    colmap_bin: str,
    custom_metrics: Dict[str, dict],
) -> dict:
    split_a_dir = Path(args.dataset_root) / "Split_A"
    gt_dir = Path(args.dataset_root) / "GT_ply_files"
    task5_root = Path(args.output_root)

    frames_root = task5_root / "frames"
    scene_frames = ensure_frames_for_scene(scene, split_a_dir, frames_root, args.frame_stride)

    scene_out_dir = task5_root / scene
    model = build_colmap_map(colmap_bin, scene_frames, scene_out_dir, args.matcher)

    images = parse_images_txt(model.images_txt)
    points3d = parse_points3d_txt(model.points3d_txt)

    db_path = scene_out_dir / "database.db"
    descriptors_by_image = load_colmap_descriptors(db_path)
    desc, point_ids, _point_repr = build_descriptor_lookup(points3d, descriptors_by_image)

    save_lookup_artifacts(scene_out_dir, points3d, images, desc, point_ids)

    rec_pts = np.stack([v["xyz"] for v in points3d.values()], axis=0) if points3d else np.empty((0, 3), dtype=np.float64)
    gt_name = "Meetingroom.ply" if scene == "meetingroom" else f"{scene.capitalize()}.ply"
    gt_path = gt_dir / gt_name
    gt_pts = read_ply_vertices_xyz(gt_path)

    mean_reproj = float(np.mean([p["error"] for p in points3d.values()])) if points3d else float("nan")
    cd = chamfer_distance(rec_pts, gt_pts)

    save_qualitative_plot(scene_out_dir, rec_pts, gt_pts, f"Task 5 Reconstruction - {scene}")

    metrics = {
        "scene": scene,
        "num_registered_cameras": int(len(images)),
        "num_points": int(len(points3d)),
        "num_descriptor_entries": int(len(desc)),
        "mean_reprojection_error": mean_reproj,
        "chamfer_distance": cd,
        "artifacts": {
            "poses": str(scene_out_dir / "poses.json"),
            "points3d": str(scene_out_dir / "points3d.json"),
            "descriptor_lookup": str(scene_out_dir / "descriptor_lookup.npz"),
            "qualitative_plot": str(scene_out_dir / "qualitative_reconstruction.png"),
        },
    }

    metrics["comparison_with_custom_pipeline"] = compare_with_custom(scene, metrics, custom_metrics)

    with (scene_out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Pre-existing SfM pipeline with COLMAP")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to dataset root (contains Split_A and GT_ply_files)")
    parser.add_argument("--output-root", default="task5_colmap", help="Output directory for Task 5 artifacts")
    parser.add_argument("--scenes", nargs="*", default=SCENES, help="Scenes to process: barn meetingroom truck")
    parser.add_argument("--frame-stride", type=int, default=25, help="Frame stride used only if scene frame folders are missing")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential", help="COLMAP matcher type")
    parser.add_argument("--colmap-bin", default=None, help="Path to colmap executable if not in PATH")
    parser.add_argument(
        "--custom-metrics-json",
        default=None,
        help="Optional JSON with custom pipeline metrics for direct comparison",
    )
    args = parser.parse_args()

    scenes = [s.lower() for s in args.scenes]
    bad = [s for s in scenes if s not in SCENES]
    if bad:
        raise ValueError(f"Invalid scenes: {bad}. Allowed: {SCENES}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    custom_metrics = load_custom_metrics(Path(args.custom_metrics_json) if args.custom_metrics_json else None)

    colmap_bin = find_colmap_binary(args.colmap_bin)

    all_metrics = {}
    for scene in scenes:
        print(f"\n=== Task 5: Processing scene '{scene}' ===")
        scene_metrics = run_scene(scene, args, colmap_bin, custom_metrics)
        all_metrics[scene] = scene_metrics
        print(
            f"Scene {scene}: cams={scene_metrics['num_registered_cameras']} "
            f"pts={scene_metrics['num_points']} "
            f"reproj={scene_metrics['mean_reprojection_error']:.4f} "
            f"chamfer={scene_metrics['chamfer_distance']:.6f}"
        )

    summary_path = out_root / "task5_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nWrote Task 5 summary to {summary_path}")


if __name__ == "__main__":
    main()
