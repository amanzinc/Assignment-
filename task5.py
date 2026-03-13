import argparse
import gc
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

try:
    import pycolmap
except Exception:
    pycolmap = None


SCENES = ["barn", "meetingroom", "truck"]


@dataclass
class ColmapModel:
    cameras_txt: Path
    images_txt: Path
    points3d_txt: Path


@dataclass
class PipelineSummary:
    name: str
    points: np.ndarray
    camera_centers: np.ndarray
    mean_reprojection_error: float
    chamfer_distance: float
    num_points: int
    num_cameras: int
    pose_lookup: Dict[str, List[float]]


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


def ensure_frames_for_scene(
    scene: str,
    split_a_dir: Path,
    frames_root: Path,
    stride: int,
    max_output_frames: int = 0,
    force_extract_frames: bool = False,
    resize_width: int = 0,
) -> Path:
    # Reuse pre-extracted frames if they already exist in workspace root.
    if not force_extract_frames:
        workspace_frames = split_a_dir.parent.parent / f"split_a_{scene}_frames"
        if workspace_frames.exists():
            existing_ws = sorted(workspace_frames.glob("*.png"))
            if len(existing_ws) >= 2:
                return workspace_frames

    width_suffix = f"_w{resize_width}" if resize_width > 0 else ""
    dir_suffix = f"split_a_{scene}_frames_s{max(1, int(stride))}{width_suffix}"
    out_dir = frames_root / dir_suffix
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

    written = 0
    stride = max(1, int(stride))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Resize during extraction to avoid writing huge 4K PNGs
        if resize_width > 0:
            h, w = frame.shape[:2]
            if w > resize_width:
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (resize_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        out_name = out_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_name), frame)
        written += 1
        if max_output_frames > 0 and written >= max_output_frames:
            break
        frame_idx += 1
    cap.release()

    if written < 2:
        raise RuntimeError(f"Frame extraction failed for {scene}: only {written} frames written")
    return out_dir


def select_image_names(image_dir: Path, max_images: int) -> List[str]:
    imgs = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    names = [p.name for p in imgs]
    if max_images > 0:
        names = names[:max_images]
    return names


def create_resized_subset(image_dir: Path, output_dir: Path, max_images: int, max_width: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("*.png"):
        old_file.unlink()
    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()
    for old_file in output_dir.glob("*.jpeg"):
        old_file.unlink()

    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if max_images > 0:
        images = images[:max_images]

    written = 0
    for idx, src in enumerate(images):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        if max_width > 0 and w > max_width:
            scale = max_width / float(w)
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

        # JPEG working frames keep Task 5 output size and I/O lower than PNG.
        out_path = output_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1

    if written < 2:
        raise RuntimeError(f"Failed to create resized subset in {output_dir}")

    return output_dir


def frame_sort_key(name: str) -> Tuple[int, str]:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits), name
    return 10**9, name


def resolve_feature_threads(requested_threads: int) -> int:
    # Auto mode (<=0): use half CPU cores capped at 4 for laptop stability.
    if requested_threads <= 0:
        cpu = os.cpu_count() or 2
        return max(1, min(4, cpu // 2 if cpu > 1 else 1))
    return max(1, int(requested_threads))


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


def build_colmap_map_cli(
    colmap_bin: str,
    image_dir: Path,
    scene_out_dir: Path,
    matcher: str,
    use_gpu: bool,
    sequential_overlap: int,
    min_num_matches: int,
    init_num_trials: int,
    mapper_init_min_num_inliers: int,
    mapper_abs_pose_min_num_inliers: int,
    mapper_abs_pose_min_inlier_ratio: float,
    mapper_init_max_error: float,
    mapper_abs_pose_max_error: float,
    mapper_filter_max_reproj_error: float,
    mapper_filter_min_tri_angle: float = 0.5,
    mapper_ba_local_min_tri_angle: float = 2.0,
    mapper_init_min_tri_angle: float = 8.0,
    mapper_max_reg_trials: int = 5,
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
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
    ])

    if matcher == "sequential":
        run_cmd([
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
            "--SequentialMatching.overlap",
            str(max(1, int(sequential_overlap))),
        ])
    else:
        run_cmd([
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
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
        "--Mapper.min_num_matches",
        str(max(1, int(min_num_matches))),
        "--Mapper.init_num_trials",
        str(max(1, int(init_num_trials))),
        "--Mapper.init_min_num_inliers",
        str(max(1, int(mapper_init_min_num_inliers))),
        "--Mapper.abs_pose_min_num_inliers",
        str(max(1, int(mapper_abs_pose_min_num_inliers))),
        "--Mapper.abs_pose_min_inlier_ratio",
        str(float(mapper_abs_pose_min_inlier_ratio)),
        "--Mapper.init_max_error",
        str(float(mapper_init_max_error)),
        "--Mapper.abs_pose_max_error",
        str(float(mapper_abs_pose_max_error)),
        "--Mapper.filter_max_reproj_error",
        str(float(mapper_filter_max_reproj_error)),
        "--Mapper.filter_min_tri_angle",
        str(float(mapper_filter_min_tri_angle)),
        "--Mapper.ba_local_min_tri_angle",
        str(float(mapper_ba_local_min_tri_angle)),
        "--Mapper.init_min_tri_angle",
        str(float(mapper_init_min_tri_angle)),
        "--Mapper.max_reg_trials",
        str(max(1, int(mapper_max_reg_trials))),
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


def build_colmap_map_pycolmap(
    image_dir: Path,
    scene_out_dir: Path,
    matcher: str,
    feature_threads: int,
    max_image_size: int,
    max_num_features: int,
    sequential_overlap: int,
    use_gpu: bool,
    min_num_matches: int,
    init_num_trials: int,
    mapper_init_min_num_inliers: int,
    mapper_abs_pose_min_num_inliers: int,
    mapper_abs_pose_min_inlier_ratio: float,
    mapper_init_max_error: float,
    mapper_abs_pose_max_error: float,
    mapper_filter_max_reproj_error: float,
    mapper_filter_min_tri_angle: float = 0.5,
    mapper_ba_local_min_tri_angle: float = 2.0,
    mapper_init_min_tri_angle: float = 8.0,
    mapper_max_reg_trials: int = 5,
    image_names: Optional[List[str]] = None,
) -> ColmapModel:
    if pycolmap is None:
        raise RuntimeError("pycolmap is not installed. Install it or use --backend colmap.")

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

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = camera_params

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = int(feature_threads)
    extraction_options.max_image_size = int(max_image_size)
    extraction_options.use_gpu = bool(use_gpu)
    extraction_options.sift.max_num_features = int(max_num_features)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = int(feature_threads)
    matching_options.use_gpu = bool(use_gpu)
    matching_options.guided_matching = True

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = max(1, int(sequential_overlap))
    pairing_options.num_threads = int(feature_threads)

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.min_num_matches = max(1, int(min_num_matches))
    mapping_options.init_num_trials = max(1, int(init_num_trials))

    mapper_options = mapping_options.mapper
    mapper_options.init_min_num_inliers = max(1, int(mapper_init_min_num_inliers))
    mapper_options.abs_pose_min_num_inliers = max(1, int(mapper_abs_pose_min_num_inliers))
    mapper_options.abs_pose_min_inlier_ratio = float(mapper_abs_pose_min_inlier_ratio)
    mapper_options.init_max_error = float(mapper_init_max_error)
    mapper_options.abs_pose_max_error = float(mapper_abs_pose_max_error)
    mapper_options.filter_max_reproj_error = float(mapper_filter_max_reproj_error)
    mapper_options.filter_min_tri_angle = float(mapper_filter_min_tri_angle)
    mapper_options.ba_local_min_tri_angle = float(mapper_ba_local_min_tri_angle)
    mapper_options.init_min_tri_angle = float(mapper_init_min_tri_angle)
    mapper_options.max_reg_trials = max(1, int(mapper_max_reg_trials))
    mapper_options.init_max_reg_trials = max(1, int(init_num_trials))

    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_dir),
        image_names=image_names or [],
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    if matcher == "sequential":
        pycolmap.match_sequential(
            database_path=str(db_path),
            matching_options=matching_options,
            pairing_options=pairing_options,
        )
    else:
        pycolmap.match_exhaustive(database_path=str(db_path), matching_options=matching_options)

    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(sparse_dir),
        options=mapping_options,
    )

    if not maps:
        raise RuntimeError(f"pycolmap mapping did not produce a sparse model for {image_dir}")

    best_idx, best_map = max(
        maps.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    text_dir = scene_out_dir / "model_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    best_map.write_text(str(text_dir))

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


def sample_points_uniform(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def read_ply_vertices_xyz(path: Path, max_points: int = 0) -> np.ndarray:
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
            step = max(1, int(np.ceil(vertex_count / max_points))) if max_points > 0 and vertex_count > max_points else 1
            for idx in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(parts) < len(vertex_props):
                    continue
                if idx % step != 0:
                    continue
                x = float(parts[xyz_indices["x"]])
                y = float(parts[xyz_indices["y"]])
                z = float(parts[xyz_indices["z"]])
                pts.append([x, y, z])
            return sample_points_uniform(np.asarray(pts, dtype=np.float64), max_points)

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise RuntimeError(f"Unsupported PLY format '{fmt}' in {path}")

        endian = "<" if fmt == "binary_little_endian" else ">"
        struct_fmt = endian + "".join(type_map[t] for t, _ in vertex_props)
        row_size = struct.calcsize(struct_fmt)
        vertex_data_offset = f.tell()

        if max_points > 0 and vertex_count > max_points:
            target_indices = np.linspace(0, vertex_count - 1, num=max_points, dtype=np.int64)
        else:
            target_indices = np.arange(vertex_count, dtype=np.int64)

        pts = []
        for vertex_idx in target_indices.tolist():
            f.seek(vertex_data_offset + int(vertex_idx) * row_size)
            row = f.read(row_size)
            if len(row) != row_size:
                continue
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


def ordered_colmap_camera_centers(images: Dict[int, dict]) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    ordered = sorted(images.values(), key=lambda item: frame_sort_key(item["name"]))
    centers = []
    pose_lookup = {}
    for item in ordered:
        R = qvec_to_rotmat(item["qvec"])
        C = (-R.T @ item["tvec"].reshape(3, 1)).reshape(-1)
        centers.append(C)
        pose_lookup[item["name"]] = C.tolist()

    if not centers:
        return np.empty((0, 3), dtype=np.float64), pose_lookup
    return np.asarray(centers, dtype=np.float64), pose_lookup


def ordered_custom_camera_centers(sfm_map) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    ordered_ids = sorted(
        [idx for idx in sfm_map.frame_indices if sfm_map.registered_cameras.get(idx, (None, None))[0] is not None]
    )
    centers = []
    pose_lookup = {}
    for idx in ordered_ids:
        R, t = sfm_map.registered_cameras[idx]
        C = (-R.T @ t).reshape(-1)
        name = f"frame_{idx:04d}.png"
        centers.append(C)
        pose_lookup[name] = C.tolist()

    if not centers:
        return np.empty((0, 3), dtype=np.float64), pose_lookup
    return np.asarray(centers, dtype=np.float64), pose_lookup


def normalize_for_visualization(points: np.ndarray, cameras: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stacks = []
    if len(points) > 0:
        stacks.append(points)
    if len(cameras) > 0:
        stacks.append(cameras)
    if not stacks:
        return points.copy(), cameras.copy()

    merged = np.vstack(stacks)
    center = np.mean(merged, axis=0, keepdims=True)
    shifted_points = points - center if len(points) > 0 else points.copy()
    shifted_cameras = cameras - center if len(cameras) > 0 else cameras.copy()

    scale = float(np.max(np.linalg.norm(np.vstack([arr for arr in [shifted_points, shifted_cameras] if len(arr) > 0]), axis=1)))
    scale = max(scale, 1e-12)
    return shifted_points / scale, shifted_cameras / scale


def annotate_sparse_frames(ax, camera_centers: np.ndarray, labels: List[str], step: int) -> None:
    if len(camera_centers) == 0:
        return
    step = max(1, int(step))
    for idx in range(0, len(camera_centers), step):
        ax.text(camera_centers[idx, 0], camera_centers[idx, 2], labels[idx], fontsize=6, alpha=0.8)


def save_trajectory_comparison(scene_out_dir: Path, scene: str, custom: PipelineSummary, colmap: PipelineSummary) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Task 5 - Trajectory Comparison ({scene.title()}, Top-Down View)", fontsize=18, fontweight="bold")

    panels = [
        (axes[0], custom, "Your Pipeline", "#e53935"),
        (axes[1], colmap, "COLMAP", "#e53935"),
    ]

    for ax, summary, title, color in panels:
        _, cams = normalize_for_visualization(np.empty((0, 3), dtype=np.float64), summary.camera_centers)
        labels = sorted(summary.pose_lookup.keys(), key=frame_sort_key)
        ax.plot(cams[:, 0], cams[:, 2], color=color, linewidth=1.6, alpha=0.95, label="Trajectory")
        ax.scatter(cams[:, 0], cams[:, 2], color=color, s=18)
        if len(cams) > 0:
            ax.scatter(cams[0, 0], cams[0, 2], color="green", s=90, label="Start", zorder=5)
            ax.scatter(cams[-1, 0], cams[-1, 2], color="blue", s=90, label="End", zorder=5)
            annotate_sparse_frames(ax, cams, labels, max(1, len(labels) // 12))
        ax.set_title(f"{title} - {summary.num_cameras} cameras", fontsize=16)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (depth)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    out_path = scene_out_dir / "trajectory_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_reconstruction_comparison(scene_out_dir: Path, scene: str, custom: PipelineSummary, colmap: PipelineSummary) -> Path:
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Task 5 - 3D Reconstruction Comparison ({scene.title()})", fontsize=18, fontweight="bold")

    panels = [
        (fig.add_subplot(1, 2, 1, projection="3d"), custom, "Custom Pipeline", "#6baed6", "orange"),
        (fig.add_subplot(1, 2, 2, projection="3d"), colmap, "COLMAP", "#64d98b", "crimson"),
    ]

    for ax, summary, title, pt_color, cam_color in panels:
        pts, cams = normalize_for_visualization(summary.points, summary.camera_centers)
        if len(pts) > 0:
            step = max(1, len(pts) // 12000)
            ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, alpha=0.35, c=pt_color, label=f"{summary.num_points} pts")
        if len(cams) > 0:
            ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=22, c=cam_color, label="cameras")
            ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c=cam_color, linewidth=1.3, alpha=0.85)
        ax.set_title(f"{title} - {summary.num_cameras} cameras, {summary.num_points} pts", fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper left")

    out_path = scene_out_dir / "reconstruction_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_custom_pipeline(scene: str, scene_frames: Path, gt_pts: np.ndarray, args, scene_out_dir: Path) -> PipelineSummary:
    from task1 import get_camera_intrinsics
    from task3 import run_incremental_sfm

    image_paths_all = sorted(scene_frames.glob("*.png")) + sorted(scene_frames.glob("*.jpg")) + sorted(scene_frames.glob("*.jpeg"))
    if args.max_images_per_scene > 0:
        image_paths_all = image_paths_all[:args.max_images_per_scene]
    if len(image_paths_all) < 2:
        raise RuntimeError(f"Need at least 2 images for custom pipeline in {scene_frames}")

    # Keep substantially more frames for custom SfM than before.
    # Previous fixed downsample to ~16 frames could collapse registration to very few cameras.
    # custom_max_images <= 0 means: do not cap custom images.
    if int(args.custom_max_images) <= 0:
        target_custom_images = len(image_paths_all)
    else:
        target_custom_images = max(2, int(args.custom_max_images))
    base_interval = max(1, int(np.ceil(len(image_paths_all) / float(target_custom_images))))

    # Retry with denser sampling if camera registration is too low.
    retry_intervals = [base_interval]
    if base_interval > 1:
        retry_intervals.append(max(1, base_interval // 2))
    if retry_intervals[-1] != 1:
        retry_intervals.append(1)

    best_sfm_map = None
    best_registered = -1

    for interval in retry_intervals:
        image_paths = image_paths_all[::interval]
        if len(image_paths) < 2:
            continue

        print(
            f"Custom pipeline retry: interval={interval} "
            f"frames={len(image_paths)} target_min_cams={int(args.custom_min_registered_cameras)}"
        )

        img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Could not read first custom frame: {image_paths[0]}")
        h, w = img0.shape[:2]
        K = get_camera_intrinsics((h, w))

        sfm_map_try = run_incremental_sfm(
            [str(p) for p in image_paths],
            K,
            live_plot=False,
            plot_every=max(1, int(args.custom_plot_every)),
        )

        reg_try = sum(1 for idx in sfm_map_try.frame_indices if sfm_map_try.registered_cameras.get(idx, (None, None))[0] is not None)
        if reg_try > best_registered:
            best_registered = reg_try
            best_sfm_map = sfm_map_try

        if reg_try >= int(args.custom_min_registered_cameras):
            break

    if best_sfm_map is None:
        raise RuntimeError("Custom pipeline did not produce a valid reconstruction.")
    
    sfm_map = best_sfm_map

    points = np.asarray(sfm_map.points_3d, dtype=np.float64) if sfm_map.points_3d else np.empty((0, 3), dtype=np.float64)
    cameras, pose_lookup = ordered_custom_camera_centers(sfm_map)
    mean_reproj = float(sfm_map.compute_global_reprojection_error()) if len(points) > 0 else float("nan")
    cd = chamfer_distance(points, gt_pts)

    summary = PipelineSummary(
        name="custom",
        points=points,
        camera_centers=cameras,
        mean_reprojection_error=mean_reproj,
        chamfer_distance=cd,
        num_points=int(len(points)),
        num_cameras=int(len(cameras)),
        pose_lookup=pose_lookup,
    )

    with (scene_out_dir / "custom_pipeline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": scene,
                "num_registered_cameras": summary.num_cameras,
                "num_points": summary.num_points,
                "mean_reprojection_error": summary.mean_reprojection_error,
                "chamfer_distance": summary.chamfer_distance,
                "poses": summary.pose_lookup,
            },
            f,
            indent=2,
        )

    return summary


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


def build_colmap_summary(images: Dict[int, dict], points3d: Dict[int, dict], mean_reproj: float, cd: float) -> PipelineSummary:
    points = np.stack([v["xyz"] for v in points3d.values()], axis=0) if points3d else np.empty((0, 3), dtype=np.float64)
    cameras, pose_lookup = ordered_colmap_camera_centers(images)
    return PipelineSummary(
        name="colmap",
        points=points,
        camera_centers=cameras,
        mean_reprojection_error=float(mean_reproj),
        chamfer_distance=float(cd),
        num_points=int(len(points)),
        num_cameras=int(len(cameras)),
        pose_lookup=pose_lookup,
    )


def run_scene(
    scene: str,
    args,
    colmap_bin: Optional[str],
    custom_metrics: Dict[str, dict],
) -> dict:
    split_a_dir = Path(args.dataset_root) / "Split_A"
    gt_dir = Path(args.dataset_root) / "GT_ply_files"
    task5_root = Path(args.output_root)
    scene_out_dir = task5_root / scene

    frames_root = task5_root / "frames"
    scene_frames = ensure_frames_for_scene(
        scene,
        split_a_dir,
        frames_root,
        args.frame_stride,
        args.max_images_per_scene,
        args.force_extract_frames,
        args.working_max_width,
    )
    working_frames = create_resized_subset(
        scene_frames,
        scene_out_dir / "working_frames",
        args.max_images_per_scene,
        args.working_max_width,
    )
    selected_names = select_image_names(working_frames, args.max_images_per_scene)

    if args.backend == "pycolmap":
        try:
            model = build_colmap_map_pycolmap(
                working_frames,
                scene_out_dir,
                args.matcher,
                args.feature_threads,
                args.max_image_size,
                args.max_num_features,
                args.sequential_overlap,
                args.colmap_use_gpu,
                args.min_num_matches,
                args.init_num_trials,
                args.mapper_init_min_num_inliers,
                args.mapper_abs_pose_min_num_inliers,
                args.mapper_abs_pose_min_inlier_ratio,
                args.mapper_init_max_error,
                args.mapper_abs_pose_max_error,
                args.mapper_filter_max_reproj_error,
                args.mapper_filter_min_tri_angle,
                args.mapper_ba_local_min_tri_angle,
                args.mapper_init_min_tri_angle,
                args.mapper_max_reg_trials,
                selected_names,
            )
        except RuntimeError as exc:
            if "did not produce a sparse model" not in str(exc) or args.matcher != "sequential":
                raise
            print("Sequential matching failed to initialize COLMAP; retrying once with exhaustive matching.")
            model = build_colmap_map_pycolmap(
                working_frames,
                scene_out_dir,
                "exhaustive",
                args.feature_threads,
                args.max_image_size,
                args.max_num_features,
                args.sequential_overlap,
                args.colmap_use_gpu,
                args.min_num_matches,
                args.init_num_trials,
                args.mapper_init_min_num_inliers,
                args.mapper_abs_pose_min_num_inliers,
                args.mapper_abs_pose_min_inlier_ratio,
                args.mapper_init_max_error,
                args.mapper_abs_pose_max_error,
                args.mapper_filter_max_reproj_error,
                args.mapper_filter_min_tri_angle,
                args.mapper_ba_local_min_tri_angle,
                args.mapper_init_min_tri_angle,
                args.mapper_max_reg_trials,
                selected_names,
            )
    else:
        if not colmap_bin:
            raise RuntimeError("COLMAP backend selected but no colmap binary is available.")
        model = build_colmap_map_cli(
            colmap_bin,
            working_frames,
            scene_out_dir,
            args.matcher,
            args.colmap_use_gpu,
            args.sequential_overlap,
            args.min_num_matches,
            args.init_num_trials,
            args.mapper_init_min_num_inliers,
            args.mapper_abs_pose_min_num_inliers,
            args.mapper_abs_pose_min_inlier_ratio,
            args.mapper_init_max_error,
            args.mapper_abs_pose_max_error,
            args.mapper_filter_max_reproj_error,
            args.mapper_filter_min_tri_angle,
            args.mapper_ba_local_min_tri_angle,
            args.mapper_init_min_tri_angle,
            args.mapper_max_reg_trials,
        )

    images = parse_images_txt(model.images_txt)
    points3d = parse_points3d_txt(model.points3d_txt)

    db_path = scene_out_dir / "database.db"
    descriptors_by_image = load_colmap_descriptors(db_path)
    desc, point_ids, _point_repr = build_descriptor_lookup(points3d, descriptors_by_image)
    del descriptors_by_image
    gc.collect()

    save_lookup_artifacts(scene_out_dir, points3d, images, desc, point_ids)

    rec_pts = np.stack([v["xyz"] for v in points3d.values()], axis=0) if points3d else np.empty((0, 3), dtype=np.float64)
    gt_name = "Meetingroom.ply" if scene == "meetingroom" else f"{scene.capitalize()}.ply"
    gt_path = gt_dir / gt_name
    gt_pts = read_ply_vertices_xyz(gt_path, args.gt_max_points)

    mean_reproj = float(np.mean([p["error"] for p in points3d.values()])) if points3d else float("nan")
    cd = chamfer_distance(rec_pts, gt_pts)
    colmap_summary = build_colmap_summary(images, points3d, mean_reproj, cd)

    save_qualitative_plot(scene_out_dir, rec_pts, gt_pts, f"Task 5 Reconstruction - {scene}")

    custom_summary = None
    custom_warning = None
    if not args.skip_custom_pipeline:
        try:
            custom_summary = run_custom_pipeline(scene, working_frames, gt_pts, args, scene_out_dir)
        except Exception as exc:
            custom_warning = f"Custom pipeline failed: {exc}"
            print(custom_warning)
            gc.collect()

    trajectory_plot = None
    reconstruction_plot = None
    if custom_summary is not None:
        try:
            trajectory_plot = save_trajectory_comparison(scene_out_dir, scene, custom_summary, colmap_summary)
            reconstruction_plot = save_reconstruction_comparison(scene_out_dir, scene, custom_summary, colmap_summary)
        except Exception as exc:
            custom_warning = f"Comparison plotting failed: {exc}"
            print(custom_warning)
            trajectory_plot = None
            reconstruction_plot = None
            gc.collect()

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
            "trajectory_comparison": str(trajectory_plot) if trajectory_plot else None,
            "reconstruction_comparison": str(reconstruction_plot) if reconstruction_plot else None,
            "custom_pipeline_metrics": str(scene_out_dir / "custom_pipeline_metrics.json") if custom_summary is not None else None,
        },
        "warnings": [custom_warning] if custom_warning else [],
    }

    if custom_summary is not None:
        metrics["comparison_with_custom_pipeline"] = {
            "pose_comparison": "top-down trajectory and 3D reconstruction comparison plots generated",
            "reprojection_error_custom": custom_summary.mean_reprojection_error,
            "reprojection_error_colmap": colmap_summary.mean_reprojection_error,
            "num_points_custom": custom_summary.num_points,
            "num_points_colmap": colmap_summary.num_points,
            "num_cameras_custom": custom_summary.num_cameras,
            "num_cameras_colmap": colmap_summary.num_cameras,
            "chamfer_custom": custom_summary.chamfer_distance,
            "chamfer_colmap": colmap_summary.chamfer_distance,
        }
    else:
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
    parser.add_argument("--force-extract-frames", action="store_true", help="Ignore existing extracted folders and rebuild frames from the source video")
    parser.add_argument("--max-images-per-scene", type=int, default=0, help="Cap number of images processed per scene (<=0 disables cap)")
    parser.add_argument("--working-max-width", type=int, default=960, help="Resize working frames to this width for faster Task 5 runs")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential", help="COLMAP matcher type")
    parser.add_argument("--sequential-overlap", type=int, default=20, help="How many neighboring frames COLMAP matches in sequential mode")
    parser.add_argument("--backend", choices=["pycolmap", "colmap"], default="pycolmap", help="SfM backend")
    parser.add_argument("--colmap-use-gpu", action="store_true", help="Enable COLMAP/pycolmap GPU feature extraction and matching when available")
    parser.add_argument("--feature-threads", type=int, default=0, help="Threads for feature extraction/matching (<=0 auto-selects a safe value)")
    parser.add_argument("--max-image-size", type=int, default=960, help="Max image size used by pycolmap feature extraction")
    parser.add_argument("--max-num-features", type=int, default=4000, help="Max SIFT features per image for pycolmap")
    parser.add_argument("--min-num-matches", type=int, default=8, help="Relaxed minimum number of matches required by the COLMAP mapper")
    parser.add_argument("--init-num-trials", type=int, default=400, help="How many initialization attempts COLMAP makes before giving up")
    parser.add_argument("--mapper-init-min-num-inliers", type=int, default=40, help="Minimum inliers required for choosing the initial image pair")
    parser.add_argument("--mapper-abs-pose-min-num-inliers", type=int, default=24, help="Minimum PnP inliers required to register a new image")
    parser.add_argument("--mapper-abs-pose-min-inlier-ratio", type=float, default=0.12, help="Minimum inlier ratio required to register a new image")
    parser.add_argument("--mapper-init-max-error", type=float, default=8.0, help="Relaxed initialization reprojection error threshold")
    parser.add_argument("--mapper-abs-pose-max-error", type=float, default=14.0, help="Relaxed absolute pose reprojection error threshold")
    parser.add_argument("--mapper-filter-max-reproj-error", type=float, default=8.0, help="Relaxed reprojection error used when filtering 3D points")
    parser.add_argument("--mapper-filter-min-tri-angle", type=float, default=0.5, help="Min triangulation angle (degrees) for keeping 3D points (default COLMAP=1.5)")
    parser.add_argument("--mapper-ba-local-min-tri-angle", type=float, default=2.0, help="Min triangulation angle for local BA (default COLMAP=6.0)")
    parser.add_argument("--mapper-init-min-tri-angle", type=float, default=8.0, help="Min triangulation angle for initialization pair (default COLMAP=16.0)")
    parser.add_argument("--mapper-max-reg-trials", type=int, default=5, help="Max times COLMAP retries registering each image (default COLMAP=3)")
    parser.add_argument("--gt-max-points", type=int, default=200000, help="Max GT points loaded from each mesh for Chamfer/plotting (<=0 loads all)")
    parser.add_argument("--skip-custom-pipeline", action="store_true", help="Skip running the custom pipeline comparison")
    parser.add_argument("--custom-plot-every", type=int, default=10, help="Plot cadence forwarded to custom pipeline when it runs")
    parser.add_argument("--custom-max-images", type=int, default=0, help="Max images used by custom pipeline (<=0 means no cap)")
    parser.add_argument("--custom-min-registered-cameras", type=int, default=12, help="Retry custom pipeline with denser sampling until this camera count is reached")
    parser.add_argument("--colmap-bin", default=None, help="Path to colmap executable if not in PATH")
    parser.add_argument(
        "--custom-metrics-json",
        default=None,
        help="Optional JSON with custom pipeline metrics for direct comparison",
    )
    args = parser.parse_args()
    args.feature_threads = resolve_feature_threads(args.feature_threads)
    print(f"Task 5 feature threads: {args.feature_threads}")

    scenes = [s.lower() for s in args.scenes]
    bad = [s for s in scenes if s not in SCENES]
    if bad:
        raise ValueError(f"Invalid scenes: {bad}. Allowed: {SCENES}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    custom_metrics = load_custom_metrics(Path(args.custom_metrics_json) if args.custom_metrics_json else None)

    colmap_bin = None
    if args.backend == "colmap":
        colmap_bin = find_colmap_binary(args.colmap_bin)
    elif pycolmap is None:
        raise RuntimeError("--backend pycolmap selected but pycolmap is unavailable.")

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
