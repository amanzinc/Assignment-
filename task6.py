import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from task1 import get_camera_intrinsics
from task3 import run_incremental_sfm
from task5 import (
    build_colmap_map_pycolmap,
    build_descriptor_lookup,
    load_colmap_descriptors,
    parse_images_txt,
    parse_points3d_txt,
    pycolmap,
    qvec_to_rotmat,
)


SCENES = ["barn", "meetingroom", "truck"]


@dataclass
class MapBundle:
    name: str
    descriptors: np.ndarray
    point_ids: np.ndarray
    point_xyz: Dict[int, np.ndarray]
    split_a_centers: np.ndarray
    map_width: int


def list_images(image_dir: Path) -> List[Path]:
    imgs = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    return imgs


def extract_video_frames(video_path: Path, output_dir: Path, stride: int, max_frames: int = 0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list_images(output_dir)
    if len(existing) >= 2:
        return output_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    written = 0
    idx = 0
    stride = max(1, int(stride))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            out_path = output_dir / f"frame_{idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            written += 1
            if max_frames > 0 and written >= max_frames:
                break
        idx += 1
    cap.release()

    if written < 2:
        raise RuntimeError(f"Failed extracting enough frames from {video_path}")
    return output_dir


def resize_image_gray(img_gray: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return img_gray
    h, w = img_gray.shape[:2]
    if w <= max_width:
        return img_gray
    scale = max_width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img_gray, (max_width, nh), interpolation=cv2.INTER_AREA)


def create_resized_copy(src_files: List[Path], output_dir: Path, max_width: int, max_images: int = 0) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()

    if max_images > 0:
        src_files = src_files[:max_images]

    out_files: List[Path] = []
    for i, p in enumerate(src_files):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = resize_image_gray(img, max_width)
        out = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(out), img)
        out_files.append(out)

    if len(out_files) < 2:
        raise RuntimeError(f"Need at least 2 valid images in {output_dir}")
    return out_files


def extract_preexisting_query_features(
    frame_files: List[Path],
    query_dir: Path,
    max_width: int,
    feature_threads: int,
    max_num_features: int,
    use_gpu: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract Split B query features using COLMAP's own extractor via pycolmap.
    Returns: {image_name: (points2d[Nx2], descriptors[Nx128])}
    """
    if pycolmap is None:
        return {}

    query_dir.mkdir(parents=True, exist_ok=True)
    for old in query_dir.glob("*.png"):
        old.unlink()
    for old in query_dir.glob("*.jpg"):
        old.unlink()

    resized_paths: List[Path] = []
    for i, src in enumerate(frame_files):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = resize_image_gray(img, max_width)
        out = query_dir / f"query_{i:04d}.png"
        cv2.imwrite(str(out), img)
        resized_paths.append(out)

    if len(resized_paths) < 1:
        return {}

    db_path = query_dir / "query_features.db"
    if db_path.exists():
        db_path.unlink()

    h, w = cv2.imread(str(resized_paths[0]), cv2.IMREAD_GRAYSCALE).shape[:2]
    fx = 0.7 * float(w)
    fy = 0.7 * float(w)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{fx},{fy},{cx},{cy}"

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = int(max(1, feature_threads))
    extraction_options.max_image_size = int(max_width)
    extraction_options.use_gpu = bool(use_gpu)
    extraction_options.sift.max_num_features = int(max_num_features)

    names = [p.name for p in resized_paths]
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(query_dir),
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    id_to_name: Dict[int, str] = {}
    cur.execute("SELECT image_id, name FROM images")
    for image_id, name in cur.fetchall():
        id_to_name[int(image_id)] = str(name)

    points_by_id: Dict[int, np.ndarray] = {}
    cur.execute("SELECT image_id, rows, cols, data FROM keypoints")
    for image_id, rows, cols, blob in cur.fetchall():
        if rows <= 0 or cols <= 0 or blob is None:
            continue
        arr = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)
        points_by_id[int(image_id)] = arr[:, :2].astype(np.float32)

    desc_by_id: Dict[int, np.ndarray] = {}
    cur.execute("SELECT image_id, rows, cols, data FROM descriptors")
    for image_id, rows, cols, blob in cur.fetchall():
        if rows <= 0 or cols <= 0 or blob is None:
            continue
        arr = np.frombuffer(blob, dtype=np.uint8).reshape(rows, cols)
        desc_by_id[int(image_id)] = arr

    conn.close()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for image_id, name in id_to_name.items():
        pts = points_by_id.get(image_id)
        des = desc_by_id.get(image_id)
        if pts is None or des is None:
            continue
        out[name] = (pts, des)

    return out


def build_custom_map(scene: str, split_a_frames_dir: Path, scene_out_dir: Path, frame_interval: int, map_width: int, max_images: int) -> MapBundle:
    src_files = list_images(split_a_frames_dir)
    if len(src_files) < 2:
        raise RuntimeError(f"Not enough Split A frames for {scene} in {split_a_frames_dir}")

    sampled = src_files[:: max(1, int(frame_interval))]
    custom_frames_dir = scene_out_dir / "custom_map_frames"
    map_files = create_resized_copy(sampled, custom_frames_dir, max_width=map_width, max_images=max_images)

    img0 = cv2.imread(str(map_files[0]), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise RuntimeError(f"Failed to read first custom map frame for {scene}")
    K = get_camera_intrinsics(img0.shape)

    sfm_map = run_incremental_sfm([str(p) for p in map_files], K, live_plot=False, plot_every=10)

    desc_entries: List[np.ndarray] = []
    pid_entries: List[int] = []
    point_xyz: Dict[int, np.ndarray] = {}
    for pid, xyz in enumerate(sfm_map.points_3d):
        point_xyz[int(pid)] = np.asarray(xyz, dtype=np.float32)
        if pid >= len(sfm_map.point_descriptors):
            continue
        for d in sfm_map.point_descriptors[pid]:
            if d is None:
                continue
            desc_entries.append(np.asarray(d, dtype=np.float32))
            pid_entries.append(int(pid))

    if desc_entries:
        descriptors = np.stack(desc_entries, axis=0).astype(np.float32)
        point_ids = np.asarray(pid_entries, dtype=np.int64)
    else:
        descriptors = np.empty((0, 128), dtype=np.float32)
        point_ids = np.empty((0,), dtype=np.int64)

    centers = []
    for idx in sfm_map.frame_indices:
        rt = sfm_map.registered_cameras.get(idx)
        if rt is None or rt[0] is None:
            continue
        R, t = rt
        C = (-R.T @ t).reshape(-1)
        centers.append(C)

    split_a_centers = np.asarray(centers, dtype=np.float64) if centers else np.empty((0, 3), dtype=np.float64)

    return MapBundle(
        name="custom",
        descriptors=descriptors,
        point_ids=point_ids,
        point_xyz=point_xyz,
        split_a_centers=split_a_centers,
        map_width=int(map_width),
    )


def build_preexisting_map(
    scene: str,
    split_a_frames_dir: Path,
    scene_out_dir: Path,
    map_width: int,
    max_images: int,
    feature_threads: int,
    max_num_features: int,
    sequential_overlap: int,
    use_gpu: bool,
) -> MapBundle:
    work_dir = scene_out_dir / "preexisting_working_frames"
    map_frames = create_resized_copy(list_images(split_a_frames_dir), work_dir, max_width=map_width, max_images=max_images)

    colmap_dir = scene_out_dir / "preexisting_map"
    model = build_colmap_map_pycolmap(
        image_dir=work_dir,
        scene_out_dir=colmap_dir,
        matcher="sequential",
        feature_threads=feature_threads,
        max_image_size=map_width,
        max_num_features=max_num_features,
        sequential_overlap=sequential_overlap,
        use_gpu=use_gpu,
        min_num_matches=8,
        init_num_trials=500,
        mapper_init_min_num_inliers=25,
        mapper_abs_pose_min_num_inliers=10,
        mapper_abs_pose_min_inlier_ratio=0.05,
        mapper_init_max_error=12.0,
        mapper_abs_pose_max_error=20.0,
        mapper_filter_max_reproj_error=12.0,
        mapper_filter_min_tri_angle=0.1,
        mapper_ba_local_min_tri_angle=0.1,
        mapper_init_min_tri_angle=2.0,
        mapper_max_reg_trials=10,
    )

    points3d = parse_points3d_txt(model.points3d_txt)
    if len(points3d) == 0:
        raise RuntimeError(
            f"Pre-existing map for scene '{scene}' has 0 reconstructed points. "
            f"Try increasing --preexisting-max-images and/or --sequential-overlap."
        )
    images = parse_images_txt(model.images_txt)
    descriptors_by_image = load_colmap_descriptors(colmap_dir / "database.db")
    descriptors, point_ids, _ = build_descriptor_lookup(points3d, descriptors_by_image)

    point_xyz = {int(pid): np.asarray(pdata["xyz"], dtype=np.float32) for pid, pdata in points3d.items()}

    split_a_centers = []
    for _, idata in sorted(images.items(), key=lambda kv: kv[1]["name"]):
        R = qvec_to_rotmat(idata["qvec"])
        t = idata["tvec"].reshape(3, 1)
        C = (-R.T @ t).reshape(-1)
        split_a_centers.append(C)
    split_a_centers = np.asarray(split_a_centers, dtype=np.float64) if split_a_centers else np.empty((0, 3), dtype=np.float64)

    _ = map_frames  # Keeps intent explicit: map frames are generated and used by COLMAP.

    return MapBundle(
        name="preexisting",
        descriptors=descriptors.astype(np.float32),
        point_ids=point_ids.astype(np.int64),
        point_xyz=point_xyz,
        split_a_centers=split_a_centers,
        map_width=int(map_width),
    )


def match_2d3d(
    kp,
    des: np.ndarray,
    map_bundle: MapBundle,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if des is None or len(des) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0
    if map_bundle.descriptors is None or len(map_bundle.descriptors) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=64))
    raw = flann.knnMatch(des.astype(np.float32), map_bundle.descriptors.astype(np.float32), k=2)

    # Keep best descriptor-space match per 3D point to avoid duplicate 2D-3D constraints.
    best_by_pid: Dict[int, Tuple[float, int]] = {}
    good = 0
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= ratio * n.distance:
            continue
        good += 1
        pid = int(map_bundle.point_ids[m.trainIdx])
        prev = best_by_pid.get(pid)
        if prev is None or m.distance < prev[0]:
            best_by_pid[pid] = (float(m.distance), int(m.queryIdx))

    obj = []
    img = []
    for pid, (_d, qidx) in best_by_pid.items():
        xyz = map_bundle.point_xyz.get(int(pid))
        if xyz is None:
            continue
        obj.append(xyz)
        if isinstance(kp, np.ndarray):
            img.append((float(kp[qidx, 0]), float(kp[qidx, 1])))
        else:
            img.append(kp[qidx].pt)

    if not obj:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), good

    return np.asarray(obj, dtype=np.float32), np.asarray(img, dtype=np.float32), good


def mean_reprojection_error(points3d: np.ndarray, points2d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> float:
    if len(points3d) == 0:
        return float("nan")
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(points3d, rvec, t, K, None)
    proj = proj.reshape(-1, 2)
    errs = np.linalg.norm(proj - points2d, axis=1)
    return float(np.mean(errs))


def localize_scene_frames(
    frame_files: List[Path],
    map_bundle: MapBundle,
    ratio: float,
    min_pnp_inliers: int,
    query_features: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[List[dict], np.ndarray]:
    records: List[dict] = []
    centers = []
    sift = cv2.SIFT_create(nfeatures=2000)

    K: Optional[np.ndarray] = None

    for idx, frame_path in enumerate(frame_files):
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = resize_image_gray(gray, map_bundle.map_width)

        if K is None:
            K = get_camera_intrinsics(gray.shape)

        if query_features is not None and frame_path.name in query_features:
            kp, des = query_features[frame_path.name]
        else:
            kp, des = sift.detectAndCompute(gray, None)
            if kp is None:
                kp = []

        obj_pts, img_pts, good_desc_matches = match_2d3d(kp, des, map_bundle, ratio)
        total_corr = int(len(obj_pts))

        rec = {
            "frame_index": int(idx),
            "frame_name": frame_path.name,
            "num_keypoints": int(len(kp)),
            "num_descriptor_matches": int(good_desc_matches),
            "num_2d3d_correspondences": int(total_corr),
            "success": False,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_reprojection_error": None,
            "camera_center": None,
        }

        if total_corr < 6:
            records.append(rec)
            continue

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            None,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if (not ok) or inliers is None or len(inliers) < min_pnp_inliers:
            records.append(rec)
            continue

        inl = inliers.reshape(-1)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        inlier_obj = obj_pts[inl]
        inlier_img = img_pts[inl]
        reproj = mean_reprojection_error(inlier_obj, inlier_img, R, t, K)

        C = (-R.T @ t).reshape(-1)
        centers.append(C)

        rec["success"] = True
        rec["num_inliers"] = int(len(inl))
        rec["inlier_ratio"] = float(len(inl) / max(1, total_corr))
        rec["mean_reprojection_error"] = float(reproj)
        rec["camera_center"] = [float(C[0]), float(C[1]), float(C[2])]
        records.append(rec)

    centers_arr = np.asarray(centers, dtype=np.float64) if centers else np.empty((0, 3), dtype=np.float64)
    return records, centers_arr


def summarize_localization(records: List[dict]) -> dict:
    ok = [r for r in records if r["success"]]
    fail_rate = 1.0 - (len(ok) / max(1, len(records)))

    if ok:
        reproj = np.asarray([r["mean_reprojection_error"] for r in ok], dtype=np.float64)
        ratios = np.asarray([r["inlier_ratio"] for r in ok], dtype=np.float64)
        stats = {
            "num_frames": int(len(records)),
            "num_success": int(len(ok)),
            "failure_rate": float(fail_rate),
            "reprojection_error_mean": float(np.mean(reproj)),
            "reprojection_error_std": float(np.std(reproj)),
            "inlier_ratio_mean": float(np.mean(ratios)),
            "inlier_ratio_std": float(np.std(ratios)),
        }
    else:
        stats = {
            "num_frames": int(len(records)),
            "num_success": 0,
            "failure_rate": float(fail_rate),
            "reprojection_error_mean": None,
            "reprojection_error_std": None,
            "inlier_ratio_mean": None,
            "inlier_ratio_std": None,
        }
    return stats


def _series_from_records(records: List[dict], key: str) -> np.ndarray:
    vals = []
    for r in records:
        if r["success"] and r[key] is not None:
            vals.append(float(r[key]))
        else:
            vals.append(np.nan)
    return np.asarray(vals, dtype=np.float64)


def save_scene_plots(
    scene_out_dir: Path,
    scene: str,
    custom_records: List[dict],
    pre_records: List[dict],
    custom_bundle: MapBundle,
    pre_bundle: MapBundle,
    custom_split_b_centers: np.ndarray,
    pre_split_b_centers: np.ndarray,
) -> None:
    x = np.arange(max(len(custom_records), len(pre_records)))

    c_err = _series_from_records(custom_records, "mean_reprojection_error")
    p_err = _series_from_records(pre_records, "mean_reprojection_error")

    c_inl = _series_from_records(custom_records, "inlier_ratio")
    p_inl = _series_from_records(pre_records, "inlier_ratio")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    if len(c_err) > 0:
        ax1.plot(np.arange(len(c_err)), c_err, label="Custom map", color="tab:blue", linewidth=2)
    if len(p_err) > 0:
        ax1.plot(np.arange(len(p_err)), p_err, label="Pre-existing map", color="tab:orange", linewidth=2)
    ax1.set_title(f"Task 6 Reprojection Error - {scene}")
    ax1.set_xlabel("Split B frame index")
    ax1.set_ylabel("Mean reprojection error (px)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(scene_out_dir / "reprojection_error_vs_frame.png", dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if len(c_inl) > 0:
        ax2.plot(np.arange(len(c_inl)), c_inl, label="Custom map", color="tab:blue", linewidth=2)
    if len(p_inl) > 0:
        ax2.plot(np.arange(len(p_inl)), p_inl, label="Pre-existing map", color="tab:orange", linewidth=2)
    ax2.set_title(f"Task 6 Inlier Ratio - {scene}")
    ax2.set_xlabel("Split B frame index")
    ax2.set_ylabel("Inlier ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(scene_out_dir / "inlier_ratio_vs_frame.png", dpi=160)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(12, 5))
    axl = fig3.add_subplot(1, 2, 1, projection="3d")
    axr = fig3.add_subplot(1, 2, 2, projection="3d")

    if len(custom_bundle.split_a_centers) > 0:
        axl.plot(custom_bundle.split_a_centers[:, 0], custom_bundle.split_a_centers[:, 1], custom_bundle.split_a_centers[:, 2],
                 color="tab:blue", linewidth=2, label="Split A map")
    if len(custom_split_b_centers) > 0:
        axl.plot(custom_split_b_centers[:, 0], custom_split_b_centers[:, 1], custom_split_b_centers[:, 2],
                 color="tab:green", linewidth=2, label="Split B localized")
    axl.set_title("Custom map trajectory")
    axl.set_xlabel("X")
    axl.set_ylabel("Y")
    axl.set_zlabel("Z")
    axl.legend(loc="upper right")

    if len(pre_bundle.split_a_centers) > 0:
        axr.plot(pre_bundle.split_a_centers[:, 0], pre_bundle.split_a_centers[:, 1], pre_bundle.split_a_centers[:, 2],
                 color="tab:orange", linewidth=2, label="Split A map")
    if len(pre_split_b_centers) > 0:
        axr.plot(pre_split_b_centers[:, 0], pre_split_b_centers[:, 1], pre_split_b_centers[:, 2],
                 color="tab:green", linewidth=2, label="Split B localized")
    axr.set_title("Pre-existing map trajectory")
    axr.set_xlabel("X")
    axr.set_ylabel("Y")
    axr.set_zlabel("Z")
    axr.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(scene_out_dir / "trajectory_splitA_splitB.png", dpi=160)
    plt.close(fig3)


def run_scene(scene: str, args: argparse.Namespace, out_root: Path) -> dict:
    scene_out = out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    split_a_frames = Path(f"split_a_{scene}_frames")
    if not split_a_frames.exists():
        raise FileNotFoundError(
            f"Missing extracted Split A frames for {scene}: {split_a_frames}. "
            f"Extract frames first (or add extraction for Split A)."
        )

    split_b_video = Path("Dataset") / "Split_B" / f"split_b_{scene}.mp4"
    if not split_b_video.exists():
        raise FileNotFoundError(f"Missing Split B video for {scene}: {split_b_video}")

    split_b_frames = out_root / "frames" / f"split_b_{scene}_s{max(1, int(args.split_b_stride))}"
    extract_video_frames(split_b_video, split_b_frames, stride=args.split_b_stride, max_frames=args.max_split_b_frames)
    split_b_files = list_images(split_b_frames)

    custom_bundle = build_custom_map(
        scene=scene,
        split_a_frames_dir=split_a_frames,
        scene_out_dir=scene_out,
        frame_interval=args.custom_frame_interval,
        map_width=args.custom_map_width,
        max_images=args.custom_max_images,
    )

    pre_bundle = build_preexisting_map(
        scene=scene,
        split_a_frames_dir=split_a_frames,
        scene_out_dir=scene_out,
        map_width=args.preexisting_map_width,
        max_images=args.preexisting_max_images,
        feature_threads=args.feature_threads,
        max_num_features=args.max_num_features,
        sequential_overlap=args.sequential_overlap,
        use_gpu=args.use_gpu,
    )

    pre_query_dir = scene_out / "preexisting_query_frames"
    pre_query_features = extract_preexisting_query_features(
        split_b_files,
        query_dir=pre_query_dir,
        max_width=pre_bundle.map_width,
        feature_threads=args.feature_threads,
        max_num_features=args.max_num_features,
        use_gpu=args.use_gpu,
    )
    # Remap COLMAP query filenames to original Split B frame names by frame order.
    if pre_query_features:
        sorted_query_names = sorted(pre_query_features.keys())
        remapped: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for i, frame_path in enumerate(split_b_files):
            if i < len(sorted_query_names):
                remapped[frame_path.name] = pre_query_features[sorted_query_names[i]]
        pre_query_features = remapped

    custom_records, custom_split_b_centers = localize_scene_frames(
        split_b_files,
        custom_bundle,
        ratio=args.ratio_test,
        min_pnp_inliers=args.min_pnp_inliers,
    )
    pre_records, pre_split_b_centers = localize_scene_frames(
        split_b_files,
        pre_bundle,
        ratio=args.ratio_test,
        min_pnp_inliers=args.min_pnp_inliers,
        query_features=pre_query_features,
    )

    with (scene_out / "custom_localization.json").open("w", encoding="utf-8") as f:
        json.dump(custom_records, f, indent=2)
    with (scene_out / "preexisting_localization.json").open("w", encoding="utf-8") as f:
        json.dump(pre_records, f, indent=2)

    custom_summary = summarize_localization(custom_records)
    pre_summary = summarize_localization(pre_records)

    save_scene_plots(
        scene_out,
        scene,
        custom_records,
        pre_records,
        custom_bundle,
        pre_bundle,
        custom_split_b_centers,
        pre_split_b_centers,
    )

    result = {
        "scene": scene,
        "custom": custom_summary,
        "preexisting": pre_summary,
        "artifacts": {
            "custom_localization": str(scene_out / "custom_localization.json"),
            "preexisting_localization": str(scene_out / "preexisting_localization.json"),
            "reprojection_plot": str(scene_out / "reprojection_error_vs_frame.png"),
            "inlier_plot": str(scene_out / "inlier_ratio_vs_frame.png"),
            "trajectory_plot": str(scene_out / "trajectory_splitA_splitB.png"),
        },
    }

    with (scene_out / "task6_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6: Camera Localization on Split B")
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--output-root", default="task6_localization")

    parser.add_argument("--split-b-stride", type=int, default=10, help="Frame extraction stride for Split B videos")
    parser.add_argument("--max-split-b-frames", type=int, default=0, help="Optional cap on extracted Split B frames (0 = all)")

    parser.add_argument("--custom-frame-interval", type=int, default=10, help="Use every Nth Split A frame for building custom map")
    parser.add_argument("--custom-map-width", type=int, default=640, help="Resize width for custom map + localization frames")
    parser.add_argument("--custom-max-images", type=int, default=80, help="Max Split A images for custom map (0 = all)")

    parser.add_argument("--preexisting-map-width", type=int, default=960, help="Resize width for pre-existing map + localization frames")
    parser.add_argument("--preexisting-max-images", type=int, default=80, help="Max Split A images for pre-existing map (0 = all)")
    parser.add_argument("--feature-threads", type=int, default=2)
    parser.add_argument("--max-num-features", type=int, default=4000)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--use-gpu", action="store_true")

    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-pnp-inliers", type=int, default=6)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for scene in args.scenes:
        print(f"\n=== Task 6 | Scene: {scene} ===")
        result = run_scene(scene, args, out_root)
        all_results.append(result)
        print(f"Custom success: {result['custom']['num_success']}/{result['custom']['num_frames']} | "
              f"failure rate {result['custom']['failure_rate']:.3f}")
        print(f"Pre-existing success: {result['preexisting']['num_success']}/{result['preexisting']['num_frames']} | "
              f"failure rate {result['preexisting']['failure_rate']:.3f}")

    summary_path = out_root / "task6_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved Task 6 summary to {summary_path}")


if __name__ == "__main__":
    main()
