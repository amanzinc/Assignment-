import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend before any other matplotlib import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

import cv2
import numpy as np
import argparse
import os
import glob
import re
from scipy.optimize import least_squares

# Import necessary functions from Task 1 and Task 2
from task1 import extract_features_and_match, get_camera_intrinsics
from task2 import linear_triangulation, compute_reprojection_error, disambiguate_pose

# We need some helper functions from task2 but they might not be exposed directly in a way we can use easily
# So I will re-implement or adapt slightly for the map structure.

class SfMMap:
    def __init__(self, K):
        self.K = K
        self.points_3d = []  # List of [x, y, z] coordinates
        self.point_descriptors = []  # List of lists of descriptors for each point
        # List of lists of observations: each 3D point has a list of (frame_idx, keypoint_idx)
        self.point_observations = [] 
        
        self.registered_cameras = {}  # {frame_idx: (R, t)}
        self.frame_data = {}  # {frame_idx: {'kp': keypoints, 'des': descriptors, 'img': image_path}}
        self.frame_indices = [] # Ordered list of processed frame indices

        # Lookup table for faster access: frame_idx -> keypoint_idx -> point3d_idx
        # If point_idx is -1, it means this keypoint is not associated with any 3D point yet
        self.feature_to_point_map = {} 

        # Per-frame progress history for summary plots.
        self.stats_frames = []
        self.stats_registered_cameras = []
        self.stats_points = []
        self.stats_mean_reproj = []

    def add_camera(self, frame_idx, R, t):
        self.registered_cameras[frame_idx] = (R, t)
        if frame_idx not in self.feature_to_point_map:
             # Initialize mapping for this frame with -1 (no association)
            num_kps = len(self.frame_data[frame_idx]['kp'])
            self.feature_to_point_map[frame_idx] = np.full(num_kps, -1, dtype=int)

    def add_point(self, point_3d, descriptors, observations):
        """
        Adds a new 3D point to the map.
        point_3d: [x, y, z]
        descriptors: list of descriptors (one or more)
        observations: list of (frame_idx, kp_idx)
        """
        point_idx = len(self.points_3d)
        self.points_3d.append(point_3d)
        self.point_descriptors.append(descriptors)
        self.point_observations.append(observations)
        
        # Update the reverse mapping
        for frame_idx, kp_idx in observations:
            if frame_idx in self.feature_to_point_map:
                self.feature_to_point_map[frame_idx][kp_idx] = point_idx
                
        return point_idx

    def get_camera_projection_matrix(self, frame_idx):
        if frame_idx not in self.registered_cameras:
            return None
        R, t = self.registered_cameras[frame_idx]
        if R is None or t is None:
            return None
        return self.K @ np.hstack((R, t))

    def compute_global_reprojection_error(self):
        total_error = 0.0
        total_obs = 0
        from task2 import compute_reprojection_error
        
        # Group points by camera to compute errors efficiently
        points_by_cam = {}
        for pt_idx, obs_list in enumerate(self.point_observations):
            for frame_idx, kp_idx in obs_list:
                if frame_idx not in points_by_cam:
                    points_by_cam[frame_idx] = {'pts3d': [], 'pts2d': []}
                points_by_cam[frame_idx]['pts3d'].append(self.points_3d[pt_idx])
                # We need the actual 2D coordinates
                kp = self.frame_data[frame_idx]['kp'][kp_idx].pt
                points_by_cam[frame_idx]['pts2d'].append(kp)
                
        for frame_idx, data in points_by_cam.items():
            P = self.get_camera_projection_matrix(frame_idx)
            if P is None: continue
            
            pts3d = np.array(data['pts3d'])
            pts2d = np.array(data['pts2d'])
            
            err = compute_reprojection_error(pts3d, P, pts2d)
            # err is a 1D array of size 2*N
            err_norms = np.linalg.norm(err.reshape(-1, 2), axis=1)
            total_error += np.sum(err_norms)
            total_obs += len(pts3d)
            
        if total_obs == 0: return 0.0
        return total_error / total_obs

    def record_stats(self, frame_idx, mean_err):
        valid_cams = sum(1 for R, t in self.registered_cameras.values() if R is not None)
        self.stats_frames.append(int(frame_idx))
        self.stats_registered_cameras.append(int(valid_cams))
        self.stats_points.append(int(len(self.points_3d)))
        self.stats_mean_reproj.append(float(mean_err))


def pose_reprojection_residuals(params, points_3d, points_2d, K):
    """
    Residual function for pose refinement (motion-only bundle adjustment).
    params: [om_x, om_y, om_z, tx, ty, tz] (Rotation vector + translation)
    """
    r_vec = params[:3]
    t_vec = params[3:]
    
    # Project 3D points
    points_2d_proj, _ = cv2.projectPoints(points_3d, r_vec, t_vec, K, None)
    points_2d_proj = points_2d_proj.reshape(-1, 2)
    
    residuals = (points_2d_proj - points_2d).flatten()
    return residuals

def refine_pose(R, t, points_3d, points_2d, K):
    """
    Refines camera pose using non-linear least squares.
    points_3d: Nx3 array of 3D points
    points_2d: Nx2 array of corresponding 2D observations
    """
    if len(points_3d) < 4:
        return R, t
        
    r_vec, _ = cv2.Rodrigues(R)
    params_init = np.hstack((r_vec.flatten(), t.flatten()))
    
    res = least_squares(
        pose_reprojection_residuals,
        params_init,
        args=(points_3d, points_2d, K),
        method='lm',
        ftol=1e-15,
        xtol=1e-15,
        max_nfev=50
    )
    
    params_opt = res.x
    r_vec_opt = params_opt[:3]
    t_opt = params_opt[3:].reshape(3, 1)
    R_opt, _ = cv2.Rodrigues(r_vec_opt)
    
    return R_opt, t_opt

def _extract_frame_number(file_path):
    base = os.path.basename(file_path)
    match = re.search(r"(\d+)(?=\.[^.]+$)", base)
    if match:
        return int(match.group(1))
    return None

def _infer_input_frame_step(files):
    nums = [_extract_frame_number(f) for f in files]
    nums = [n for n in nums if n is not None]

    if len(nums) < 2:
        return 1

    diffs = [b - a for a, b in zip(nums[:-1], nums[1:]) if b > a]
    if not diffs:
        return 1

    return max(1, int(round(float(np.median(diffs)))))

def load_frames(dataset_path, frame_interval=25):
    # Depending on structure, assuming split A frames are images or video
    # Check if directory contains images
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dataset_path, ext)))
    
    # If no images, check if it's a video file (as per task 1/2 examples)
    # But usually SfM works on a sequence. We'll assume the user provides a folder or video.
    # For now, let's assume the user extracted frames or provides a pattern.
    # The workspace shows extract_frames.py, suggesting we might need to extract them first.
    # However, for Task 3 we need "regular intervals".
    # I will assume the user provides a directory of ordered frames.
    
    files.sort()

    input_step = _infer_input_frame_step(files)
    target_interval = max(1, int(frame_interval))

    # Sample in units of original video frames, not file index, to avoid double-subsampling
    # when input has already been extracted at a coarse stride (e.g., every 25th frame).
    if input_step >= target_interval:
        stride_in_files = 1
    else:
        stride_in_files = max(1, int(round(target_interval / float(input_step))))

    selected = files[::stride_in_files]
    effective_interval = input_step * stride_in_files
    return selected, input_step, effective_interval, stride_in_files

def write_resized_frame_subset(image_files, output_dir, max_width):
    """
    Writes a resized copy of each selected frame and returns new file paths.
    If max_width <= 0, returns the original file list unchanged.
    """
    if max_width <= 0:
        return image_files

    os.makedirs(output_dir, exist_ok=True)

    # Ensure old files do not mix with this run's subset.
    for old_file in glob.glob(os.path.join(output_dir, "*.png")):
        os.remove(old_file)

    resized_files = []
    for i, file_path in enumerate(image_files):
        img = cv2.imread(file_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, img)
        resized_files.append(out_path)

    return resized_files

def match_features_knn(des1, des2, ratio=0.75):
    # Use FLANN matching for speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def _set_equal_3d_limits(ax, mins, maxs):
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

def _compute_dense_view_bounds(points_3d):
    if len(points_3d) == 0:
        return None

    pts = np.asarray(points_3d, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return None

    if len(pts) <= 50:
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        padding = np.maximum((maxs - mins) * 0.1, 1e-3)
        return mins - padding, maxs + padding

    bins = max(8, min(24, int(round(len(pts) ** (1.0 / 3.0)))))
    hist, edges = np.histogramdd(pts, bins=bins)
    peak_idx = np.unravel_index(np.argmax(hist), hist.shape)

    local_mask = np.ones(len(pts), dtype=bool)
    for axis in range(3):
        low_bin = max(0, peak_idx[axis] - 1)
        high_bin = min(len(edges[axis]) - 2, peak_idx[axis] + 1)
        low_edge = edges[axis][low_bin]
        high_edge = edges[axis][high_bin + 1]
        local_mask &= (pts[:, axis] >= low_edge) & (pts[:, axis] <= high_edge)

    local_pts = pts[local_mask]
    if len(local_pts) < 10:
        local_pts = pts

    mins = np.percentile(local_pts, 5, axis=0)
    maxs = np.percentile(local_pts, 95, axis=0)
    center = (mins + maxs) / 2.0
    half_span = np.maximum((maxs - mins) * 0.65, 1e-2)
    radius = max(float(np.max(half_span)), 1e-2)

    return center - radius, center + radius

def update_live_plot(sfm_map, ax, fig):
    ax.clear()
    pts = np.array(sfm_map.points_3d)
    cams = []
    for idx in sfm_map.frame_indices:
        if sfm_map.registered_cameras.get(idx, (None, None))[0] is None:
            continue
        R, t = sfm_map.registered_cameras[idx]
        C = -R.T @ t
        cams.append(C.flatten())
    cams = np.array(cams)
    
    if len(pts) > 0:
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1, c='steelblue', alpha=0.5, label='Structure')

    if len(cams) > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2],
                   s=40, c='red', marker='^', zorder=5, label='Cameras')
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c='red', alpha=0.6, linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=8)

    live_bounds = _compute_dense_view_bounds(pts)
    if live_bounds is not None:
        _set_equal_3d_limits(ax, live_bounds[0], live_bounds[1])

    valid_cams = sum(1 for R, t in sfm_map.registered_cameras.values() if R is not None)
    ax.set_title(f'Incremental SfM — live\nPoints: {len(pts)} | Cameras: {valid_cams}', fontsize=11)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)   # give TkAgg event loop time to render the frame

def run_incremental_sfm(image_files, K, live_plot=True, plot_every=1):
    # Initialize Map
    sfm_map = SfMMap(K)
    
    # Initialize live plotting
    fig = None
    ax = None
    if live_plot:
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.manager.set_window_title('Incremental SfM — live map')
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)  # let the window open before processing begins
    
    # Process First Two Frames (Initialization)
    print(f"Initializing with Frame 0 and Frame 1: {image_files[0]}, {image_files[1]}")
    
    # We use Task 1 code structure but need to store data properly
    kp1, des1, kp2, des2, matches, pts1, pts2, img1, img2 = extract_features_and_match(image_files[0], image_files[1])
    
    # Store frame data
    sfm_map.frame_data[0] = {'kp': kp1, 'des': des1}
    sfm_map.frame_data[1] = {'kp': kp2, 'des': des2}
    sfm_map.frame_indices.extend([0, 1])
    
    # Estimate Essential Matrix (Task 1)
    from task1 import estimate_essential_matrices
    E1, E2, mask = estimate_essential_matrices(pts1, pts2, K)
    inlier_mask = mask.ravel() == 1
    
    # Recover Pose (Task 2)
    from task2 import decompose_essential_matrix, disambiguate_pose, nonlinear_refinement
    
    # Filter inliers for pose estimation
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    matches_in = [matches[i] for i in range(len(matches)) if inlier_mask[i]]
    
    pairs = decompose_essential_matrix(E2)
    best_pair, X_init, P1, P2 = disambiguate_pose(pairs, K, pts1_in, pts2_in)
    R_init, t_init = best_pair
    
    # Set Cams
    sfm_map.add_camera(0, np.eye(3), np.zeros((3, 1)))  # Frame 0 is origin
    sfm_map.add_camera(1, R_init, t_init) # Frame 1 is relative
    
    # Refine Initial Structure (Task 2)
    print(f"Refining initial structure with {len(X_init)} points...")
    X_refined = nonlinear_refinement(X_init, P1, P2, pts1_in, pts2_in, max_nfev=20)
    
    # Add points to map
    count_added = 0
    for i, pt in enumerate(X_refined):
        # Only add valid points (e.g., positive depth check again or low reprojection error)
        # Using simple check: if it was an inlier in disambiguation, it's good enough for now
        # Get descriptors corresponding to this match
        m = matches_in[i]
        d1 = des1[m.queryIdx]
        d2 = des2[m.trainIdx]
        
        observations = [(0, m.queryIdx), (1, m.trainIdx)]
        sfm_map.add_point(pt, [d1, d2], observations)
        count_added += 1
        
    print(f"Initialization complete. Map has {count_added} points.")
    if live_plot:
        update_live_plot(sfm_map, ax, fig)
    mean_err = sfm_map.compute_global_reprojection_error()
    sfm_map.record_stats(1, mean_err)
    print(f"  System State: 2 cameras, {len(sfm_map.points_3d)} points")
    print(f"  Global Mean Reprojection Error: {mean_err:.4f} pixels")
    
    # Incremental Loop
    for i in range(2, len(image_files)):
        curr_idx = i
        prev_idx = i - 1
        img_path = image_files[i]
        print(f"\nProcessing Frame {curr_idx}: {img_path}")
        
        # 1. Feature Extraction
        import cv2
        img_curr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        kp_curr, des_curr = sift.detectAndCompute(img_curr, None)
        sfm_map.frame_data[curr_idx] = {'kp': kp_curr, 'des': des_curr}
        sfm_map.add_camera(curr_idx, None, None) # Placeholder
        
        # 2. Match to Map (PnP)
        # Strategy: match against the last WINDOW_SIZE successfully registered frames.
        # This prevents cascade failures where skipping one frame starves all subsequent ones.
        WINDOW_SIZE = 5
        registered = [idx for idx in sfm_map.frame_indices
                      if sfm_map.registered_cameras.get(idx, (None, None))[0] is not None]
        ref_indices = registered[-WINDOW_SIZE:] if len(registered) >= 1 else []

        # Collect correspondences from all reference frames; deduplicate by 3D-point index
        seen_pt3d = {}   # pt3d_idx -> (object_point, image_point, match)

        for ref_idx in ref_indices:
            kp_ref = sfm_map.frame_data[ref_idx]['kp']
            des_ref = sfm_map.frame_data[ref_idx]['des']
            matches_curr_ref = match_features_knn(des_curr, des_ref)

            for m in matches_curr_ref:
                idx_curr_kp = m.queryIdx
                idx_ref_kp  = m.trainIdx
                pt3d_idx = sfm_map.feature_to_point_map[ref_idx][idx_ref_kp]
                if pt3d_idx != -1 and pt3d_idx not in seen_pt3d:
                    seen_pt3d[pt3d_idx] = (
                        sfm_map.points_3d[pt3d_idx],
                        kp_curr[idx_curr_kp].pt,
                        m,
                        ref_idx
                    )

        object_points = []
        image_points  = []
        used_matches_for_pnp = []
        used_ref_indices = []   # which reference frame each match came from
        for pt3d_idx, (obj_pt, img_pt, m, r_idx) in seen_pt3d.items():
            object_points.append(obj_pt)
            image_points.append(img_pt)
            used_matches_for_pnp.append(m)
            used_ref_indices.append(r_idx)
                
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        
        if len(object_points) < 6:
            print(f"Warning: Not enough PnP matches ({len(object_points)}) for frame {curr_idx}. Skipping registration.")
            continue
            
        print(f"  PnP using {len(object_points)} correspondences.")
        
        # Estimate Pose
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, None,
            reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print(f"  PnP Failed for frame {curr_idx}.")
            continue
            
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec
        
        # Refine Pose (Non-linear)
        # Filter outliers from RANSAC
        if inliers is not None and len(inliers) > 0:
            inliers = inliers.flatten()
            obj_pts_in = object_points[inliers]
            img_pts_in = image_points[inliers]
            
            R_refined, t_refined = refine_pose(R_new, t_new, obj_pts_in, img_pts_in, K)
            
            # Update Map Camera
            sfm_map.add_camera(curr_idx, R_refined, t_refined)
            sfm_map.frame_indices.append(curr_idx)
            
            # Add new observations for existing points
            for list_idx, inlier_idx in enumerate(inliers):
                m = used_matches_for_pnp[inlier_idx]
                r_idx = used_ref_indices[inlier_idx]
                idx_curr_kp = m.queryIdx
                idx_ref_kp  = m.trainIdx
                pt3d_idx = sfm_map.feature_to_point_map[r_idx][idx_ref_kp]

                # Add observation
                sfm_map.point_observations[pt3d_idx].append((curr_idx, idx_curr_kp))
                sfm_map.point_descriptors[pt3d_idx].append(des_curr[idx_curr_kp])
                sfm_map.feature_to_point_map[curr_idx][idx_curr_kp] = pt3d_idx
        else:
            print("  PnP RANSAC returned 0 inliers.")
            continue
            
        # 3. Triangulate New Points against most-recently registered frame
        new_pts_count = 0
        triang_ref_idx = registered[-1] if registered else None
        if triang_ref_idx is not None and sfm_map.registered_cameras.get(triang_ref_idx, (None, None))[0] is not None:
            P_curr_t = sfm_map.get_camera_projection_matrix(curr_idx)
            P_ref_t  = sfm_map.get_camera_projection_matrix(triang_ref_idx)
            kp_ref_t  = sfm_map.frame_data[triang_ref_idx]['kp']
            des_ref_t = sfm_map.frame_data[triang_ref_idx]['des']
            matches_t = match_features_knn(des_curr, des_ref_t)
            kp1c, kp2c, des1c, des2c, midx = [], [], [], [], []
            for m in matches_t:
                iq, ir = m.queryIdx, m.trainIdx
                if sfm_map.feature_to_point_map[triang_ref_idx][ir] == -1 and \
                   sfm_map.feature_to_point_map[curr_idx][iq] == -1:
                    kp1c.append(kp_ref_t[ir].pt); kp2c.append(kp_curr[iq].pt)
                    des1c.append(des_ref_t[ir]);   des2c.append(des_curr[iq])
                    midx.append((ir, iq))
            if len(kp1c) > 0:
                from task2 import compute_reprojection_error
                pts4d = cv2.triangulatePoints(P_ref_t, P_curr_t,
                                               np.array(kp1c).T, np.array(kp2c).T)
                pts3d = (pts4d[:3] / pts4d[3]).T
                e1a = compute_reprojection_error(pts3d, P_ref_t,  np.array(kp1c))
                e2a = compute_reprojection_error(pts3d, P_curr_t, np.array(kp2c))
                R_r, t_r = sfm_map.registered_cameras[triang_ref_idx]
                R_c, t_c = sfm_map.registered_cameras[curr_idx]
                dr = ((R_r @ pts3d.T).T + t_r.flatten())[:, 2]
                dc = ((R_c @ pts3d.T).T + t_c.flatten())[:, 2]
                thr = 4.0
                for j in range(len(pts3d)):
                    if dr[j]>0 and dc[j]>0 and np.linalg.norm(e1a[2*j:2*j+2])<thr \
                       and np.linalg.norm(e2a[2*j:2*j+2])<thr:
                        ir_p, iq_p = midx[j]
                        sfm_map.add_point(pts3d[j], [des1c[j], des2c[j]],
                                          [(triang_ref_idx, ir_p), (curr_idx, iq_p)])
                        new_pts_count += 1
        print(f"  Added {new_pts_count} new 3D points.")
        
        valid_cams = sum(1 for R, t in sfm_map.registered_cameras.values() if R is not None)
        mean_err = sfm_map.compute_global_reprojection_error()
        sfm_map.record_stats(curr_idx, mean_err)
        print(f"  System State: {valid_cams} cameras, {len(sfm_map.points_3d)} points")
        print(f"  Global Mean Reprojection Error: {mean_err:.4f} pixels")
        
        if live_plot and (curr_idx % max(1, plot_every) == 0):
            update_live_plot(sfm_map, ax, fig)

    if live_plot:
        plt.ioff()
    return sfm_map

def visualize_map(sfm_map):
    pts = np.array(sfm_map.points_3d)
    registered = [idx for idx in sfm_map.frame_indices
                  if sfm_map.registered_cameras.get(idx, (None, None))[0] is not None]
    cams = []
    for idx in registered:
        R, t = sfm_map.registered_cameras[idx]
        C = -R.T @ t
        cams.append(C.flatten())
    cams = np.array(cams)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(pts) > 0:
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=2, c='steelblue', alpha=0.55, label='Structure')
        
    if len(cams) > 0:
        cam_color = np.linspace(0, 1, len(cams))
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c='indigo', alpha=0.45, linewidth=2)
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2],
                   c=cam_color, cmap='plasma', s=28, marker='o', label='Trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    ax.set_title(f"Trajectory - {len(cams)} cameras, {len(pts)} pts")

    final_bounds = _compute_dense_view_bounds(pts)
    if final_bounds is not None:
        _set_equal_3d_limits(ax, final_bounds[0], final_bounds[1])

    plt.tight_layout()
    plt.savefig('task3_map.png', dpi=140)
    plt.savefig('task3_trajectory.png', dpi=140)
    print("Saved trajectory visualization to task3_map.png and task3_trajectory.png")

def visualize_metrics(sfm_map, err_threshold=2.0):
    frames = np.array(sfm_map.stats_frames, dtype=int)
    cams = np.array(sfm_map.stats_registered_cameras, dtype=float)
    pts = np.array(sfm_map.stats_points, dtype=float)
    errs = np.array(sfm_map.stats_mean_reproj, dtype=float)

    if len(frames) == 0:
        frames = np.array([0], dtype=int)
        cams = np.array([0.0], dtype=float)
        pts = np.array([0.0], dtype=float)
        errs = np.array([0.0], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(frames, cams, color='#1f77b4', linewidth=3)
    axes[0].set_title('Registered Cameras')
    axes[0].set_xlabel('Frame')

    axes[1].plot(frames, pts, color='#f2a100', linewidth=3)
    axes[1].set_title('Map Points')
    axes[1].set_xlabel('Frame')

    axes[2].plot(frames, errs, color='red', linewidth=3)
    axes[2].axhline(err_threshold, color='gray', linestyle='--', linewidth=1.6, label='threshold')
    axes[2].set_title('Mean Reprojection Error (px)')
    axes[2].set_xlabel('Frame')
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('task3_metrics.png', dpi=140)
    print("Saved metrics dashboard to task3_metrics.png")

def main():
    parser = argparse.ArgumentParser(description="Task 3: Incremental Mapping")
    parser.add_argument("image_dir", help="Directory containing sequence of images")
    parser.add_argument("--frame-interval", type=int, default=25, help="Use every Nth frame from image_dir")
    parser.add_argument("--max-width", type=int, default=640, help="Resize selected frames to this max width (<=0 disables resize)")
    parser.add_argument("--optimized-dir", default="task3_optimized_frames", help="Where resized selected frames are written")
    parser.add_argument("--no-live-plot", action="store_true", help="Disable real-time incremental map visualization")
    parser.add_argument("--plot-every", type=int, default=1, help="Update live map every N processed frames")
    args = parser.parse_args()

    # Load images
    raw_files, input_step, _, _ = load_frames(args.image_dir, frame_interval=1)
    print(f"Found {len(raw_files)} candidate images in {args.image_dir}")
    print(f"Detected input frame step: {input_step}")

    image_files, _, effective_interval, stride_in_files = load_frames(
        args.image_dir,
        frame_interval=max(1, args.frame_interval)
    )
    print(
        f"Using {len(image_files)} images | target interval {max(1, args.frame_interval)} | "
        f"effective interval {effective_interval} | file stride {stride_in_files}"
    )

    image_files = write_resized_frame_subset(image_files, args.optimized_dir, args.max_width)
    if args.max_width > 0:
        print(f"Wrote resized subset to {args.optimized_dir} (max width {args.max_width})")

    if len(image_files) < 2:
        print("Need at least 2 images.")
        return

    # Get intrinsics (assuming same size)
    img0 = cv2.imread(image_files[0])
    h, w = img0.shape[:2]
    K = get_camera_intrinsics((h, w))
    print(f"Intrinsics:\n{K}")

    sfm_map = run_incremental_sfm(
        image_files,
        K,
        live_plot=not args.no_live_plot,
        plot_every=max(1, args.plot_every)
    )
    
    visualize_map(sfm_map)
    visualize_metrics(sfm_map, err_threshold=2.0)
    plt.show()
    
    # Save map? The prompt doesn't explicitly ask to save to file, but we should make it reusable for Task 4
    # We'll just run it as a script for now. 
    # For Task 4 integration, Task 4 will likely import from here or we pickle the object.

if __name__ == "__main__":
    main()
