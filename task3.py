import cv2
import numpy as np
import argparse
import os
import glob
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

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

def load_frames(dataset_path):
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
    return files

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
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, c='b', label='Structure')
        
    if len(cams) > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=20, c='r', marker='^', label='Cameras')
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    valid_cams = sum(1 for R, t in sfm_map.registered_cameras.values() if R is not None)
    ax.set_title(f'Incremental SfM Map\nPoints: {len(pts)} | Cameras: {valid_cams}')
    plt.pause(0.01)

def run_incremental_sfm(image_files, K):
    # Initialize Map
    sfm_map = SfMMap(K)
    
    # Initialize live plotting
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
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
    update_live_plot(sfm_map, ax, fig)
    mean_err = sfm_map.compute_global_reprojection_error()
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
        # Strategy: Match against Frame (i-1).
        # Identify which keypoints in Frame (i-1) match to a 3D point.
        # Use those matches to build 2D-3D correspondences for Frame i.
        
        # Match Frame i -> Frame i-1
        kp_prev = sfm_map.frame_data[prev_idx]['kp']
        des_prev = sfm_map.frame_data[prev_idx]['des']
        
        matches_curr_prev = match_features_knn(des_curr, des_prev)
        
        # Build 3D-2D correspondences
        object_points = []
        image_points = []
        used_matches_for_pnp = []
        
        for m in matches_curr_prev:
            # queryIdx is curr, trainIdx is prev
            idx_curr = m.queryIdx
            idx_prev = m.trainIdx
            
            # Check if prev feature has a 3D point
            pt3d_idx = sfm_map.feature_to_point_map[prev_idx][idx_prev]
            
            if pt3d_idx != -1: # It has a 3D point!
                pt3d = sfm_map.points_3d[pt3d_idx]
                object_points.append(pt3d)
                image_points.append(kp_curr[idx_curr].pt)
                used_matches_for_pnp.append(m)
                
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
            for idx in inliers:
                m = used_matches_for_pnp[idx]
                idx_curr = m.queryIdx
                idx_prev = m.trainIdx
                pt3d_idx = sfm_map.feature_to_point_map[prev_idx][idx_prev]
                
                # Add observation
                sfm_map.point_observations[pt3d_idx].append((curr_idx, idx_curr))
                sfm_map.point_descriptors[pt3d_idx].append(des_curr[idx_curr])
                sfm_map.feature_to_point_map[curr_idx][idx_curr] = pt3d_idx
        else:
            print("  PnP RANSAC returned 0 inliers.")
            continue
            
        # 3. Triangulate New Points (Map Expansion)
        # Use matches_curr_prev again. Find matches where prev has NO 3D point.
        # But wait, we only triangulate if we have a valid baseline and good intersection.
        
        P_curr = sfm_map.get_camera_projection_matrix(curr_idx)
        P_prev = sfm_map.get_camera_projection_matrix(prev_idx)
        
        new_pts_count = 0
        
        # Collect candidates
        kp1_cand = [] # prev
        kp2_cand = [] # curr
        des1_cand = []
        des2_cand = []
        match_indices = [] # (idx_prev, idx_curr)
        
        for m in matches_curr_prev:
            idx_curr = m.queryIdx
            idx_prev = m.trainIdx
            
            # Conditions:
            # 1. point must not already exist (or maybe we allow merging? stick to simple: not exist)
            # 2. we need sufficient parallax (skipped for now, reliance on reproj error)
            
            pt3d_idx_prev = sfm_map.feature_to_point_map[prev_idx][idx_prev]
            pt3d_idx_curr = sfm_map.feature_to_point_map[curr_idx][idx_curr] # Should be -1 usually
            
            if pt3d_idx_prev == -1 and pt3d_idx_curr == -1:
                kp1_cand.append(kp_prev[idx_prev].pt)
                kp2_cand.append(kp_curr[idx_curr].pt)
                des1_cand.append(des_prev[idx_prev])
                des2_cand.append(des_curr[idx_curr])
                match_indices.append((idx_prev, idx_curr))
        
        if len(kp1_cand) > 0:
            kp1_cand = np.array(kp1_cand)
            kp2_cand = np.array(kp2_cand)
            
            # Triangulate
            pts4d = cv2.triangulatePoints(P_prev, P_curr, kp1_cand.T, kp2_cand.T)
            pts3d_cand = (pts4d[:3, :] / pts4d[3, :]).T
            
            # Filter bad points
            # 1. Positive depth in both
            # 2. Reprojection error
            
            # Project to both
            from task2 import compute_reprojection_error
            err1 = compute_reprojection_error(pts3d_cand, P_prev, kp1_cand)
            err2 = compute_reprojection_error(pts3d_cand, P_curr, kp2_cand)
            
            # Check depths
            # Transform to camera coordinates
            pts3d_cand_hom = np.hstack((pts3d_cand, np.ones((len(pts3d_cand), 1))))
            
            # P = K[R|t] -> X_cam = [R|t]X_world
            R_prev, t_prev = sfm_map.registered_cameras[prev_idx]
            R_curr, t_curr = sfm_map.registered_cameras[curr_idx]
            
            X_cam_prev = (R_prev @ pts3d_cand.T).T + t_prev.flatten()
            X_cam_curr = (R_curr @ pts3d_cand.T).T + t_curr.flatten()
            
            depth_prev = X_cam_prev[:, 2]
            depth_curr = X_cam_curr[:, 2]
            
            # Thresholds
            max_repr_err = 4.0
            
            for j in range(len(pts3d_cand)):
                e1 = np.linalg.norm(err1[2*j:2*j+2])
                e2 = np.linalg.norm(err2[2*j:2*j+2])
                d1 = depth_prev[j]
                d2 = depth_curr[j]
                
                if d1 > 0 and d2 > 0 and e1 < max_repr_err and e2 < max_repr_err:
                    # Good point
                    idx_prev, idx_curr = match_indices[j]
                    
                    obs = [(prev_idx, idx_prev), (curr_idx, idx_curr)]
                    des = [des1_cand[j], des2_cand[j]]
                    
                    sfm_map.add_point(pts3d_cand[j], des, obs)
                    new_pts_count += 1
        
        print(f"  Added {new_pts_count} new 3D points.")
        
        valid_cams = sum(1 for R, t in sfm_map.registered_cameras.values() if R is not None)
        mean_err = sfm_map.compute_global_reprojection_error()
        print(f"  System State: {valid_cams} cameras, {len(sfm_map.points_3d)} points")
        print(f"  Global Mean Reprojection Error: {mean_err:.4f} pixels")
        
        update_live_plot(sfm_map, ax, fig)

    plt.ioff()
    return sfm_map

def visualize_map(sfm_map):
    pts = np.array(sfm_map.points_3d)
    cams = []
    for idx in sfm_map.frame_indices:
        R, t = sfm_map.registered_cameras[idx]
        # Camera center C = -R^T * t
        C = -R.T @ t
        cams.append(C.flatten())
    cams = np.array(cams)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(pts) > 0:
        # Downsample for visualization if too many
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, c='b', label='Structure')
        
    if len(cams) > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=20, c='r', marker='^', label='Cameras')
        # Plot trajectory line
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig('task3_map.png')
    print("Saved map visualization to task3_map.png")
    plt.show() # Added to keep final plot open

def main():
    parser = argparse.ArgumentParser(description="Task 3: Incremental Mapping")
    parser.add_argument("image_dir", help="Directory containing sequence of images")
    args = parser.parse_args()

    # Load images
    image_files = load_frames(args.image_dir)
    print(f"Loaded {len(image_files)} images from {args.image_dir}")
    if len(image_files) < 2:
        print("Need at least 2 images.")
        return

    # Get intrinsics (assuming same size)
    img0 = cv2.imread(image_files[0])
    h, w = img0.shape[:2]
    K = get_camera_intrinsics((h, w))
    print(f"Intrinsics:\n{K}")

    sfm_map = run_incremental_sfm(image_files, K)
    
    visualize_map(sfm_map)
    
    # Save map? The prompt doesn't explicitly ask to save to file, but we should make it reusable for Task 4
    # We'll just run it as a script for now. 
    # For Task 4 integration, Task 4 will likely import from here or we pickle the object.

if __name__ == "__main__":
    main()
