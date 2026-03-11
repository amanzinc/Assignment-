import cv2
import numpy as np
import argparse
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Import necessary functions from Task 1
from task1 import extract_features_and_match, get_camera_intrinsics, estimate_essential_matrices

def decompose_essential_matrix(E):
    """
    Decomposes the Essential Matrix into 4 possible (R, t) pairs.
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # The 4 possible pairs are:
    # 1. R1, t
    # 2. R1, -t
    # 3. R2, t
    # 4. R2, -t
    
    pairs = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    return pairs

def linear_triangulation(P1, P2, pts1, pts2):
    """
    Implements linear triangulation to recover 3D points.
    P1, P2: 3x4 projection matrices (K[R|t])
    pts1, pts2: Nx2 arrays of normalized or image coordinates
    """
    # cv2.triangulatePoints expects 2xN arrays
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to 3D Cartesian coordinates
    pts3d = pts4d[:3, :] / pts4d[3, :]
    return pts3d.T  # Return as Nx3

def compute_reprojection_error(X, P, pts_2d):
    """
    Computes total reprojection error for a single view.
    """
    X_hom = np.hstack((X, np.ones((X.shape[0], 1))))
    x_proj_hom = (P @ X_hom.T).T
    x_proj = x_proj_hom[:, :2] / x_proj_hom[:, 2, np.newaxis]
    error = (x_proj - pts_2d).flatten()
    return error

def reprojection_residuals(X_flat, P1, P2, pts1, pts2):
    """
    Calculates the residuals for the least squares optimization over both views.
    """
    N = int(len(X_flat) / 3)
    X = X_flat.reshape((N, 3))
    
    err1 = compute_reprojection_error(X, P1, pts1)
    err2 = compute_reprojection_error(X, P2, pts2)
    
    return np.concatenate([err1, err2])

def nonlinear_refinement(X_init, P1, P2, pts1, pts2, max_nfev=10):
    """
    Refines 3D points by minimizing reprojection error over all observing views.
    """
    X_flat_init = X_init.flatten()
    
    # Create the block diagonal sparsity structure for the jacobian
    # There are 2 views * 2 coordinates = 4 residuals per 3D point
    from scipy.sparse import lil_matrix
    N = len(X_init)
    A = lil_matrix((4 * N, 3 * N), dtype=int)
    for i in range(N):
        A[i*2:(i*2)+2, i*3:(i*3)+3] = 1         # 2 residuals from View 1 depend on 3 params of point i
        A[(2*N)+i*2:(2*N)+(i*2)+2, i*3:(i*3)+3] = 1 # 2 residuals from View 2 depend on 3 params of point i
        
    res = least_squares(
        reprojection_residuals, 
        X_flat_init, 
        args=(P1, P2, pts1, pts2),
        method='trf',      # Trust Region Reflective (faster for large scale)
        jac_sparsity=A,    # CRUCIAL: Makes it run in < 1 second instead of hours
        ftol=1e-15,         
        xtol=1e-15,
        max_nfev=max_nfev 
    )
    
    X_refined = res.x.reshape((len(X_init), 3))
    return X_refined

def disambiguate_pose(pairs, K, pts1, pts2):
    """
    Selects the correct (R, t) pair using the Cheirality condition (Z > 0 in both cameras).
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    best_pair_idx = -1
    max_positive_depths = -1
    best_X = None
    best_P2 = None
    
    print("Evaluating the 4 candidate poses:")
    for i, (R, t) in enumerate(pairs):
        P2 = K @ np.hstack((R, t))
        
        # Triangulate
        X = linear_triangulation(P1, P2, pts1, pts2)
        
        # Calculate depth in Camera 1
        depth1 = X[:, 2]
        
        # Calculate depth in Camera 2
        X_c2 = (R @ X.T) + t
        depth2 = X_c2[2, :]
        
        # Condition: Z > 0 in both cameras
        valid_points = np.sum((depth1 > 0) & (depth2 > 0))
        
        print(f"  Pose {i+1}: Valid points (Z > 0) = {valid_points} / {len(X)}")
        
        if valid_points > max_positive_depths:
            max_positive_depths = valid_points
            best_pair_idx = i
            best_X = X
            best_P2 = P2
            
    return pairs[best_pair_idx], best_X, P1, best_P2

def plot_3d_points(X_init, X_refined):
    """
    Plots the 3D point cloud before and after refinement.
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Initial
    ax1 = fig.add_subplot(121, projection='3d')
    if X_init is not None and len(X_init) > 0:
        ax1.scatter(X_init[:, 0], X_init[:, 1], X_init[:, 2], c='r', marker='o', s=15, alpha=0.7) # Increased size and added alpha
    ax1.set_title('3D Points (Initial Linear Triangulation)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Refined
    ax2 = fig.add_subplot(122, projection='3d')
    if X_refined is not None and len(X_refined) > 0:
        ax2.scatter(X_refined[:, 0], X_refined[:, 1], X_refined[:, 2], c='g', marker='^', s=15, alpha=0.7) # Increased size and added alpha
    ax2.set_title('3D Points (Non-linear Refinement)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('task2_3d_reconstruction.png')
    print("Saved 3D plot to task2_3d_reconstruction.png")
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Task 2: Triangulation and Pose Recovery")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    args = parser.parse_args()

    print("Running Task 1 Feature Detection...")
    kp1, des1, kp2, des2, good_matches, pts1, pts2, img1, img2 = extract_features_and_match(args.image1, args.image2)
    K = get_camera_intrinsics(img1.shape)
    
    # We use E2 (RANSAC) and its mask to use only inliers for pose recovery and triangulation
    E1, E2, mask = estimate_essential_matrices(pts1, pts2, K)
    
    # Filter points to only include inliers from Essential matrix estimation
    inlier_mask = mask.ravel() == 1
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    
    print(f"Using {len(pts1_inliers)} inlier correspondences for Task 2.")

    # 2. Pose Decomposition
    pairs = decompose_essential_matrix(E2)
    
    # 3. Pose Disambiguation (Cheirality) and Linear Triangulation
    best_pair, X_init, P1, P2 = disambiguate_pose(pairs, K, pts1_inliers, pts2_inliers)
    best_R, best_t = best_pair
    
    print(f"\nSelected Candidate Pose:")
    print(f"R = \n{best_R}")
    print(f"t = \n{best_t}")

    # 4. Calculate initial reprojection error
    err1_init = compute_reprojection_error(X_init, P1, pts1_inliers)
    err2_init = compute_reprojection_error(X_init, P2, pts2_inliers)
    mean_err_init = np.mean([np.linalg.norm(err1_init.reshape(-1, 2), axis=1), 
                             np.linalg.norm(err2_init.reshape(-1, 2), axis=1)])
    
    print(f"\nInitial Average Reprojection Error: {mean_err_init:.4f} pixels")

    # 5. Nonlinear Refinement
    print("Running non-linear refinement (10 iterations)...")
    X_refined = nonlinear_refinement(X_init, P1, P2, pts1_inliers, pts2_inliers, max_nfev=10)
    
    # Calculate refined reprojection error
    err1_ref = compute_reprojection_error(X_refined, P1, pts1_inliers)
    err2_ref = compute_reprojection_error(X_refined, P2, pts2_inliers)
    mean_err_ref = np.mean([np.linalg.norm(err1_ref.reshape(-1, 2), axis=1), 
                            np.linalg.norm(err2_ref.reshape(-1, 2), axis=1)])
                            
    print(f"Refined Average Reprojection Error: {mean_err_ref:.4f} pixels")
    
    # 6. Plot the 3D results
    plot_3d_points(X_init, X_refined)

if __name__ == "__main__":
    main()
