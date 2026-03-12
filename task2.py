import cv2
import numpy as np
import argparse
import time
from scipy.optimize import least_squares
from scipy.sparse import kron, eye as speye
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
    Uses a vectorized sparse Jacobian pattern (no Python loops) for speed.
    """
    X_flat_init = X_init.flatten()
    N = len(X_init)

    # Block-diagonal sparsity: each 3D point has 4 residuals (2 per view)
    # and 3 parameters. kron builds this in one vectorised call.
    A = kron(speye(N, format='csr', dtype=np.int8), np.ones((4, 3), dtype=np.int8))

    res = least_squares(
        reprojection_residuals,
        X_flat_init,
        args=(P1, P2, pts1, pts2),
        method='trf',
        jac_sparsity=A,
        ftol=1e-15,
        xtol=1e-15,
        max_nfev=max_nfev
    )

    X_refined = res.x.reshape((N, 3))
    return X_refined

def disambiguate_pose(pairs, K, pts1, pts2):
    """
    Selects the correct (R, t) pair using the Cheirality condition (Z > 0 in both cameras).
    Prints all 4 candidate poses and their cheirality counts.
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_pair_idx = -1
    max_positive_depths = -1
    best_X = None
    best_P2 = None

    print("\n--- 4 Candidate (R, t) Pairs from Essential Matrix Decomposition ---")
    for i, (R, t) in enumerate(pairs):
        P2 = K @ np.hstack((R, t))
        X = linear_triangulation(P1, P2, pts1, pts2)

        depth1 = X[:, 2]
        X_c2 = (R @ X.T) + t
        depth2 = X_c2[2, :]
        valid_points = int(np.sum((depth1 > 0) & (depth2 > 0)))

        print(f"\n  Pose {i+1}:")
        print(f"    R =\n{np.array2string(R, prefix='      ', precision=6)}")
        print(f"    t = {t.flatten()}")
        print(f"    Points with Z>0 in both cameras: {valid_points} / {len(X)}")

        if valid_points > max_positive_depths:
            max_positive_depths = valid_points
            best_pair_idx = i
            best_X = X
            best_P2 = P2

    print(f"\n  -> Selected Pose {best_pair_idx + 1} "
          f"(max positive-depth points: {max_positive_depths} / {len(pts1)})")
    return pairs[best_pair_idx], best_X, P1, best_P2

def plot_3d_points(X_init, X_refined, output_path='task2_3d_reconstruction.png'):
    """
    Plots 3D point cloud before and after non-linear refinement.
    """
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Sparse 3D Point Cloud \u2014 Task 2', fontsize=14, fontweight='bold')

    for col, (X, color, subtitle) in enumerate([
        (X_init,    '#4C8BE2', 'Before Refinement'),
        (X_refined, '#E87D2B', 'After Refinement'),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.set_title(subtitle, fontsize=11)

        if X is not None and len(X) > 0:
            step = max(1, len(X) // 3000)
            n_shown = len(X[::step])
            ax.scatter(X[::step, 0], X[::step, 1], X[::step, 2],
                       c=color, marker='o', s=2, alpha=0.7,
                       label=f'{n_shown} pts')

        # Camera 1 at origin — green dot
        ax.scatter([0], [0], [0], c='green', s=60, zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left', markerscale=4, fontsize=9, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)

    # Show the saved image in an OpenCV window for live demo
    buf = cv2.imread(output_path)
    if buf is not None:
        cv2.imshow('Task 2 \u2013 3D Reconstruction (before / after refinement)', buf)
        cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Task 2: Triangulation and Pose Recovery")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--output", default="task2_3d_reconstruction.png",
                        help="Output PNG filename (default: task2_3d_reconstruction.png)")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 2: Triangulation and Pose Recovery")
    print("=" * 60)

    # ─ Feature detection & matching (Task 1 reuse) ─
    t0 = time.perf_counter()
    kp1, des1, kp2, des2, good_matches, pts1, pts2, img1, img2 = \
        extract_features_and_match(args.image1, args.image2)
    K = get_camera_intrinsics(img1.shape)
    E1, E2, mask = estimate_essential_matrices(pts1, pts2, K)
    t_feat = time.perf_counter() - t0
    print(f"\nFeature detection + matching + E estimation: {t_feat*1000:.1f} ms  "
          f"(~{1/t_feat:.1f} FPS)")

    inlier_mask = mask.ravel() == 1
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    print(f"Using {len(pts1_inliers)} inlier correspondences.")

    # ─ 1. Pose Decomposition ─
    t0 = time.perf_counter()
    pairs = decompose_essential_matrix(E2)

    # ─ 2 & 4. Linear Triangulation + Cheirality Disambiguation ─
    best_pair, X_init, P1, P2 = disambiguate_pose(pairs, K, pts1_inliers, pts2_inliers)
    best_R, best_t = best_pair
    t_tri = time.perf_counter() - t0

    print(f"\n  Triangulation + disambiguation time: {t_tri*1000:.1f} ms")
    print(f"\nFinal Selected Pose:")
    print(f"  R =\n{best_R}")
    print(f"  t = {best_t.flatten()}")

    # ─ 3. Reprojection error BEFORE refinement ─
    err1_init = compute_reprojection_error(X_init, P1, pts1_inliers)
    err2_init = compute_reprojection_error(X_init, P2, pts2_inliers)
    errs_init = np.concatenate([
        np.linalg.norm(err1_init.reshape(-1, 2), axis=1),
        np.linalg.norm(err2_init.reshape(-1, 2), axis=1)
    ])
    print(f"\nReprojection Error (before refinement):")
    print(f"  Mean : {errs_init.mean():.4f} px")
    print(f"  Std  : {errs_init.std():.4f} px")
    print(f"  Max  : {errs_init.max():.4f} px")

    # ─ 3. Nonlinear Refinement (fixed 10 iterations, no early stopping) ─
    print("\nRunning non-linear refinement (10 iterations, no early stopping)...")
    t0 = time.perf_counter()
    X_refined = nonlinear_refinement(X_init, P1, P2, pts1_inliers, pts2_inliers, max_nfev=10)
    t_ref = time.perf_counter() - t0

    err1_ref = compute_reprojection_error(X_refined, P1, pts1_inliers)
    err2_ref = compute_reprojection_error(X_refined, P2, pts2_inliers)
    errs_ref = np.concatenate([
        np.linalg.norm(err1_ref.reshape(-1, 2), axis=1),
        np.linalg.norm(err2_ref.reshape(-1, 2), axis=1)
    ])
    print(f"Reprojection Error (after refinement):")
    print(f"  Mean : {errs_ref.mean():.4f} px")
    print(f"  Std  : {errs_ref.std():.4f} px")
    print(f"  Max  : {errs_ref.max():.4f} px")
    print(f"  Refinement time: {t_ref*1000:.1f} ms")

    # ─ Visualisation ─
    plot_3d_points(X_init, X_refined, args.output)

    print("\nPress any key in the OpenCV windows to close them.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
