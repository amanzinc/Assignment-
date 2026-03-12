import cv2
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt

# --- FLANN parameters for SIFT (float32 descriptors, KD-Tree)
_FLANN_INDEX_KDTREE = 1
_FLANN_INDEX_PARAMS = dict(algorithm=_FLANN_INDEX_KDTREE, trees=5)
_FLANN_SEARCH_PARAMS = dict(checks=50)   # 50 checks: good speed/accuracy trade-off

# Maximum SIFT features per image. Limiting this is the single biggest speed-up.
MAX_SIFT_FEATURES = 2000
# Resize images so the longer dimension is at most this value before SIFT.
# 4K (3840x2160) → 640x360 reduces pixel count by ~36x: SIFT < 50 ms/frame → >10 FPS.
MAX_IMG_DIM = 640


def resize_for_detection(img):
    """
    Downscales an image so its longer side is at most MAX_IMG_DIM pixels.
    Returns (resized_img, scale_factor).
    """
    h, w = img.shape[:2]
    scale = min(MAX_IMG_DIM / max(h, w), 1.0)   # never upscale
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def extract_features_and_match(img1_path, img2_path):
    """
    Reads two images, downscales for fast SIFT detection, detects SIFT features
    (capped at MAX_SIFT_FEATURES), matches with FLANN, and scales keypoint
    coordinates back to the original image resolution.
    Returns keypoints, descriptors, good matches, point arrays, and ORIGINAL images.
    """
    img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1_orig is None or img2_orig is None:
        raise ValueError("Could not read one or both of the images.")

    img1_small, scale1 = resize_for_detection(img1_orig)
    img2_small, scale2 = resize_for_detection(img2_orig)

    # --- Feature Detection: SIFT with an upper bound on keypoints for speed
    sift = cv2.SIFT_create(nfeatures=MAX_SIFT_FEATURES)
    kp1_small, des1 = sift.detectAndCompute(img1_small, None)
    kp2_small, des2 = sift.detectAndCompute(img2_small, None)

    # Scale keypoints back to ORIGINAL image coordinates
    def scale_kps(kps, scale):
        if scale == 1.0:
            return kps
        scaled = []
        for kp in kps:
            kp2 = cv2.KeyPoint(
                x=kp.pt[0] / scale, y=kp.pt[1] / scale,
                size=kp.size / scale, angle=kp.angle,
                response=kp.response, octave=kp.octave,
                class_id=kp.class_id)
            scaled.append(kp2)
        return scaled

    kp1 = scale_kps(kp1_small, scale1)
    kp2 = scale_kps(kp2_small, scale2)

    # --- Matching: FLANN is 5-10x faster than BFMatcher for float descriptors
    flann = cv2.FlannBasedMatcher(_FLANN_INDEX_PARAMS, _FLANN_SEARCH_PARAMS)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    return kp1, des1, kp2, des2, good_matches, pts1, pts2, img1_orig, img2_orig

def get_camera_intrinsics(image_shape):
    """
    Initializes the camera intrinsics matrix Ks per the assignment requirements.
    fx = fy = 0.7 * W
    cx = W / 2
    cy = H / 2
    """
    h, w = image_shape
    fx = 0.7 * w
    fy = 0.7 * w
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def enforce_rank2(E):
    """
    Enforces the rank-2 constraint on an Essential matrix using SVD.
    """
    U, S, Vt = np.linalg.svd(E)
    S_new = np.diag([S[0], S[1], 0]) # Set smallest singular value to 0
    E_rank2 = U @ S_new @ Vt
    return E_rank2

def estimate_essential_matrices(pts1, pts2, K):
    """
    Estimates E1 (using all correspondences, non-robust) and E2 (using RANSAC, robust).
    """
    # 1. Non-robust estimation (E1) using all points
    # We first find the Fundamental matrix F using the 8-point algorithm without RANSAC
    # cv2.FM_8POINT requires at least 8 matches and uses all provided points
    F1, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    
    # E = K.T * F * K
    if F1 is not None and F1.shape == (3, 3):
        E1_unconstrained = K.T @ F1 @ K
        E1 = enforce_rank2(E1_unconstrained)
    else:
        print("Warning: Could not compute valid F1. Falling back to identity or invalid matrix.")
        E1 = np.eye(3)

    # 2. Robust estimation (E2) using RANSAC
    # cv2.findEssentialMat directly calculates the Essential Matrix using robust methods
    E2, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Enforce rank-2 on E2 just to be safe, though OpenCV usually handles it for the primary solution
    # Note: cv2.findEssentialMat can return multiple 3x3 matrices stacked vertically (e.g. 9x3) if there are multiple solutions.
    # We just take the first one (top 3x3 block).
    if E2 is not None and E2.shape[0] >= 3:
        E2 = E2[:3, :]
        E2 = enforce_rank2(E2)
    else:
        print("Warning: Could not compute valid E2.")
        E2 = np.eye(3)

    return E1, E2, mask

def compute_sampson_distance(pts1, pts2, F):
    """
    Computes the mean Sampson distance (first-order epipolar error) for all point pairs.
    A lower value means the fundamental matrix better satisfies the epipolar constraint.
    """
    pts1_h = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2_h = np.hstack((pts2, np.ones((len(pts2), 1))))
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    num = (pts2_h * Fx1).sum(axis=1) ** 2
    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(np.mean(num / denom))


def draw_epipolar_lines(img1, img2, pts1, pts2, F1, F2, inlier_mask):
    """
    Draws epipolar lines in img2 for 10 sampled INLIER points from img1.
    Computes Sampson distance on inlier points for fair comparison.
    """
    # Use inlier points only for quality comparison
    pts1_in = pts1[inlier_mask.ravel() == 1]
    pts2_in = pts2[inlier_mask.ravel() == 1]

    np.random.seed(42)
    indices = np.random.choice(len(pts1_in), min(10, len(pts1_in)), replace=False)
    rand_pts1 = pts1_in[indices]
    rand_pts2 = pts2_in[indices]

    lines1 = cv2.computeCorrespondEpilines(rand_pts1.reshape(-1, 1, 2), 1, F1).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(rand_pts1.reshape(-1, 1, 2), 1, F2).reshape(-1, 3)

    h2, w2 = img2.shape
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    colors = [tuple(c.tolist()) for c in np.random.randint(30, 230, (10, 3))]

    def draw_lines_on_image(img, lines, pts_img2):
        out = img.copy()
        for r, pt, color in zip(lines, pts_img2, colors):
            if abs(r[1]) > 1e-6:
                x0, y0 = 0, int(-r[2] / r[1])
                x1, y1 = w2, int(-(r[2] + r[0] * w2) / r[1])
            else:
                x0, y0 = int(-r[2] / r[0]), 0
                x1, y1 = int(-(r[2] + r[1] * h2) / r[0]), h2
            out = cv2.line(out, (x0, y0), (x1, y1), color, 3)
            out = cv2.circle(out, tuple(map(int, pt)), 10, color, -1)
        return out

    img_lines1 = draw_lines_on_image(img2_color, lines1, rand_pts2)
    img_lines2 = draw_lines_on_image(img2_color, lines2, rand_pts2)

    # Compute Sampson distance on INLIER set (fair comparison)
    sd1 = compute_sampson_distance(pts1_in, pts2_in, F1)
    sd2 = compute_sampson_distance(pts1_in, pts2_in, F2)
    ratio = sd1 / sd2 if sd2 > 1e-12 else float('inf')
    winner = "E2 (RANSAC)" if sd2 < sd1 else "E1 (8-point)"
    print(f"\nEpipolar Constraint Quality (mean Sampson distance on {len(pts1_in)} inliers):")
    print(f"  E1 (8-point, all points) : {sd1:.6f}")
    print(f"  E2 (RANSAC)              : {sd2:.6f}")
    if sd2 < sd1:
        print(f"  -> E2 is {ratio:.1f}x better at satisfying the epipolar constraint.")
    else:
        print(f"  -> E1 has lower Sampson distance on inliers (ratio {ratio:.2f}).")
        print(f"     Note: RANSAC robustness is its key advantage — it rejects outliers.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(img_lines1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Epipolar Lines (E1: All points)\nSampson dist = {sd1:.4f}")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_lines2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Epipolar Lines (E2: RANSAC)\nSampson dist = {sd2:.4f}")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('epipolar_lines_comparison.png', dpi=150)
    print("Saved epipolar_lines_comparison.png")

    # Live window for demo
    disp = np.hstack([img_lines1, img_lines2])
    disp_small = cv2.resize(disp, (min(1800, disp.shape[1]), min(500, disp.shape[0])))
    cv2.imshow('Epipolar Lines: E1 (left) vs E2-RANSAC (right)', disp_small)
    cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Task 1: Feature Correspondence and Essential Matrix Estimation")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    args = parser.parse_args()

    print(f"Loading images: {args.image1} and {args.image2}")
    print(f"SIFT feature limit: {MAX_SIFT_FEATURES}  |  Matcher: FLANN")

    # ── 1. Feature Detection & Matching ─────────────────────────────────────
    t0 = time.perf_counter()
    kp1, des1, kp2, des2, good_matches, pts1, pts2, img1, img2 = \
        extract_features_and_match(args.image1, args.image2)
    t_match = time.perf_counter() - t0

    print(f"Detected {len(kp1)} / {len(kp2)} keypoints.")
    print(f"Found {len(good_matches)} good matches.")
    print(f"  Feature detection + matching time: {t_match*1000:.1f} ms  "
          f"(~{1/t_match:.1f} FPS if run per frame)")

    if len(good_matches) < 8:
        print("Error: Not enough good matches (need ≥ 8).")
        return

    # Live match visualisation for demo
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_small = cv2.resize(match_vis, (min(1800, match_vis.shape[1]),
                                         min(400, match_vis.shape[0])))
    cv2.imshow('Task 1 – Feature Matches (top-50)', match_small)
    cv2.waitKey(1)

    # ── 2. Essential Matrix Estimation ──────────────────────────────────────
    K = get_camera_intrinsics(img1.shape)
    print(f"\nCamera Intrinsics K:\n{K}")

    t0 = time.perf_counter()
    E1, E2, mask = estimate_essential_matrices(pts1, pts2, K)
    t_E = time.perf_counter() - t0
    n_inliers = int(mask.sum())

    print(f"\nEssential Matrix E1 (8-point / all correspondences):\n{E1}")
    print(f"\nEssential Matrix E2 (RANSAC, {n_inliers} inliers / {len(pts1)} total):\n{E2}")
    print(f"  Essential matrix estimation time: {t_E*1000:.1f} ms")

    # Fundamental matrices for epipolar line drawing
    K_inv = np.linalg.inv(K)
    F1 = K_inv.T @ E1 @ K_inv
    F2 = K_inv.T @ E2 @ K_inv

    # ── 3. Epipolar Lines Visualisation + Quantitative Comparison ───────────
    print("\nGenerating epipolar lines visualisation...")
    draw_epipolar_lines(img1, img2, pts1, pts2, F1, F2, mask)

    print("\nPress any key in the OpenCV windows to close them.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
