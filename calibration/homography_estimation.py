"""
Homography Estimation -- Software-First Validation
Dataset : FLIR ADAS aligned (Zhang et al. ICIP2020)

Validation strategy
-------------------
This script validates the SIFT + RANSAC pipeline's ability to recover a known
homography from noisy feature correspondences.

Same-modal test (thermal -> warped thermal):
  1. Load a thermal image from the aligned dataset.
  2. Apply a known synthetic H_gt (rotation + translation + mild perspective).
  3. Match SIFT features between the original and warped thermal using
     BFMatcher + Lowe ratio test.
  4. Estimate H_est with cv2.findHomography(..., RANSAC).
  5. Evaluate: 4-corner reprojection error (H_gt vs H_est).

Design note -- why same-modal, not cross-modal:
  SIFT cross-modal matching (RGB <-> Thermal) yields only ~10-30 good matches
  because the two modalities carry fundamentally different texture content
  (visual reflectance vs. temperature distribution).  This is expected and
  is not a bug.  Cross-modal correspondence extraction in the physical setup
  is handled via the structured calibration target (PTC hotspot Otsu-threshold
  + RGB checkerboard blob detection), not SIFT.  The purpose of this script
  is to validate the RANSAC pipeline mechanics in isolation -- same-modal
  gives 1200+ matches, which is the clean test for that.

Acceptance criteria:
  Mean corner reprojection error < 2 px
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time

# -- Config --------------------------------------------------------------------
ALIGNED_ROOT  = Path("C:/wildfire_uav/data/flir_adas/aligned")
ANNOTATED_DIR = ALIGNED_ROOT / "AnnotatedImages"
RESULTS_DIR   = Path("C:/wildfire_uav/calibration/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TXT     = ALIGNED_ROOT / "align_train.txt"
N_PAIRS       = 50          # image pairs to evaluate
RANSAC_THRESH = 3.0         # pixels
RANSAC_CONF   = 0.995
LOWE_RATIO    = 0.75
SIFT_NFEATURES = 2000

# Synthetic misalignment: ~5deg rotation, 8px x-translation, -5px y-translation,
# plus tiny perspective coefficients simulating non-parallel camera axes.
# Parameters reflect realistic camera-mount offset for the 30mm baseline rig.
SYNTHETIC_H = np.array([
    [0.9962, -0.0872,  8.0],
    [0.0872,  0.9962, -5.0],
    [0.0001,  0.0001,  1.0],
], dtype=np.float64)

# -- Helpers -------------------------------------------------------------------

def load_thermal_gray(stem: str) -> np.ndarray | None:
    """Load thermal PreviewData as single-channel uint8.
    The FLIR ADAS PreviewData.jpeg files are true grayscale (R=G=B),
    not pseudocolor, so we take any one channel directly.
    """
    th_path = ANNOTATED_DIR / f"{stem}.jpeg"
    th_bgr  = cv2.imread(str(th_path), cv2.IMREAD_COLOR)
    if th_bgr is None:
        return None
    return th_bgr[:, :, 0]   # R=G=B, any channel is identical


def apply_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def corner_reprojection_error(H_gt: np.ndarray, H_est: np.ndarray,
                               img_shape) -> float:
    """
    Warp the 4 image corners with H_gt and H_est; return mean L2 distance.
    Measures how accurately H_est reproduces H_gt's mapping -- independent
    of which specific feature points were used for estimation.
    """
    h, w = img_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    def warp_pts(H, pts):
        h_pts = np.concatenate([pts, np.ones((4, 1))], axis=1).T   # 3x4
        w_pts = H @ h_pts
        w_pts /= w_pts[2]
        return w_pts[:2].T   # 4x2

    c_gt  = warp_pts(H_gt,  corners)
    c_est = warp_pts(H_est, corners)
    return float(np.mean(np.linalg.norm(c_gt - c_est, axis=1)))


def estimate_homography_sift(img1: np.ndarray, img2: np.ndarray):
    """
    CLAHE -> SIFT keypoints -> BFMatcher + Lowe ratio -> RANSAC findHomography.
    Returns (H_est, n_inliers, n_matches_after_lowe) or (None, 0, n) on failure.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img1_eq = clahe.apply(img1)
    img2_eq = clahe.apply(img2)

    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    kp1, des1 = sift.detectAndCompute(img1_eq, None)
    kp2, des2 = sift.detectAndCompute(img2_eq, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]
    if len(good) < 8:
        return None, 0, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_est, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        confidence=RANSAC_CONF,
    )
    if H_est is None:
        return None, 0, len(good)

    n_inliers = int(mask.sum()) if mask is not None else 0
    return H_est, n_inliers, len(good)


# -- Main ----------------------------------------------------------------------

def main():
    stems = [l.strip() for l in TRAIN_TXT.read_text().splitlines() if l.strip()]
    stems = stems[:N_PAIRS]
    print(f"Homography Estimation -- same-modal validation")
    print(f"Pairs: {len(stems)}  |  RANSAC thresh: {RANSAC_THRESH}px  |  Lowe: {LOWE_RATIO}")
    print(f"\nSynthetic H_gt:\n{SYNTHETIC_H}\n")

    results  = []
    errors   = []
    n_failed = 0
    t0 = time.time()

    for i, stem in enumerate(stems):
        th = load_thermal_gray(stem)
        if th is None:
            print(f"  [{i+1:3d}] SKIP  {stem}  (load error)")
            n_failed += 1
            continue

        h, w = th.shape

        # Apply synthetic misalignment to get "query" thermal
        th_warped = apply_homography(th, SYNTHETIC_H)

        # Estimate H from (original thermal) -> (warped thermal)
        H_est, n_inliers, n_matches = estimate_homography_sift(th, th_warped)

        if H_est is None:
            print(f"  [{i+1:3d}] FAIL  {stem:<42s}  matches={n_matches:4d}  RANSAC failed")
            n_failed += 1
            continue

        err = corner_reprojection_error(SYNTHETIC_H, H_est, (h, w))
        errors.append(err)
        results.append({
            "stem": stem,
            "n_matches": n_matches,
            "n_inliers": n_inliers,
            "reproj_error_px": round(err, 4),
            "pass": err < 2.0,
        })
        status = "PASS" if err < 2.0 else "WARN"
        print(f"  [{i+1:3d}] {status}  {stem:<42s}  "
              f"matches={n_matches:4d}  inliers={n_inliers:4d}  err={err:.3f}px")

    elapsed = time.time() - t0

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)

    if errors:
        mean_err   = float(np.mean(errors))
        median_err = float(np.median(errors))
        max_err    = float(np.max(errors))
        pass_rate  = sum(1 for e in errors if e < 2.0) / len(errors) * 100
        verdict    = "PASS" if mean_err < 2.0 else "FAIL"

        print(f"  Evaluated   : {len(results)} / {len(stems)}  ({n_failed} failed)")
        print(f"  Mean err    : {mean_err:.4f} px  (target < 2.0 px)")
        print(f"  Median err  : {median_err:.4f} px")
        print(f"  Max err     : {max_err:.4f} px")
        print(f"  Pass rate   : {pass_rate:.1f}%  (<2px)")
        print(f"  Elapsed     : {elapsed:.1f}s  ({elapsed/len(stems)*1000:.0f}ms/pair)")
        print(f"\n  Acceptance criterion (mean < 2px): {verdict}")
    else:
        mean_err = median_err = max_err = pass_rate = None
        print("  No results collected.")

    # -- Save ------------------------------------------------------------------
    report_path = RESULTS_DIR / "step2_homography_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "synthetic_H": SYNTHETIC_H.tolist(),
            "config": {
                "n_pairs": N_PAIRS,
                "ransac_thresh": RANSAC_THRESH,
                "ransac_confidence": RANSAC_CONF,
                "lowe_ratio": LOWE_RATIO,
                "sift_nfeatures": SIFT_NFEATURES,
            },
            "summary": {
                "n_evaluated": len(results),
                "n_failed": n_failed,
                "mean_reproj_error_px": round(mean_err, 4) if mean_err is not None else None,
                "median_reproj_error_px": round(median_err, 4) if median_err is not None else None,
                "max_reproj_error_px": round(max_err, 4) if max_err is not None else None,
                "pass_rate_pct": round(pass_rate, 1) if pass_rate is not None else None,
                "verdict": "PASS" if (mean_err is not None and mean_err < 2.0) else "FAIL",
            },
            "per_pair": results,
        }, f, indent=2)

    np.savez(RESULTS_DIR / "H_synthetic_gt.npz", H=SYNTHETIC_H)
    print(f"\n  Report -> {report_path}")
    print(f"  H_gt   -> {RESULTS_DIR / 'H_synthetic_gt.npz'}")


if __name__ == "__main__":
    main()
