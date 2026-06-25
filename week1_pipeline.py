"""
Week 1 Pipeline: load -> warpPerspective -> overlay
Dataset: LLVIP (Low-Light Visible-Infrared Pairs)
Root:    C:/wildfire_uav/LLVIP/LLVIP/
"""

import sys
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
LLVIP_ROOT  = Path("C:/wildfire_uav/LLVIP/LLVIP")
OUTPUT_DIR  = Path("C:/wildfire_uav/output/week1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_ALPHA = 0.5     # weight for visible channel in blend
N_SAMPLES     = 5       # number of pairs to process


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 · LOAD
# ══════════════════════════════════════════════════════════════════════════════

def discover_dataset(root: Path):
    """
    Find visible / infrared sub-dirs regardless of exact naming.
    LLVIP typically uses: visible/ and infrared/ (train & test inside each).
    """
    candidates = {
        "visible":  ["visible", "rgb", "VIS", "visible_light"],
        "infrared": ["infrared", "ir", "thermal", "IR", "lwir"],
    }
    found = {}
    for key, names in candidates.items():
        for name in names:
            p = root / name
            if p.exists():
                found[key] = p
                break
    return found


def collect_pairs(dirs: dict, split="train"):
    """Return sorted list of (vis_path, ir_path) tuples for a given split."""
    vis_dir = dirs["visible"] / split
    ir_dir  = dirs["infrared"] / split

    if not vis_dir.exists():
        vis_dir = dirs["visible"]
    if not ir_dir.exists():
        ir_dir  = dirs["infrared"]

    vis_files = sorted(vis_dir.glob("*.jpg")) + sorted(vis_dir.glob("*.png"))
    ir_files  = sorted(ir_dir.glob("*.jpg"))  + sorted(ir_dir.glob("*.png"))

    vis_map = {f.stem: f for f in vis_files}
    ir_map  = {f.stem: f for f in ir_files}
    common  = sorted(set(vis_map) & set(ir_map))

    return [(vis_map[s], ir_map[s]) for s in common]


def load_pair(vis_path: Path, ir_path: Path):
    """Load visible (BGR) and infrared (gray→BGR) image."""
    vis = cv2.imread(str(vis_path))
    ir  = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

    if vis is None:
        raise FileNotFoundError(f"Cannot read visible: {vis_path}")
    if ir is None:
        raise FileNotFoundError(f"Cannot read infrared: {ir_path}")

    ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    return vis, ir_bgr, ir


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 · WARP PERSPECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def compute_homography(src_gray: np.ndarray, dst_gray: np.ndarray):
    """
    Estimate homography (IR → visible) via ORB feature matching + RANSAC.
    Falls back to identity if too few matches.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(dst_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  [WARN] Too few keypoints — using identity homography")
        return np.eye(3, dtype=np.float64), 0

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    good    = matches[:min(200, len(matches))]

    if len(good) < 4:
        print(f"  [WARN] Only {len(good)} matches — using identity homography")
        return np.eye(3, dtype=np.float64), len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    if H is None:
        print("  [WARN] Homography failed — using identity")
        return np.eye(3, dtype=np.float64), 0

    # reject degenerate H: too few inliers or bad condition number
    if inliers < 20:
        print(f"  [WARN] Only {inliers} inliers — H unreliable, using identity")
        return np.eye(3, dtype=np.float64), inliers

    cond = np.linalg.cond(H)
    if cond > 1e6:
        print(f"  [WARN] H condition number {cond:.1e} — degenerate, using identity")
        return np.eye(3, dtype=np.float64), inliers

    return H, inliers


def warp_perspective(ir_bgr: np.ndarray, H: np.ndarray, target_shape):
    """Warp IR image into visible coordinate space."""
    h, w = target_shape[:2]
    warped = cv2.warpPerspective(ir_bgr, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    return warped


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 · OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def colormap_ir(ir_gray: np.ndarray):
    """Apply COLORMAP_JET to infrared for false-colour overlay."""
    norm = cv2.normalize(ir_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def overlay_images(vis: np.ndarray, ir_warped: np.ndarray,
                   ir_gray: np.ndarray, alpha: float = 0.5):
    """
    Blend visible + false-colour IR.
    Returns (blended_bgr, colourised_ir_bgr).
    """
    # resize IR to match vis if shapes differ
    if ir_warped.shape[:2] != vis.shape[:2]:
        ir_warped = cv2.resize(ir_warped, (vis.shape[1], vis.shape[0]))

    ir_color = colormap_ir(cv2.cvtColor(ir_warped, cv2.COLOR_BGR2GRAY))
    blended  = cv2.addWeighted(vis, alpha, ir_color, 1 - alpha, 0)
    return blended, ir_color


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helper
# ══════════════════════════════════════════════════════════════════════════════

def save_result(vis, ir_bgr, warped, blended, stem, inliers):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (vis,     "Visible (RGB)"),
        (ir_bgr,  "Infrared (gray→BGR)"),
        (warped,  f"Warped IR\n({inliers} inliers)"),
        (blended, f"Overlay (α={OVERLAY_ALPHA})"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{stem}_pipeline.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Week 1 Pipeline: load → warpPerspective → overlay")
    print("=" * 60)

    # ── locate sub-dirs ──
    dirs = discover_dataset(LLVIP_ROOT)
    if "visible" not in dirs or "infrared" not in dirs:
        print(f"\n[ERROR] Cannot find visible/infrared dirs under {LLVIP_ROOT}")
        print("  Found:", list(LLVIP_ROOT.iterdir()) if LLVIP_ROOT.exists() else "root missing")
        sys.exit(1)

    print(f"\n  visible  : {dirs['visible']}")
    print(f"  infrared : {dirs['infrared']}")

    # ── collect pairs ──
    pairs = collect_pairs(dirs, split="train")
    if not pairs:
        pairs = collect_pairs(dirs, split="")       # flat layout fallback
    if not pairs:
        print("[ERROR] No matching image pairs found.")
        sys.exit(1)

    print(f"\n  Total pairs found : {len(pairs)}")
    sample = pairs[:N_SAMPLES]
    print(f"  Processing        : {N_SAMPLES} samples")

    results = []

    for i, (vis_path, ir_path) in enumerate(sample):
        stem = vis_path.stem
        print(f"\n[{i+1}/{N_SAMPLES}] {stem}")

        # STEP 1 · load
        vis, ir_bgr, ir_gray = load_pair(vis_path, ir_path)
        print(f"  Load     : vis={vis.shape}  ir={ir_bgr.shape}")

        # STEP 2 · warpPerspective
        vis_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        H, inliers = compute_homography(ir_gray, vis_gray)
        warped = warp_perspective(ir_bgr, H, vis.shape)
        print(f"  Warp     : inliers={inliers}  H=\n{np.round(H,4)}")

        # STEP 3 · overlay
        blended, _ = overlay_images(vis, warped, ir_gray, OVERLAY_ALPHA)
        print(f"  Overlay  : blended={blended.shape}")

        # save
        out_path = save_result(vis, ir_bgr, warped, blended, stem, inliers)
        print(f"  Saved    : {out_path}")
        results.append((stem, inliers, str(out_path)))

    # ── summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  {'Stem':<20} {'Inliers':>8}  Output")
    for stem, inliers, path in results:
        print(f"  {stem:<20} {inliers:>8}  {Path(path).name}")
    print(f"\n  Output dir : {OUTPUT_DIR}")
    print("  Pipeline completed successfully.")


if __name__ == "__main__":
    run()
