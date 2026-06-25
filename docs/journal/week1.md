# Week 1 Journal — Pipeline Foundations

**Period:** 2026-06-24 to 2026-06-25  
**Goal:** Get the full load → warpPerspective → overlay pipeline running end-to-end on real data

---

## What I set out to do

Build the skeleton of the RGB-Thermal fusion pipeline using the LLVIP dataset. The three stages were clear from the start: load a pair of images, geometrically align them, and blend them into a single fused output. The interesting question was how to handle the alignment step when the two images come from different sensors.

---

## What I actually did

### Environment setup
Created a dedicated conda environment (`wildfire_uav`, Python 3.11) with OpenCV, NumPy, Ultralytics, and SciPy. Chose Python 3.11 over 3.12/3.14 because Ultralytics 8.0.196 has the most tested compatibility at 3.11, and I want stability over novelty for a research baseline.

### Dataset validation (verify_dataset.py)
Before writing any model code, I wrote a dataset integrity checker for the FLIR ADAS aligned subset. This caught:
- All 5,142 RGB-thermal pairs present and correctly named
- All Pascal VOC XML annotations valid
- No train/val overlap
- Uniform image size (512 × 640 × 3)

I did this first because discovering a corrupted dataset mid-training wastes days.

### Pipeline (week1_pipeline.py)
Implemented the three-stage pipeline:
1. `load_pair()` — reads visible (BGR) and IR (grayscale → BGR)
2. `compute_homography()` — ORB + BFMatcher + RANSAC to estimate H
3. `warp_perspective()` + `overlay_images()` — JET colourmap on IR, then `addWeighted` blend

### The streak problem
The first real output looked completely wrong — the Warped IR frame was almost entirely black with a few bright streaks radiating from a point. I initially assumed a bug in the warp call or the H matrix indexing.

Investigating the H matrices output by RANSAC:
```
[[-6.79e-02  -5.92e-01   3.81e+02]
 [ 1.08e-02  -6.83e-01   3.94e+02]
 [-1.00e-04  -1.70e-03   1.00e+00]]
```
The bottom row being non-zero means this is a perspective (not affine) transform. When the perspective parameters are wrong, the denominator in the projective division `(h₂₀x + h₂₁y + 1)` approaches zero for large areas of the output → those pixels back-project to infinity → black fill.

I then quantified the damage:
```python
Valid IR pixels : 5,193 / 1,310,720 = 0.40%
```
And realised even that 0.40% was spatially incorrect — it came from arbitrary regions of the IR image mapped to arbitrary output locations. The overlay was 100% useless: 99.6% of it was just the visible image dimmed by 50%.

### Root cause
Cross-modal matching. ORB descriptors encode local intensity relationships. Visible and IR images form by completely different physics — the same patch in both images has uncorrelated (sometimes inverted) intensities. ORB "matches" are noise. RANSAC can't recover a valid H from pure noise.

The correct answer for LLVIP (which is pre-aligned) is H = Identity.

---

## Decisions made

1. **Added homography quality gate** — if RANSAC inliers < 20 or condition number > 1e⁶, fall back to Identity H. This prevents the streak artefact without hard-coding the dataset-specific knowledge into the warp function.

2. **Phase 2 will use ECC or phase correlation** — descriptor-free methods that optimise alignment on raw intensities, robust to cross-modal differences.

3. **Quantitative output checks are non-negotiable** — the streak failure was silent (no exception, plausible-looking output). I'll add pixel coverage assertions to the pipeline going forward.

---

## What surprised me

I expected the hard part of Week 1 to be the overlay blending. The actual hard part was understanding *why* a standard feature-matching pipeline produces a geometrically coherent-looking (ray pattern with a clear vanishing point) but completely wrong result. The output wasn't random noise — it had clear structure — which made it deceptive. The vanishing point of the degenerate H is visible in the output as the convergence point of all the streaks.

---

## Next week

- Apply the Identity H fix and confirm clean overlay output
- Investigate ECC-based alignment on a pair with actual perspective offset
- Start reading on fusion architectures (DenseFuse, FusionGAN) as Phase 3 prep
