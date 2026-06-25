# Decision 001 — Why ORB-based homography fails for RGB-IR alignment

**Date:** 2026-06-25  
**Phase:** 1 — Initial pipeline  
**Status:** Resolved → use Identity H for pre-aligned datasets

---

## Context

The Week 1 pipeline includes a `warpPerspective` step to align the infrared image into the visible camera's coordinate space before blending. My initial implementation estimated the homography matrix H using ORB feature matching + RANSAC, which is the standard approach for same-modality image registration.

---

## What I observed

Running the pipeline on LLVIP produced Warped IR frames that looked like **ray-like streaks converging to a single point**, with approximately 99.6% of the output filled with black pixels.

Quantifying this for sample `010002`:

```
Total pixels      : 1,310,720
Valid IR pixels   : 5,193  (0.40%)
Black fill        : 1,305,527  (99.60%)
```

Even the 0.40% of visible IR content was spatially incorrect — the wrong region of the IR image mapped to the wrong location in the output.

---

## Root cause analysis

**Why does ORB fail here?**

ORB's BRIEF descriptor works by comparing local pixel intensity relationships within a patch. This works well when both images are captured by the same type of sensor, because the same physical point produces similar intensity patterns in both images.

RGB and infrared images are formed by completely different physical processes:

| | Visible (RGB) | Infrared (thermal) |
|--|--|--|
| Source | Reflected ambient/artificial light | Self-emitted thermal radiation |
| Bright = | Reflective surfaces (white walls, street lamps) | Hot objects (people, engines) |
| Dark = | Absorptive surfaces | Cool surfaces (road, sky) |

The same physical patch produces **uncorrelated or inverted** intensity patterns across the two modalities. ORB descriptors computed from one modality do not match descriptors from the other — the "matches" RANSAC receives are pure noise.

**Why does a noisy H produce the streak pattern?**

A homography H with non-zero perspective parameters (bottom row ≠ [0, 0, 1]) introduces projective division. When H is estimated from noise matches, these parameters are arbitrary. `warpPerspective` uses inverse mapping: for each output pixel, it back-projects through H⁻¹ to find the source IR coordinate. With a degenerate H, almost all output pixels back-project to coordinates outside the IR image bounds → filled with `borderValue=0` (black). Only a thin strip of output pixels back-project to valid IR coordinates, forming the ray-like streaks.

The convergence point of the streaks is the **vanishing point** of the projective transformation — the image of the point at infinity under H.

---

## Decision

For the LLVIP dataset, which is **pre-aligned** (visible and IR cameras are hardware-synchronized and pre-rectified), the correct homography is the identity matrix:

```python
H = np.eye(3, dtype=np.float64)
```

No spatial correction is needed. Applying ORB estimation actively harms the result.

**Homography quality gate added to `week1_pipeline.py`:**
- If RANSAC inliers < 20 → fall back to Identity H
- If condition number of H > 1e⁶ → fall back to Identity H

**For Phase 2** (unregistered or partially registered data), I will replace ORB with:
- **ECC (Enhanced Correlation Coefficient)** — optimises alignment directly on intensity without descriptors, robust to cross-modal differences
- **Phase correlation** — frequency-domain approach, descriptor-free

---

## What this taught me

The failure mode here is not a bug in my code — it reveals a fundamental assumption mismatch: ORB was designed for single-modality matching. Applying it cross-modally produces plausible-looking output (a warp happens, code runs without error) but the result is entirely wrong. This kind of silent failure is harder to catch than a crash, and catching it required inspecting the output quantitatively (pixel coverage analysis) rather than just visually.
