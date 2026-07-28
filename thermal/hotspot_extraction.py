"""
ThermalHotspotExtractor
Implements Stage 1-2 of the thermal analysis algorithm (CLAUDE.md Sec. 7.2).

Stage 1 -- Adaptive thresholding
    T_threshold = max(mean_temp + 3 * std_temp, 60.0)  # hard floor 60 degC
    hotspot_mask = temp_celsius > T_threshold

Stage 2 -- Hotspot feature extraction
    Connected components on hotspot_mask.
    Per-hotspot features: T_max, area_px, circularity.
    Circularity = 4*pi*area / perimeter^2 in [0, 1].
    Near-1 = compact/circular (anthropogenic heat source).
    Near-0 = elongated (natural fire front).

Output per frame: list of Hotspot dataclasses, one per detected region.
This module is stateless across frames -- it says nothing about whether a
hotspot is growing. Stage 3 (temporal growth rate, dA/dt) is a separate,
harder module built on top of this one: see growth_rate_tracker.py.

Test suite (no FLAME3 required):
    A. Synthetic sequence -- known fire growing from radius 10 -> 50 px over
       30 frames at 8 FPS. Validates Stage 1-2 recall as the fire grows.
    B. FLIR ADAS proxy -- grayscale thermal rescaled to simulated degC range.
       Validates Stage 1-2 on real (non-synthetic) thermal imagery.
       Note: driving scenes have no actual wildfires; Stage 1's 60 degC floor
       suppresses most detections, which is the correct behavior.

Acceptance criteria (CLAUDE.md Sec. 7):
    - Hotspot recall (synthetic, T > 80 degC regions): > 90%
    - Stage 1-2 latency: < 200 ms/frame
"""

import cv2
import numpy as np
import time
import json
from dataclasses import dataclass
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
FLIR_ANNOTATED = Path("C:/wildfire_uav/data/flir_adas/aligned/AnnotatedImages")
RESULTS_DIR    = Path("C:/wildfire_uav/calibration/results")   # shared results dir
FUSION_RESULTS = Path("C:/wildfire_uav/results")
FUSION_RESULTS.mkdir(parents=True, exist_ok=True)

# -- Config --------------------------------------------------------------------
SIGMA_FACTOR   = 3.0     # adaptive threshold: mean + SIGMA_FACTOR * std
HARD_FLOOR_C   = 60.0    # degC -- minimum threshold (3-sigma reasoning below)
MIN_AREA_PX    = 9       # ignore regions smaller than this (noise filter)
FPS            = 8.0     # inference rate (Jetson Nano target)
DT             = 1.0 / FPS


# -- Data structures -----------------------------------------------------------

@dataclass
class Hotspot:
    """One detected thermal hotspot in a single frame (Stage 1-2 output only)."""
    bbox:         tuple        # (x, y, w, h) in thermal pixel coords
    T_max:        float        # max temperature in the region (degC)
    area_px:      int          # pixel area
    circularity:  float        # 4*pi*area/perimeter^2; 1=circle, 0=line
    centroid:     tuple        # (cx, cy)


# -- ThermalHotspotExtractor ---------------------------------------------------

class ThermalHotspotExtractor:
    """
    Frame-by-frame thermal hotspot detector. Stateless across calls -- each
    process() call only looks at the current frame (Stage 1-2). Cross-frame
    tracking / growth rate is added by HotspotGrowthTracker in
    growth_rate_tracker.py.

    Usage:
        extractor = ThermalHotspotExtractor()
        for temp_frame in video:          # temp_frame: 2D float32 in degC
            hotspots = extractor.process(temp_frame)
    """

    def __init__(self,
                 sigma_factor: float = SIGMA_FACTOR,
                 hard_floor:   float = HARD_FLOOR_C,
                 min_area:     int   = MIN_AREA_PX):
        self.sigma_factor = sigma_factor
        self.hard_floor   = hard_floor
        self.min_area     = min_area

    def _adaptive_threshold(self, temp_c: np.ndarray) -> tuple[np.ndarray, float]:
        """Stage 1: compute mask of pixels above adaptive threshold."""
        mean_ = float(temp_c.mean())
        std_  = float(temp_c.std())
        thresh = max(mean_ + self.sigma_factor * std_, self.hard_floor)
        mask = (temp_c > thresh).astype(np.uint8)
        return mask, thresh

    @staticmethod
    def _circularity(area_px: int, contour) -> float:
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 1e-6:
            return 0.0
        return float(4 * np.pi * area_px / (perimeter ** 2))

    def process(self, temp_c: np.ndarray) -> list[Hotspot]:
        """
        Process one thermal frame.
        temp_c : 2D float32 array, values in degrees Celsius.
        Returns list of Hotspot (may be empty).
        """
        if temp_c.ndim != 2:
            raise ValueError("temp_c must be a 2D array (H x W)")

        # Stage 1 -- adaptive thresholding
        mask, _thresh = self._adaptive_threshold(temp_c)

        # Morphological cleanup: remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Stage 2 -- connected components + feature extraction
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        hotspots = []

        for lbl in range(1, n_labels):   # skip background (label 0)
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < self.min_area:
                continue

            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h_box = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])

            # T_max in this region
            region_temp = temp_c[labels == lbl]
            T_max = float(region_temp.max())

            # Contour for circularity
            region_mask = (labels == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            circ = self._circularity(area_px, contours[0]) if contours else 0.0

            hotspots.append(Hotspot(
                bbox        = (x, y, w, h_box),
                T_max       = round(T_max, 2),
                area_px     = area_px,
                circularity = round(circ, 4),
                centroid    = (round(cx, 1), round(cy, 1)),
            ))

        return hotspots


# -- Synthetic test sequence ---------------------------------------------------

def make_synthetic_frame(height: int, width: int,
                          fire_cx: float, fire_cy: float,
                          fire_radius: float,
                          bg_mean: float = 25.0,
                          bg_std:  float = 3.0,
                          fire_T:  float = 95.0,
                          noise:   float = 2.0,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a synthetic thermal frame with one circular hotspot.
    Background: Gaussian noise around bg_mean +- bg_std degC.
    Fire region: disk of radius fire_radius at (fire_cx, fire_cy), T ~ fire_T.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    frame = rng.normal(bg_mean, bg_std, (height, width)).astype(np.float32)

    # Draw fire disk
    yy, xx = np.ogrid[:height, :width]
    dist2  = (xx - fire_cx) ** 2 + (yy - fire_cy) ** 2
    mask   = dist2 <= fire_radius ** 2
    # Radial temperature gradient: hottest at center
    dist   = np.sqrt(dist2)
    T_dist = fire_T * (1 - dist / (fire_radius + 1e-6))
    T_dist = np.clip(T_dist, bg_mean, fire_T)
    frame[mask] = T_dist[mask] + rng.normal(0, noise, mask.sum()).astype(np.float32)
    return frame


def run_synthetic_test():
    """
    Test A -- Expanding circular fire (30 frames, R: 10->50 px, fire_T=95 degC).
    Verifies Stage 1-2 recall and circularity as the fire grows.
    No temporal claim is made here -- growth direction (dA/dt) is Stage 3's
    job, tested separately in growth_rate_tracker.py.
    """
    print("=" * 60)
    print("TEST A -- Synthetic expanding fire sequence (Stage 1-2 only)")
    print("=" * 60)

    H, W       = 120, 160
    N_FRAMES   = 30
    R0, R1     = 10.0, 50.0
    BG_MEAN    = 25.0
    FIRE_T     = 95.0
    FX, FY     = W / 2, H / 2

    extractor = ThermalHotspotExtractor()
    rng       = np.random.default_rng(42)
    detected  = 0
    latencies = []

    print(f"  Frames: {N_FRAMES}  FPS: {FPS}  R: {R0:.0f}->{R1:.0f}px  Fire T: {FIRE_T}degC\n")

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        r = R0 + t * (R1 - R0)
        frame = make_synthetic_frame(H, W, FX, FY, r, BG_MEAN,
                                     fire_T=FIRE_T, rng=rng)
        t0 = time.perf_counter()
        hotspots = extractor.process(frame)
        latencies.append((time.perf_counter() - t0) * 1000)

        if hotspots:
            detected += 1
            hs = max(hotspots, key=lambda h: h.area_px)
            if i % 5 == 0 or i == N_FRAMES - 1:
                print(f"  Frame {i+1:2d}: area={hs.area_px:5d}px^2  "
                      f"T_max={hs.T_max:.1f}C  circ={hs.circularity:.3f}  "
                      f"lat={latencies[-1]:.1f}ms")
        else:
            if i % 5 == 0:
                print(f"  Frame {i+1:2d}: no hotspot detected")

    recall       = detected / N_FRAMES * 100
    mean_latency = float(np.mean(latencies))

    print(f"\n  Recall         : {recall:.1f}%   (target >90%)")
    print(f"  Mean latency   : {mean_latency:.2f} ms  (target <200ms)")

    r_verdict = "PASS" if recall       > 90.0 else "FAIL"
    l_verdict = "PASS" if mean_latency < 200.0 else "FAIL"
    print(f"\n  Recall [{r_verdict}]  Latency [{l_verdict}]")

    return {
        "recall_pct":      round(recall,       1),
        "mean_latency_ms": round(mean_latency, 2),
        "recall_verdict":  r_verdict,
        "latency_verdict": l_verdict,
    }


# -- FLIR ADAS proxy test ------------------------------------------------------

def flir_gray_to_celsius(gray: np.ndarray,
                          T_min: float = 15.0,
                          T_max: float = 80.0) -> np.ndarray:
    """
    Linear map grayscale [0, 255] -> temperature [T_min, T_max] degC.
    This is a rough proxy; real FLIR radiometric data uses per-pixel calibration.
    Brighter = hotter (as in typical thermal camera display).
    """
    return (gray.astype(np.float32) / 255.0) * (T_max - T_min) + T_min


def run_flir_proxy_test(n_images: int = 30):
    """
    Test B: FLIR ADAS grayscale thermal converted to simulated degC.
    Validates Stage 1-2 on real (non-synthetic) texture.
    Note: 60 degC floor maps to grayscale ~217/255 in [15, 80] range.
    Most driving-scene pixels are < 60 degC, so recall is expected to be low
    (correct behavior -- no wildfires in driving data).
    Reports: detection rate, mean T_max of detected hotspots, mean latency.
    """
    print("\n" + "=" * 60)
    print("TEST B -- FLIR ADAS grayscale proxy (simulated degC)")
    print("=" * 60)

    stems_file = Path("C:/wildfire_uav/data/flir_adas/aligned/align_train.txt")
    if not stems_file.exists():
        print("  SKIP: align_train.txt not found")
        return None

    stems = [l.strip() for l in stems_file.read_text().splitlines() if l.strip()]
    stems = stems[:n_images]

    extractor  = ThermalHotspotExtractor()
    latencies  = []
    detections = 0
    T_maxes    = []

    for stem in stems:
        th_path = FLIR_ANNOTATED / f"{stem}.jpeg"
        th_bgr  = cv2.imread(str(th_path), cv2.IMREAD_COLOR)
        if th_bgr is None:
            continue
        gray    = th_bgr[:, :, 0]
        temp_c  = flir_gray_to_celsius(gray)

        t0 = time.perf_counter()
        hotspots = extractor.process(temp_c)
        latencies.append((time.perf_counter() - t0) * 1000)

        if hotspots:
            detections += 1
            T_maxes.append(max(h.T_max for h in hotspots))

    if not latencies:
        print("  SKIP: no images loaded")
        return None

    det_rate   = detections / len(latencies) * 100
    mean_lat   = float(np.mean(latencies))
    mean_T_max = float(np.mean(T_maxes)) if T_maxes else 0.0

    print(f"  Images processed : {len(latencies)}")
    print(f"  Detection rate   : {det_rate:.1f}%  "
          f"(low expected -- 60 degC floor, no wildfires in driving data)")
    print(f"  Mean T_max (det) : {mean_T_max:.1f} degC")
    print(f"  Mean latency     : {mean_lat:.2f} ms  (target <200ms)")

    l_verdict = "PASS" if mean_lat < 200.0 else "FAIL"
    print(f"\n  Latency [{l_verdict}]")
    return {
        "n_images":         len(latencies),
        "detection_rate_pct": round(det_rate, 1),
        "mean_T_max_degC":  round(mean_T_max, 1),
        "mean_latency_ms":  round(mean_lat, 2),
        "latency_verdict":  l_verdict,
    }


# -- 60 degC hard floor justification note (printed once) ----------------------

FLOOR_JUSTIFICATION = """
--- 60 degC hard floor justification ---
Ambient surface temperatures vary diurnally:
  Vegetation/soil:  5-40 degC (winter night to summer midday)
  Sun-warmed rock:  up to ~55 degC in direct sunlight
  Vehicle engines:  60-120 degC (a known false-positive source)
The 3-sigma rule adapts to ambient drift but can still trigger on local
hot-spots that are 3-sigma above a cool background (e.g., a 45 degC rock
when mean=30, std=4 -> threshold=42 degC).
The 60 degC floor eliminates these marginal cases:
  - Vegetation never reaches 60 degC under natural conditions.
  - Active flame fronts routinely exceed 200-500 degC.
  - Smoldering ignition points exceed 80-150 degC.
The floor therefore targets the discriminating temperature between
passively-heated non-fire surfaces and actively-burning material.
(Ben: verify against local ambient temperature data for your test site.)
"""


# -- Main ----------------------------------------------------------------------

def main():
    print("ThermalHotspotExtractor -- Step 4 Validation (Stage 1-2 only)")
    print("Stage 3 (temporal growth rate) is validated separately in")
    print("step4b_growth_kalman.py")
    print(FLOOR_JUSTIFICATION)

    results_a = run_synthetic_test()
    results_b = run_flir_proxy_test(n_images=30)

    # -- Overall summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    if results_a:
        print(f"  [synthetic] recall={results_a['recall_pct']}%  [{results_a['recall_verdict']}]  "
              f"latency={results_a['mean_latency_ms']}ms  [{results_a['latency_verdict']}]")
    if results_b:
        print(f"  [FLIR proxy] detection_rate={results_b['detection_rate_pct']}%  "
              f"latency={results_b['mean_latency_ms']}ms  [{results_b['latency_verdict']}]")

    # -- Save ------------------------------------------------------------------
    report = {
        "algorithm": {
            "sigma_factor":    SIGMA_FACTOR,
            "hard_floor_degC": HARD_FLOOR_C,
            "min_area_px":     MIN_AREA_PX,
            "fps":             FPS,
        },
        "test_A_synthetic": results_a,
        "test_B_flir_proxy": results_b,
    }
    def to_python(obj):
        """Recursively convert numpy scalars to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    out_path = FUSION_RESULTS / "step4_hotspot_report.json"
    with open(out_path, "w") as f:
        json.dump(to_python(report), f, indent=2)
    print(f"\n  Report saved -> {out_path}")


if __name__ == "__main__":
    main()
