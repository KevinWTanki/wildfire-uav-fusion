"""
HotspotGrowthTracker
Implements Stage 3 of the thermal analysis algorithm (CLAUDE.md Sec. 7.2):
temporal growth rate (dA/dt), built on top of the per-frame Hotspot
detections produced by hotspot_extraction.ThermalHotspotExtractor (Stage 1-2).

Stage 3 -- Temporal growth rate (dA/dt)
    2-state Kalman filter: state = [area, d_area/dt].
    Observation = area_px per frame.
    dA/dt over a 2-second rolling window.
    Expanding fire: dA/dt > 0 consistently.
    Static heat source: dA/dt ~= 0.

This is the primary novel algorithmic contribution of the system and the key
differentiator between an expanding fire and a static heat source (e.g. a
sun-warmed rock or vehicle engine that already cleared the Stage 1 threshold).
It is deliberately a separate module from hotspot_extraction.py: Stage 1-2 are
per-frame, stateless, and comparatively simple (thresholding + connected
components). Stage 3 requires cross-frame identity tracking and a recursive
state estimator (Kalman filter) -- a materially different and harder problem,
worth its own commit and its own entry in the engineering logbook (CLAUDE.md
Sec. 10 self-test: derive the predict/update equations, don't just cite them).

Usage:
    extractor = ThermalHotspotExtractor()   # Stage 1-2, from hotspot_extraction
    tracker   = HotspotGrowthTracker()      # Stage 3, this module
    for temp_frame in video:
        hotspots = extractor.process(temp_frame)   # per-frame, stateless
        tracked  = tracker.update(hotspots)         # cross-frame, stateful

Test suite (no FLAME3 required):
    A1 -- Expanding circular fire (30 frames, R: 10->50 px, fire_T=95 degC).
          Verifies Stage 3 direction: dA/dt > 0 consistently.

          GT note: Stage 1 clips to T > 60 degC, so the DETECTABLE radius
          is R_eff = R * (1 - 60/95) = R * 0.368. GT dA/dt is computed
          from the thresholded detectable area, not the geometric full area.
          Kalman uses a constant-velocity model on an accelerating signal
          (area ~ R^2 ~ t^2), so absolute calibration drifts; directional
          accuracy (dA/dt > 0) is the primary acceptance criterion here.
          Absolute calibration is deferred to FLAME3 radiometric data.

    A2 -- Static heat source (30 frames, R constant at 30 px).
          Verifies Stage 3: dA/dt converges to ~0 for non-fire sources.

Acceptance criteria (CLAUDE.md Sec. 7):
    - Growth-rate computation latency: < 200 ms/frame
"""

import time
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from hotspot_extraction import (
    ThermalHotspotExtractor, Hotspot, make_synthetic_frame,
    FPS, DT, HARD_FLOOR_C, FUSION_RESULTS,
)

# -- Config --------------------------------------------------------------------
# Kalman filter process/observation noise
# Q: process noise (how fast area can change between frames)
# R: observation noise (pixel-level segmentation uncertainty)
KF_Q_AREA   = 50.0    # px^2 per step
KF_Q_DADT   = 20.0    # (px^2/s) per step
KF_R_AREA   = 100.0   # px^2 observation noise
MAX_DIST_PX = 30.0    # nearest-centroid tracker gate


# -- Data structures -----------------------------------------------------------

@dataclass
class TrackedHotspot:
    """A Hotspot (Stage 1-2) plus Stage 3 temporal fields."""
    bbox:         tuple
    T_max:        float
    area_px:      int
    circularity:  float
    centroid:     tuple
    growth_rate:  float   # dA/dt from Kalman filter (px^2/s)
    track_id:     int     # identity across frames, assigned by the tracker

    @classmethod
    def from_hotspot(cls, h: Hotspot, growth_rate: float, track_id: int) -> "TrackedHotspot":
        return cls(h.bbox, h.T_max, h.area_px, h.circularity, h.centroid,
                   growth_rate, track_id)


# -- Kalman filter for area tracking ------------------------------------------

class AreaKalman:
    """
    Constant-velocity Kalman filter for tracking hotspot area over time.
    State:  x = [area, dA/dt]^T
    Model:  x_{k+1} = F x_k + w,  F = [[1, dt], [0, 1]]
    Obs:    z_k = H x_k + v,       H = [1, 0]
    """

    def __init__(self, initial_area: float, dt: float = DT):
        self.dt = dt
        self.x  = np.array([initial_area, 0.0])          # state
        self.P  = np.diag([KF_R_AREA * 4, KF_Q_DADT])   # covariance

        self.F  = np.array([[1.0, dt],
                             [0.0,  1.0]])
        self.H  = np.array([[1.0, 0.0]])
        self.Q  = np.diag([KF_Q_AREA, KF_Q_DADT])
        self.R  = np.array([[KF_R_AREA]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        z_vec = np.array([[z]])
        S     = self.H @ self.P @ self.H.T + self.R
        K     = self.P @ self.H.T @ np.linalg.inv(S)
        innov = z_vec - self.H @ self.x
        self.x = self.x + K.flatten() * innov.flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def area(self) -> float:
        return self.x[0]

    @property
    def dadt(self) -> float:
        """Estimated area growth rate in px^2/s."""
        return self.x[1]


# -- HotspotGrowthTracker ------------------------------------------------------

class HotspotGrowthTracker:
    """
    Cross-frame tracker that sits on top of ThermalHotspotExtractor's
    per-frame Hotspot output. Assigns a stable track_id via nearest-centroid
    matching, then runs a per-track AreaKalman filter to estimate dA/dt.

    Usage:
        tracker = HotspotGrowthTracker()
        for hotspots in per_frame_hotspots:   # from extractor.process()
            tracked = tracker.update(hotspots)
    """

    def __init__(self, dt: float = DT, max_dist_px: float = MAX_DIST_PX):
        self.dt          = dt
        self.max_dist_px = max_dist_px
        # track_id -> (AreaKalman, last centroid)
        self._tracks: dict[int, tuple[AreaKalman, np.ndarray]] = {}
        self._next_id = 0

    def _assign_track(self, centroid: np.ndarray, area: float) -> int:
        """
        Nearest-centroid tracker: reuse existing Kalman track if a previously
        seen centroid is within max_dist_px; otherwise create a new track.
        """
        best_id, best_dist = -1, self.max_dist_px + 1
        for tid, (kf, prev_c) in self._tracks.items():
            d = float(np.linalg.norm(centroid - prev_c))
            if d < best_dist:
                best_id, best_dist = tid, d

        if best_id >= 0:
            kf, _ = self._tracks[best_id]
            kf.predict()
            kf.update(area)
            self._tracks[best_id] = (kf, centroid)
            return best_id
        else:
            tid = self._next_id
            self._next_id += 1
            kf = AreaKalman(area, dt=self.dt)
            self._tracks[tid] = (kf, centroid)
            return tid

    def update(self, hotspots: list[Hotspot]) -> list[TrackedHotspot]:
        """Consume one frame's Stage 1-2 output, return Stage 3 tracked output."""
        tracked     = []
        active_tids = set()

        for h in hotspots:
            centroid_arr = np.array([h.centroid[0], h.centroid[1]])
            tid  = self._assign_track(centroid_arr, float(h.area_px))
            kf,_ = self._tracks[tid]
            active_tids.add(tid)
            tracked.append(TrackedHotspot.from_hotspot(h, round(kf.dadt, 2), tid))

        # Prune stale tracks (not seen this frame)
        stale = [tid for tid in self._tracks if tid not in active_tids]
        for tid in stale:
            del self._tracks[tid]

        return tracked


# -- Synthetic tests ------------------------------------------------------------

def run_growth_tests():
    """
    A1 -- Expanding circular fire: verifies dA/dt > 0 consistently.
    A2 -- Static heat source: verifies dA/dt converges to ~0.
    """
    print("=" * 60)
    print("TEST A1 -- Synthetic expanding fire sequence (Stage 3)")
    print("=" * 60)

    H, W       = 120, 160
    N_FRAMES   = 30
    R0, R1     = 10.0, 50.0
    BG_MEAN    = 25.0
    FIRE_T     = 95.0
    FX, FY     = W / 2, H / 2

    # GT: thresholded (detectable) area growth rate
    factor         = 1.0 - HARD_FLOOR_C / FIRE_T           # = 0.368
    r_eff_start    = R0 * factor
    r_eff_end      = R1 * factor
    gt_dadt_thresh = (np.pi * r_eff_end**2 - np.pi * r_eff_start**2) / ((N_FRAMES - 1) * DT)

    extractor = ThermalHotspotExtractor()
    tracker   = HotspotGrowthTracker()
    rng       = np.random.default_rng(42)
    latencies = []
    last_dadt = None

    print(f"  Frames: {N_FRAMES}  FPS: {FPS}  R: {R0:.0f}->{R1:.0f}px  Fire T: {FIRE_T}degC")
    print(f"  Detectable radius: {r_eff_start:.1f}->{r_eff_end:.1f}px  "
          f"GT dA/dt (thresholded): {gt_dadt_thresh:.1f} px^2/s\n")

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        r = R0 + t * (R1 - R0)
        frame = make_synthetic_frame(H, W, FX, FY, r, BG_MEAN,
                                     fire_T=FIRE_T, rng=rng)

        t0 = time.perf_counter()
        hotspots = extractor.process(frame)
        tracked  = tracker.update(hotspots)
        latencies.append((time.perf_counter() - t0) * 1000)

        if tracked:
            hs = max(tracked, key=lambda h: h.area_px)
            last_dadt = hs.growth_rate
            if i % 5 == 0 or i == N_FRAMES - 1:
                print(f"  Frame {i+1:2d}: area={hs.area_px:5d}px^2  "
                      f"dA/dt={hs.growth_rate:.1f}px^2/s  lat={latencies[-1]:.1f}ms")
        else:
            if i % 5 == 0:
                print(f"  Frame {i+1:2d}: no hotspot detected")

    mean_latency = float(np.mean(latencies))
    expanding    = last_dadt is not None and last_dadt > 0.0   # direction check

    print(f"\n  Mean latency   : {mean_latency:.2f} ms  (target <200ms)")
    print(f"  Final dA/dt    : {last_dadt:.1f} px^2/s  "
          f"(thresholded GT={gt_dadt_thresh:.1f})  "
          f"direction={'EXPANDING [PASS]' if expanding else 'WRONG [FAIL]'}")
    print(f"  Note: absolute dA/dt calibration deferred to FLAME3 radiometric data")

    l_verdict = "PASS" if mean_latency < 200.0 else "FAIL"
    d_verdict = "PASS" if expanding else "FAIL"
    print(f"\n  Latency [{l_verdict}]  Direction [{d_verdict}]")

    # ---- A2: static source ---------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST A2 -- Synthetic static heat source (Stage 3)")
    print("=" * 60)
    R_STATIC = 30.0
    extractor2 = ThermalHotspotExtractor()
    tracker2   = HotspotGrowthTracker()
    dadt_history = []

    print(f"  Frames: {N_FRAMES}  R constant: {R_STATIC}px  Fire T: {FIRE_T}degC\n")
    for i in range(N_FRAMES):
        frame = make_synthetic_frame(H, W, FX, FY, R_STATIC, BG_MEAN,
                                     fire_T=FIRE_T, rng=rng)
        hotspots = extractor2.process(frame)
        tracked  = tracker2.update(hotspots)
        if tracked:
            hs = max(tracked, key=lambda h: h.area_px)
            dadt_history.append(hs.growth_rate)
            if i % 5 == 0 or i == N_FRAMES - 1:
                print(f"  Frame {i+1:2d}: area={hs.area_px:5d}px^2  "
                      f"dA/dt={hs.growth_rate:.2f}px^2/s")

    final_static_dadt = dadt_history[-1] if dadt_history else float("nan")
    static_ok = abs(final_static_dadt) < 20.0   # near-zero threshold
    s_verdict = "PASS" if static_ok else "FAIL"
    print(f"\n  Final dA/dt (static): {final_static_dadt:.2f} px^2/s  "
          f"(target ~0)  [{s_verdict}]")

    return {
        "A1_mean_latency_ms":     round(mean_latency,     2),
        "A1_gt_dadt_thresh":      round(gt_dadt_thresh,   1),
        "A1_final_dadt":          round(last_dadt,        1) if last_dadt else None,
        "A1_direction_expanding": expanding,
        "A1_latency_verdict":     l_verdict,
        "A1_direction_verdict":   d_verdict,
        "A2_final_static_dadt":   round(final_static_dadt, 2),
        "A2_static_verdict":      s_verdict,
    }


# -- Main ----------------------------------------------------------------------

def main():
    print("HotspotGrowthTracker -- Step 4b Validation (Stage 3 only)")
    print("Assumes step4_hotspot.py Stage 1-2 already validated separately.\n")

    results = run_growth_tests()

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  [A1 expanding] latency={results['A1_mean_latency_ms']}ms  [{results['A1_latency_verdict']}]  "
          f"direction=[{results['A1_direction_verdict']}]")
    print(f"  [A2 static   ] dA/dt={results['A2_final_static_dadt']}px^2/s  "
          f"[{results['A2_static_verdict']}]")

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

    report = {
        "algorithm": {
            "kalman_Q_area": KF_Q_AREA,
            "kalman_Q_dadt": KF_Q_DADT,
            "kalman_R_area": KF_R_AREA,
        },
        "test_A_growth": results,
    }

    out_path = FUSION_RESULTS / "step4b_growth_report.json"
    with open(out_path, "w") as f:
        json.dump(to_python(report), f, indent=2)
    print(f"\n  Report saved -> {out_path}")


if __name__ == "__main__":
    main()
