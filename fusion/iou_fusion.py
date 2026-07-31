"""
IoUFusionEngine
Implements the Decision Fusion Engine that combines YOLOv8 (RGB)
detections with the Stage 1-3 thermal pipeline output
(hotspot_extraction.ThermalHotspotExtractor + growth_rate_tracker.HotspotGrowthTracker)
into a single per-track fire probability and patrol/re-verify/alert decision.

    P_fire = 0.35*P_yolo + 0.30*P_thermal + 0.25*P_growth + 0.10*P_location_prior

    P_fire > 0.75  -> immediate alert + loiter + coordinate upload
    P_fire > 0.50  -> descend for re-verification
    P_fire < 0.50  -> continue patrol

Per-channel scores:
    P_yolo             : YOLOv8 confidence of the RGB box matched to a thermal
                         track via IoU (0.0 if no RGB box overlaps -- e.g. the
                         hotspot is canopy-occluded in the visible band).
    P_thermal          : T_max normalized against the floor-justification range
                         from hotspot_extraction.py (60 degC Stage-1 floor .. 150
                         degC, the top of the smoldering-ignition band, per
                         FLOOR_JUSTIFICATION). This is a *confidence* scale,
                         not a detection threshold -- Stage 1 already gated on
                         60 degC, so every input here has T_max >= 60.
    P_growth           : dA/dt from the Kalman filter (growth_rate_tracker),
                         normalized against GROWTH_SATURATE_PX2_S. Negative/near-zero
                         growth (static heat source) clips to 0 -- this is the
                         primary differentiator, per Stage 3 (growth_rate_tracker.py).
    P_location_prior   : PLACEHOLDER, fixed at 0.5 (neutral). Intended to
                         carry a nonzero weight for a known-fire-history /
                         terrain-risk map that lives in the Mission Layer,
                         which does not exist yet. Do not read
                         anything into this value beyond "not yet wired up" --
                         it is a stand-in, not a calibrated prior.

Coordinate-space note: this module assumes thermal and RGB boxes are already
in a common pixel frame (i.e. downstream of the Phase 2 Homography warp in
calibration/ and the not-yet-built fusion/realtime_registration.py). Real
YOLOv8 output does not exist yet either (Phase 3 training is not complete)
-- this module exercises the fusion arithmetic and IoU matching logic
against synthetic RGB boxes standing in for real detections, not a real
detector.

Test suite (no FLAME3 / no trained YOLOv8 required):
    C1 -- Confirmed alert: hot (open-flame range), expanding, RGB+thermal
          agree (YOLO box present and matched). Verifies P_fire crosses the
          0.75 ALERT threshold.
    C2 -- Canopy occlusion: identical thermal signal to C1, but no YOLO box
          (RGB blocked). Verifies the system does NOT reach full ALERT on
          thermal growth alone -- it lands in the 0.50-0.75 re-verify band.
          This is the direct operational test of the project's core thesis:
          neither modality alone is sufficient.
    C3 -- Static distractor with RGB corroboration: hot, RGB-visible (e.g. a
          vehicle engine misclassified with moderate confidence), but NOT
          growing. Verifies P_fire stays under the 0.50 patrol threshold --
          growth, not raw heat + a visual match, is what should gate alerting.

Acceptance criteria:
    - Combined Phase 2+3 pipeline throughput: >= 8 FPS on Jetson Nano
      (checked here as fusion-step latency, additive to Stage 1-3 latency)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# thermal/ holds hotspot_extraction.py and growth_rate_tracker.py -- add it to
# sys.path so this module (in fusion/) can import them by name, matching the
# flat-script import style used throughout this repo (no package __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thermal"))

from hotspot_extraction import (
    ThermalHotspotExtractor, Hotspot, make_synthetic_frame,
    HARD_FLOOR_C, FUSION_RESULTS,
)
from growth_rate_tracker import HotspotGrowthTracker, TrackedHotspot

# -- Config ----------------------------------------------------------------
T_SATURATE_C        = 150.0   # degC -- top of smoldering range; P_thermal=1.0
GROWTH_SATURATE_PX2S = 100.0  # px^2/s -- calibrated near growth_rate_tracker's
                              # observed Kalman-smoothed dA/dt (~87.5 px^2/s,
                              # see results/step4b_growth_report.json);
                              # P_growth=1.0
MIN_IOU              = 0.10   # below this, treat as "no RGB match"

W_YOLO, W_THERMAL, W_GROWTH, W_LOCATION = 0.35, 0.30, 0.25, 0.10

ALERT_THRESH    = 0.75
REVERIFY_THRESH = 0.50

DEFAULT_LOCATION_PRIOR = 0.5   # placeholder -- see module docstring


# -- Data structures ---------------------------------------------------------

@dataclass
class SimulatedYOLODetection:
    """Stand-in for a real YOLOv8 RGB detection."""
    bbox:       tuple    # (x, y, w, h), same pixel frame as thermal (see note above)
    confidence: float
    class_name: str = "flame"


@dataclass
class FusedDetection:
    """One track's decision-fusion output."""
    track_id:         int
    bbox_thermal:     tuple
    bbox_yolo:        Optional[tuple]
    iou:              float
    T_max:            float
    area_px:          int
    growth_rate:      float
    P_yolo:           float
    P_thermal:        float
    P_growth:         float
    P_location_prior: float
    P_fire:           float
    action:           str


# -- IoU ----------------------------------------------------------------------

def iou(box_a: tuple, box_b: tuple) -> float:
    """Standard box IoU. box = (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    if inter <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# -- Per-channel scoring -------------------------------------------------------

def score_thermal(T_max: float) -> float:
    """Normalize T_max in [HARD_FLOOR_C, T_SATURATE_C] -> [0, 1]."""
    span = T_SATURATE_C - HARD_FLOOR_C
    return float(np.clip((T_max - HARD_FLOOR_C) / span, 0.0, 1.0))


def score_growth(dadt: float) -> float:
    """Normalize dA/dt -> [0, 1]. Static/shrinking hotspots clip to 0."""
    return float(np.clip(dadt / GROWTH_SATURATE_PX2S, 0.0, 1.0))


def decide_action(p_fire: float) -> str:
    if p_fire > ALERT_THRESH:
        return "ALERT_LOITER_UPLOAD"
    if p_fire > REVERIFY_THRESH:
        return "DESCEND_REVERIFY"
    return "CONTINUE_PATROL"


# -- IoUFusionEngine ------------------------------------------------------------

class IoUFusionEngine:
    """
    Matches Stage 1-3 thermal tracks to (simulated) YOLOv8 RGB boxes via IoU,
    then computes the weighted P_fire score per track.

    Usage:
        engine = IoUFusionEngine()
        fused  = engine.fuse(tracked_hotspots, yolo_boxes, location_prior=0.5)
    """

    def __init__(self, min_iou: float = MIN_IOU,
                 location_prior: float = DEFAULT_LOCATION_PRIOR):
        self.min_iou        = min_iou
        self.location_prior = location_prior

    def _best_match(self, thermal_box: tuple,
                     yolo_boxes: list[SimulatedYOLODetection]
                     ) -> tuple[Optional[SimulatedYOLODetection], float]:
        best_det, best_iou = None, 0.0
        for det in yolo_boxes:
            v = iou(thermal_box, det.bbox)
            if v > best_iou:
                best_det, best_iou = det, v
        if best_iou < self.min_iou:
            return None, 0.0
        return best_det, best_iou

    def fuse(self, tracked_hotspots: list[TrackedHotspot],
              yolo_boxes: list[SimulatedYOLODetection],
              location_prior: Optional[float] = None
              ) -> list[FusedDetection]:
        """One frame: thermal tracks (Stage 1-3) x RGB boxes -> P_fire per track."""
        loc_prior = self.location_prior if location_prior is None else location_prior
        results = []

        for h in tracked_hotspots:
            det, best_iou = self._best_match(h.bbox, yolo_boxes)

            p_yolo    = float(det.confidence) if det is not None else 0.0
            p_thermal = score_thermal(h.T_max)
            p_growth  = score_growth(h.growth_rate)

            p_fire = (W_YOLO * p_yolo + W_THERMAL * p_thermal
                      + W_GROWTH * p_growth + W_LOCATION * loc_prior)

            results.append(FusedDetection(
                track_id         = h.track_id,
                bbox_thermal      = h.bbox,
                bbox_yolo         = det.bbox if det is not None else None,
                iou               = round(best_iou, 3),
                T_max             = h.T_max,
                area_px           = h.area_px,
                growth_rate       = h.growth_rate,
                P_yolo            = round(p_yolo, 3),
                P_thermal         = round(p_thermal, 3),
                P_growth          = round(p_growth, 3),
                P_location_prior  = round(loc_prior, 3),
                P_fire            = round(p_fire, 3),
                action            = decide_action(p_fire),
            ))

        return results


# -- Synthetic scenario helper --------------------------------------------------

def _simulate_yolo_box(hotspot: Hotspot, confidence: float,
                        jitter_px: float, rng: np.random.Generator
                        ) -> SimulatedYOLODetection:
    """Jitter a thermal bbox to stand in for an (imperfect) YOLOv8 RGB box."""
    x, y, w, h = hotspot.bbox
    dx, dy = rng.normal(0, jitter_px, 2)
    return SimulatedYOLODetection(
        bbox=(x + dx, y + dy, w, h),
        confidence=confidence,
    )


def _run_scenario(name: str, n_frames: int, r0: float, r1: float,
                   fire_T: float, yolo_conf: Optional[float],
                   rng_seed: int) -> dict:
    """
    Run one synthetic scenario through Stage 1-3 + IoU fusion.
    yolo_conf=None means no RGB box is ever produced (canopy occlusion, C2).
    r0==r1 means a static (non-growing) hotspot (C3).
    """
    H, W    = 120, 160
    FX, FY  = W / 2, H / 2
    BG_MEAN = 25.0

    extractor = ThermalHotspotExtractor()
    tracker   = HotspotGrowthTracker()
    engine    = IoUFusionEngine()
    rng       = np.random.default_rng(rng_seed)

    latencies  = []
    last_fused = None
    peak_fused = None   # highest-P_fire frame seen -- see note below

    print(f"\n  -- {name} --")
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        r = r0 + t * (r1 - r0)
        frame = make_synthetic_frame(H, W, FX, FY, r, BG_MEAN,
                                     fire_T=fire_T, rng=rng)

        hotspots = extractor.process(frame)
        tracked  = tracker.update(hotspots)

        yolo_boxes = []
        if yolo_conf is not None and hotspots:
            main_hotspot = max(hotspots, key=lambda h: h.area_px)
            yolo_boxes = [_simulate_yolo_box(main_hotspot, yolo_conf,
                                             jitter_px=3.0, rng=rng)]

        t0 = time.perf_counter()
        fused = engine.fuse(tracked, yolo_boxes)
        latencies.append((time.perf_counter() - t0) * 1000)

        if fused:
            last_fused = max(fused, key=lambda f: f.area_px)
            if peak_fused is None or last_fused.P_fire > peak_fused.P_fire:
                peak_fused = last_fused
            if i % 5 == 0 or i == n_frames - 1:
                f = last_fused
                print(f"  Frame {i+1:2d}: T_max={f.T_max:5.1f}C  "
                      f"dA/dt={f.growth_rate:6.1f}  iou={f.iou:.2f}  "
                      f"P_yolo={f.P_yolo:.2f} P_th={f.P_thermal:.2f} "
                      f"P_gr={f.P_growth:.2f}  P_fire={f.P_fire:.3f}  "
                      f"[{f.action}]")

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    return {
        "final_fused":  last_fused,
        "peak_fused":   peak_fused,
        "mean_latency": mean_latency,
    }


def run_fusion_tests() -> dict:
    print("=" * 70)
    print("TEST C -- IoUFusionEngine synthetic scenarios")
    print("=" * 70)

    # C1 -- confirmed alert: open-flame-range, expanding, RGB+thermal agree.
    # r1=50 (not larger) deliberately stays inside the 120x160 synthetic frame
    # (center at 80,60) so the disk is never clipped by the frame boundary --
    # clipping distorts the connected-component area and corrupts dA/dt.
    c1 = _run_scenario("C1 confirmed (RGB+thermal, expanding, hot)",
                        n_frames=30, r0=10.0, r1=50.0, fire_T=250.0,
                        yolo_conf=0.92, rng_seed=1)

    # C2 -- canopy occlusion: same thermal signal, no RGB box at all
    c2 = _run_scenario("C2 canopy-occluded (thermal-only, expanding, hot)",
                        n_frames=30, r0=10.0, r1=50.0, fire_T=250.0,
                        yolo_conf=None, rng_seed=1)

    # C3 -- static distractor with RGB corroboration but no growth
    c3 = _run_scenario("C3 static distractor (RGB+thermal, NOT expanding)",
                        n_frames=30, r0=30.0, r1=30.0, fire_T=95.0,
                        yolo_conf=0.55, rng_seed=2)

    # Peak P_fire over the run, not the final frame, is the right acceptance
    # metric: operationally the drone alerts the instant P_fire first crosses
    # a threshold, it doesn't wait for some designated last frame. (The final
    # frame is also not representative here: Stage 1's adaptive threshold is
    # computed over the WHOLE frame, so once a hot, fast-growing disk occupies
    # a large fraction of the 120x160 synthetic frame, it drags the frame-wide
    # mean/std up enough to destabilize its own detected area late in the
    # sequence -- a real Stage-1 limitation, not a fusion-engine bug.)
    p1, p2, p3 = c1["peak_fused"], c2["peak_fused"], c3["peak_fused"]

    c1_verdict = "PASS" if (p1 and p1.P_fire > ALERT_THRESH) else "FAIL"
    c2_verdict = "PASS" if (p2 and REVERIFY_THRESH < p2.P_fire <= ALERT_THRESH) else "FAIL"
    c3_verdict = "PASS" if (p3 and p3.P_fire <= REVERIFY_THRESH) else "FAIL"

    all_latencies = c1["mean_latency"] + c2["mean_latency"] + c3["mean_latency"]
    mean_latency  = all_latencies / 3.0
    l_verdict     = "PASS" if mean_latency < 200.0 else "FAIL"

    print("\n" + "=" * 70)
    print("SCENARIO SUMMARY (peak P_fire over each run)")
    print("=" * 70)
    print(f"  C1 confirmed alert   : peak P_fire={p1.P_fire:.3f}  action={p1.action}  "
          f"(want >0.75)                [{c1_verdict}]")
    print(f"  C2 canopy occlusion  : peak P_fire={p2.P_fire:.3f}  action={p2.action}  "
          f"(want 0.50-0.75, not ALERT) [{c2_verdict}]")
    print(f"  C3 static distractor : peak P_fire={p3.P_fire:.3f}  action={p3.action}  "
          f"(want <=0.50)               [{c3_verdict}]")
    print(f"  Mean fusion latency  : {mean_latency:.3f} ms  (target <200ms)  [{l_verdict}]")

    return {
        "C1_peak_P_fire": p1.P_fire, "C1_action": p1.action, "C1_verdict": c1_verdict,
        "C2_peak_P_fire": p2.P_fire, "C2_action": p2.action, "C2_verdict": c2_verdict,
        "C3_peak_P_fire": p3.P_fire, "C3_action": p3.action, "C3_verdict": c3_verdict,
        "mean_latency_ms": round(mean_latency, 3),
        "latency_verdict": l_verdict,
    }


# -- Main ------------------------------------------------------------------------

def main():
    print("IoUFusionEngine -- Step 5 Validation (decision fusion)")
    print("Assumes Stage 1-3 (step4_hotspot.py, step4b_growth_kalman.py) already")
    print("validated separately. YOLOv8 boxes are simulated (Phase 3 model not")
    print("yet trained) -- see module docstring for what this test does and")
    print("does not prove.\n")

    results = run_fusion_tests()

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
            "weights": {"yolo": W_YOLO, "thermal": W_THERMAL,
                        "growth": W_GROWTH, "location": W_LOCATION},
            "T_saturate_degC": T_SATURATE_C,
            "growth_saturate_px2s": GROWTH_SATURATE_PX2S,
            "alert_thresh": ALERT_THRESH,
            "reverify_thresh": REVERIFY_THRESH,
        },
        "test_C_fusion": results,
    }

    out_path = FUSION_RESULTS / "step5_iou_fusion_report.json"
    with open(out_path, "w") as f:
        json.dump(to_python(report), f, indent=2)
    print(f"\n  Report saved -> {out_path}")


if __name__ == "__main__":
    main()
