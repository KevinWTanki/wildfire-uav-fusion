"""
YOLOv8n RGB inference wrapper for the fusion pipeline.

Wraps a detection/train_yolov8.py checkpoint so its output matches exactly
what fusion/iou_fusion.py's IoUFusionEngine.fuse() already expects for
yolo_boxes: a list of fusion.iou_fusion.SimulatedYOLODetection (bbox as
(x, y, w, h), confidence, class_name). That dataclass is imported directly
from iou_fusion.py rather than redefined here, so there is exactly one
definition of the shape fuse() consumes.

Standalone only in this step: this module does not import, modify, or
wire into fusion/iou_fusion.py's own placeholder detection generator
(_simulate_yolo_box) or the live fusion path. Wiring this wrapper in to
replace that placeholder is a separate follow-up once it is validated
on its own.

Coordinate-space note (see iou_fusion.py's module docstring): IoUFusionEngine
assumes thermal and RGB boxes already share a common pixel frame, downstream
of the Phase 2 Homography warp. This wrapper does not perform that warp --
the caller must pass an already-registered frame.
"""
import sys
from pathlib import Path

import numpy as np
import torch

# fusion/ holds the SimulatedYOLODetection shape this wrapper must match --
# add it to sys.path, matching the flat-script import style iou_fusion.py
# itself already uses to import from thermal/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion"))

# see train_yolov8.py for why this is patched (ultralytics 8.0.196 predates
# PyTorch 2.6's weights_only=True default; official checkpoint, trusted source)
_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO

from iou_fusion import SimulatedYOLODetection

DEFAULT_WEIGHTS = (Path(__file__).resolve().parent / "runs" / "detect"
                   / "fasdd_uav_integration_baseline" / "weights" / "best.pt")
CLASS_NAMES = {0: "flame", 1: "smoke"}   # id order per fasdd_uav.yaml Sec. 1b


class YOLOv8Detector:
    """Loads a trained checkpoint once; call .detect(frame) per frame."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.25, device: str = "cpu"):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"No trained weights at {weights}. Run detection/train_yolov8.py first."
            )
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list[SimulatedYOLODetection]:
        """RGB frame (H, W, 3) -> list[SimulatedYOLODetection], the exact
        shape IoUFusionEngine.fuse() already consumes for yolo_boxes.
        bbox is (x, y, w, h) in the input frame's own pixel coordinates."""
        results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                detections.append(SimulatedYOLODetection(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(box.conf[0].item()),
                    class_name=CLASS_NAMES.get(class_id, str(class_id)),
                ))
        return detections


def _smoke_test():
    """Standalone check: load a checkpoint, run one frame, print the
    returned SimulatedYOLODetection list. Does not touch iou_fusion.py."""
    import cv2

    detector = YOLOv8Detector()
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "FASDD_UAV" / "images"
    sample = next(sample_dir.glob("bothFireAndSmoke_UAV*.jpg"))
    frame = cv2.imread(str(sample))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = detector.detect(frame)
    print(f"{sample.name}: {len(detections)} detection(s)")
    for d in detections:
        print(f"  {d}")


if __name__ == "__main__":
    _smoke_test()
