"""
YOLOv8n training on FASDD_UAV -- minimal integration baseline.

This is deliberately NOT an attempt at a strong detector. Its only job is
to replace fusion/iou_fusion.py's simulate_yolo_boxes() placeholder with a
model that produces real boxes, so the Phase 2+3 pipeline can be measured
end to end against the throughput criterion. The mAP / false-positive-rate
rows tracked in detection/evaluate.py are the bar a dedicated detector
effort would have to clear -- not a target this run is expected to hit.
Report results as the "integration baseline", never as a validated
detector.

Config fixed here (not exposed as flags): YOLOv8n, COCO-pretrained init,
first 10 backbone layers frozen, imgsz 640. Everything else is a named
parameter with a small default -- a long run is wasted on a model this
task doesn't require to be strong.

Per-epoch validation is intentionally disabled (val=False): CPU benchmarking
on this dev workstation measured ~2.27s/batch train vs. ~2.25s/batch val
(batch=16 train / 32 val) -- a full validation pass over all 8,365 val
images costs ~10 minutes on its own, EVERY epoch, for a number nothing in
this script consumes (detection/evaluate.py is the dedicated, separate
evaluation step). Ultralytics still always validates on
the final epoch regardless of this flag (ultralytics/engine/trainer.py), so
a training run's own log does show one real validation pass at the end.

Epoch default of 3 reflects that same CPU benchmark: ~30 min/epoch train-only
+ 1 unavoidable ~10 min final validation, so epochs=3 is roughly 1h40 wall
time on this hardware -- already the boundary of what "small" can mean here,
not a token number picked to look small.

Usage:
    python detection/train_yolov8.py
    python detection/train_yolov8.py --epochs 10 --batch 8 --device 0
"""
import argparse
from pathlib import Path

import torch

# ultralytics==8.0.196 calls torch.load() on the pretrained checkpoint
# without weights_only=False. PyTorch 2.6 flipped that default to True,
# which rejects ultralytics.nn.tasks.DetectionModel as an unlisted global
# -- unrelated to this project's own code, torch itself is not pinned to a
# specific version and pip resolved a torch newer than ultralytics
# 8.0.196 expected. Patched narrowly here rather than pinning an
# older torch, since the checkpoint is ultralytics' own official release
# asset (github.com/ultralytics/assets), not user-supplied data.
_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO

from dataset_prep import DATASET_YAML, resolve_dataset_yaml

# Freeze the COCO-pretrained backbone, fine-tune only the detection head --
# appropriate given the limited wildfire-specific training volume.
FROZEN_BACKBONE_LAYERS = 10
# Matches the COCO-pretrained checkpoint's expected input resolution.
IMGSZ = 640


def train(epochs: int = 3,
          batch: int = 16,
          lr0: float = 0.01,
          device: str = "cpu",
          workers: int = 4,
          project: str = "runs/detect",
          name: str = "fasdd_uav_integration_baseline",
          dataset_yaml: Path = DATASET_YAML):
    """Fine-tune YOLOv8n on FASDD_UAV. Returns the Ultralytics training
    results object (used by detection/evaluate.py for the run directory)."""
    resolved_yaml = resolve_dataset_yaml(dataset_yaml)

    model = YOLO("yolov8n.pt")  # COCO-pretrained init
    return model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=IMGSZ,
        lr0=lr0,
        freeze=FROZEN_BACKBONE_LAYERS,
        device=device,
        workers=workers,
        val=False,   # see module docstring -- detection/evaluate.py evaluates separately
        project=project,
        name=name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3,
                         help="default kept small -- integration baseline, not a quality run "
                              "(~30min/epoch CPU-measured on this workstation; default: 3)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu', a CUDA index like '0', or 'mps'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="fasdd_uav_integration_baseline")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
