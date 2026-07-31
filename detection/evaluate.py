"""
Evaluation and reporting for the FASDD_UAV YOLOv8n run produced by
detection/train_yolov8.py.

This run is the "integration baseline" -- a detector good enough to replace
fusion/iou_fusion.py's placeholder so the pipeline can be measured end to
end, not a validated detector. Every label in this module's output uses
that term; the mAP / false-positive targets recorded here are the bar a
dedicated detector effort would have to clear, not a claim this run makes.

Reports per-class AP50 for flame and smoke SEPARATELY (FASDD_UAV carries a
~2:1 fire:smoke instance imbalance, so an overall mAP would be carried by
the majority class and hide exactly the minority-class weakness that
matters for an early-warning system), and false positives
per image on the neitherFireNorSmoke negative subset identified in the
Step 1 dataset audit (Decision 002's false-positive criterion).

Usage:
    python detection/evaluate.py --weights runs/detect/fasdd_uav_integration_baseline/weights/best.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch

# see train_yolov8.py for why this is patched (ultralytics 8.0.196 predates
# PyTorch 2.6's weights_only=True default; official checkpoint, trusted source)
_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{**kw, "weights_only": False})

import yaml
from ultralytics import YOLO

from dataset_prep import DATASET_YAML, FASDD_UAV_ROOT, resolve_dataset_yaml

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLASS_NAMES = {0: "flame", 1: "smoke"}   # id order per fasdd_uav.yaml Sec. 1b
FASDD_VERSION = "V9"
NEGATIVE_PREFIX = "neitherFireNorSmoke_UAV"   # Step 1 audit Sec. 1a/1d


def _read_split_stems(split: str) -> list[str]:
    lines = (FASDD_UAV_ROOT / "annotations" / "YOLO_UAV" / f"{split}.txt").read_text().strip().splitlines()
    return [Path(line.strip().split("/")[-1]).stem for line in lines if line.strip()]


def negative_false_positive_rate(model: YOLO, split: str, conf: float, device: str,
                                  batch_size: int = 32) -> dict:
    """False positives per image on the neitherFireNorSmoke (empty-label)
    subset of `split` -- the negative-set false-positive criterion from
    Decision 002."""
    images_dir = FASDD_UAV_ROOT / "images"
    stems = [s for s in _read_split_stems(split) if s.startswith(NEGATIVE_PREFIX)]
    image_paths = [str(images_dir / f"{s}.jpg") for s in stems]

    total_fp = 0
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for r in model.predict(batch, conf=conf, device=device, verbose=False):
            total_fp += len(r.boxes)

    n = len(image_paths)
    return {
        "split": split,
        "n_negative_images": n,
        "confidence_threshold": conf,
        "total_false_positive_boxes": total_fp,
        "false_positives_per_image": round(total_fp / n, 4) if n else None,
    }


def _training_config_from_weights(weights_path: Path) -> dict:
    """Best-effort: pull the actual training args Ultralytics saved next to
    the weights (runs/detect/<name>/args.yaml), rather than asking the
    caller to retype them."""
    args_path = weights_path.parent.parent / "args.yaml"
    if not args_path.exists():
        return {}
    raw = yaml.safe_load(args_path.read_text())
    keys = ("epochs", "batch", "imgsz", "lr0", "freeze", "device")
    return {k: raw[k] for k in keys if k in raw}


def evaluate(weights: str, split: str = "test", conf: float = 0.25, device: str = "cpu"):
    weights_path = Path(weights)
    resolved_yaml = resolve_dataset_yaml(DATASET_YAML)
    model = YOLO(str(weights_path))

    val_metrics = model.val(data=str(resolved_yaml), split=split, device=device, verbose=False)
    box = val_metrics.box

    per_class_ap50 = {}
    for i, class_id in enumerate(box.ap_class_index):
        per_class_ap50[CLASS_NAMES[int(class_id)]] = round(float(box.ap50[i]), 4)
    for name in CLASS_NAMES.values():
        per_class_ap50.setdefault(name, None)   # class absent from predictions/GT on this split

    fp_report = negative_false_positive_rate(model, split=split, conf=conf, device=device)

    report = {
        "label": "integration_baseline",
        "note": "Working detector for Phase 2+3 pipeline integration. "
                "Integration baseline, not a validated detector -- "
                "detection quality was not the objective of this run.",
        "dataset": {"source": "FASDD_UAV", "version": FASDD_VERSION, "eval_split": split},
        "weights": str(weights_path),
        "training_config": _training_config_from_weights(weights_path),
        "metrics": {
            "overall_mAP50": round(float(box.map50), 4),
            "overall_mAP50_95": round(float(box.map), 4),
            "per_class_AP50": per_class_ap50,
        },
        "negative_set_false_positives": fp_report,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "yolov8_fasdd_uav_integration_baseline_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport saved -> {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    evaluate(args.weights, split=args.split, conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()
