"""
FASDD_UAV layout adapter for Ultralytics.

FASDD_UAV ships train/val/test image lists and label files nested two levels
below images/, under annotations/YOLO_UAV/. Ultralytics assumes a flatter
layout in two places that do not hold for this dataset as downloaded:

  1. ultralytics/data/base.py:get_img_files() resolves a "./images/NAME.jpg"
     entry in a split .txt file relative to that .txt file's OWN directory
     -- not relative to any path configured in the dataset YAML. For that
     substitution to land on the real images/ folder, train.txt/val.txt/
     test.txt must sit next to images/, at the FASDD_UAV root, not under
     annotations/YOLO_UAV/ where FASDD ships them.
  2. ultralytics/data/utils.py:img2label_paths() derives each label path by
     swapping the LAST "/images/" segment of the image path for "/labels/".
     That only finds real label files if a labels/ directory exists as a
     sibling of images/ at the FASDD_UAV root -- FASDD ships labels/ under
     annotations/YOLO_UAV/ instead.

ensure_fasdd_uav_layout() bridges both gaps without moving or duplicating
any image or label bytes: the three split lists are small text files (safe
to copy), and labels/ is bridged with a directory junction (Windows,
non-admin) rather than copying 25,097 label files.

resolve_dataset_yaml() handles a separate Ultralytics quirk: check_det_dataset
(ultralytics/data/utils.py) resolves a *relative* `path:` value against
Ultralytics' own global DATASETS_DIR setting, not against the YAML file's
own location -- so the committed, path-free detection/fasdd_uav.yaml cannot
be passed to Ultralytics as-is from an arbitrary clone location. This writes
a resolved copy (absolute `path`, computed from this file's own location) to
a temp file and returns that path, so the checked-in YAML never needs a
hardcoded absolute path.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

DETECTION_DIR  = Path(__file__).resolve().parent
FASDD_UAV_ROOT = DETECTION_DIR.parent / "data" / "FASDD_UAV"
DATASET_YAML   = DETECTION_DIR / "fasdd_uav.yaml"
SPLITS = ("train", "val", "test")


def ensure_fasdd_uav_layout(root: Path = FASDD_UAV_ROOT) -> None:
    """Create the root-level split lists + labels junction FASDD_UAV needs
    to match Ultralytics' path conventions. Idempotent; safe before every
    train/eval run. Never touches images/ or the canonical annotations/
    tree -- only adds a junction and copies three small .txt files."""
    annot_dir = root / "annotations" / "YOLO_UAV"
    if not annot_dir.is_dir():
        raise FileNotFoundError(f"FASDD_UAV annotations not found at {annot_dir}")

    for split in SPLITS:
        shutil.copyfile(annot_dir / f"{split}.txt", root / f"{split}.txt")

    labels_link = root / "labels"
    if not labels_link.exists():
        labels_target = annot_dir / "labels"
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(labels_link), str(labels_target)],
            check=True, capture_output=True,
        )


def resolve_dataset_yaml(yaml_path: Path = DATASET_YAML,
                          fasdd_root: Path = FASDD_UAV_ROOT) -> Path:
    """Run ensure_fasdd_uav_layout(), then write a resolved copy of
    yaml_path (absolute `path`, everything else unchanged) to a temp file
    and return its path -- what Ultralytics' `data=` argument should
    receive. See module docstring for why the committed YAML can't carry
    an absolute path itself."""
    ensure_fasdd_uav_layout(fasdd_root)
    data = yaml.safe_load(yaml_path.read_text())
    data["path"] = str(fasdd_root)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


if __name__ == "__main__":
    ensure_fasdd_uav_layout()
    print(f"FASDD_UAV layout ready at {FASDD_UAV_ROOT}")
    for split in SPLITS:
        print(f"  {split}.txt -> {(FASDD_UAV_ROOT / f'{split}.txt').exists()}")
    print(f"  labels/    -> {(FASDD_UAV_ROOT / 'labels').exists()}")
    resolved = resolve_dataset_yaml()
    print(f"  resolved dataset yaml -> {resolved}")
