"""
Step 2 · Verify Dataset Integrity
Dataset: FLIR ADAS aligned subset
Root:    C:/wildfire_uav/data/flir_adas/aligned

Directory layout expected:
  aligned/
  ├── AnnotatedImages/   FLIR_XXXXX_PreviewData.jpeg  (thermal)
  │                      FLIR_XXXXX_RGB.jpg            (RGB)
  ├── Annotations/       FLIR_XXXXX_PreviewData.xml    (Pascal VOC bbox)
  │                      FLIR_XXXXX_mask.jpg           (segmentation mask)
  ├── JPEGImages/        FLIR_XXXXX_PreviewData.jpeg
  │                      FLIR_XXXXX_RGB.jpg
  ├── align_train.txt    stems, e.g. "FLIR_00258_PreviewData"
  └── align_validation.txt
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import cv2

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_ROOT = Path("C:/wildfire_uav/data/flir_adas/aligned")
SAMPLE_SIZE  = 20   # images / XMLs checked for readability

ANNOTATED_DIR  = ALIGNED_ROOT / "AnnotatedImages"
ANNOTATIONS_DIR = ALIGNED_ROOT / "Annotations"
JPEG_DIR       = ALIGNED_ROOT / "JPEGImages"
TRAIN_TXT      = ALIGNED_ROOT / "align_train.txt"
VAL_TXT        = ALIGNED_ROOT / "align_validation.txt"

EXPECTED_THERMAL_SUFFIX = "_PreviewData.jpeg"
EXPECTED_RGB_SUFFIX     = "_RGB.jpg"
EXPECTED_XML_SUFFIX     = "_PreviewData.xml"
EXPECTED_MASK_SUFFIX    = "_mask.jpg"

# ── Helpers ───────────────────────────────────────────────────────────────────
ok_count   = 0
fail_count = 0
warnings   = []

def ok(msg):
    global ok_count
    ok_count += 1
    print(f"  [PASS] {msg}")

def fail(msg):
    global fail_count
    fail_count += 1
    print(f"  [FAIL] {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  [WARN] {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Check 1 · Directory & file existence ─────────────────────────────────────
section("CHECK 1 · Required paths")

for path in [ALIGNED_ROOT, ANNOTATED_DIR, ANNOTATIONS_DIR, JPEG_DIR, TRAIN_TXT, VAL_TXT]:
    if path.exists():
        ok(str(path))
    else:
        fail(f"Missing: {path}")

# ── Check 2 · File counts ─────────────────────────────────────────────────────
section("CHECK 2 · File counts")

thermal_files  = sorted(ANNOTATED_DIR.glob("*_PreviewData.jpeg"))
rgb_files      = sorted(ANNOTATED_DIR.glob("*_RGB.jpg"))
xml_files      = sorted(ANNOTATIONS_DIR.glob("*_PreviewData.xml"))
mask_files     = sorted(ANNOTATIONS_DIR.glob("*_mask.jpg"))
jpeg_thermal   = sorted(JPEG_DIR.glob("*_PreviewData.jpeg"))
jpeg_rgb       = sorted(JPEG_DIR.glob("*_RGB.jpg"))

counts = {
    "AnnotatedImages / thermal (.jpeg)": (len(thermal_files), 5142),
    "AnnotatedImages / RGB (.jpg)":      (len(rgb_files),     5142),
    "Annotations / XML":                 (len(xml_files),     5142),
    "Annotations / mask (.jpg)":         (len(mask_files),    5142),
    "JPEGImages / thermal (.jpeg)":      (len(jpeg_thermal),  5142),
    "JPEGImages / RGB (.jpg)":           (len(jpeg_rgb),      5142),
}

for label, (got, expected) in counts.items():
    if got == expected:
        ok(f"{label}: {got}")
    else:
        fail(f"{label}: got {got}, expected {expected}")

# ── Check 3 · Split files ─────────────────────────────────────────────────────
section("CHECK 3 · Train / Val split lists")

train_stems = [l.strip() for l in TRAIN_TXT.read_text().splitlines() if l.strip()]
val_stems   = [l.strip() for l in VAL_TXT.read_text().splitlines()   if l.strip()]

print(f"  align_train.txt      : {len(train_stems)} entries")
print(f"  align_validation.txt : {len(val_stems)} entries")
print(f"  Total                : {len(train_stems) + len(val_stems)}")

if len(train_stems) + len(val_stems) == len(thermal_files):
    ok("train + val count matches AnnotatedImages thermal count")
else:
    fail(f"train+val ({len(train_stems)+len(val_stems)}) != thermal images ({len(thermal_files)})")

overlap = set(train_stems) & set(val_stems)
if not overlap:
    ok("No overlap between train and val splits")
else:
    fail(f"Train/val overlap: {len(overlap)} stems — e.g. {list(overlap)[:3]}")

# ── Check 4 · Per-stem file pairing ──────────────────────────────────────────
section("CHECK 4 · Per-stem file pairing (all splits)")

missing_per_stem = defaultdict(list)

for stem in train_stems + val_stems:
    base = stem.replace("_PreviewData", "")   # e.g. "FLIR_00258"

    checks = {
        f"AnnotatedImages/{stem}.jpeg":          ANNOTATED_DIR  / f"{stem}.jpeg",
        f"AnnotatedImages/{base}_RGB.jpg":        ANNOTATED_DIR  / f"{base}_RGB.jpg",
        f"Annotations/{stem}.xml":               ANNOTATIONS_DIR / f"{stem}.xml",
        f"Annotations/{base}_mask.jpg":           ANNOTATIONS_DIR / f"{base}_mask.jpg",
        f"JPEGImages/{stem}.jpeg":               JPEG_DIR        / f"{stem}.jpeg",
        f"JPEGImages/{base}_RGB.jpg":            JPEG_DIR        / f"{base}_RGB.jpg",
    }
    for label, path in checks.items():
        if not path.exists():
            missing_per_stem[stem].append(label)

if not missing_per_stem:
    ok(f"All {len(train_stems)+len(val_stems)} stems have complete file sets")
else:
    for stem, missing in list(missing_per_stem.items())[:10]:
        fail(f"Stem {stem} missing: {missing}")
    if len(missing_per_stem) > 10:
        fail(f"... and {len(missing_per_stem)-10} more stems with missing files")

# ── Check 5 · Coverage (AnnotatedImages stems vs split lists) ─────────────────
section("CHECK 5 · AnnotatedImages coverage")

all_thermal_stems = {f.stem for f in thermal_files}          # e.g. "FLIR_00002_PreviewData"
all_split_stems   = set(train_stems) | set(val_stems)

unlisted = all_thermal_stems - all_split_stems
uncovered = all_split_stems - all_thermal_stems

if not unlisted:
    ok("All AnnotatedImages thermal files appear in split lists")
else:
    warn(f"{len(unlisted)} thermal images not listed in train/val — e.g. {list(unlisted)[:3]}")

if not uncovered:
    ok("All split-list stems have corresponding AnnotatedImages files")
else:
    fail(f"{len(uncovered)} split stems have no matching thermal image — e.g. {list(uncovered)[:3]}")

# ── Check 6 · XML validity (VOC format) ──────────────────────────────────────
section(f"CHECK 6 · XML validity (sample {SAMPLE_SIZE})")

step = max(1, len(xml_files) // SAMPLE_SIZE)
sample_xmls = xml_files[::step][:SAMPLE_SIZE]

xml_errors = []
class_counts = defaultdict(int)

for xml_path in sample_xmls:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        assert root.tag == "Annotation",  f"root tag = {root.tag}"
        size = root.find("size")
        assert size is not None,           "missing <size>"
        assert int(size.findtext("width"))  > 0, "width <= 0"
        assert int(size.findtext("height")) > 0, "height <= 0"
        for obj in root.findall("object"):
            cls = obj.findtext("name")
            class_counts[cls] += 1
            bb = obj.find("bndbox")
            assert bb is not None, "missing <bndbox>"
            xmin, ymin = int(bb.findtext("xmin")), int(bb.findtext("ymin"))
            xmax, ymax = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
            assert xmax > xmin and ymax > ymin, f"invalid bbox {xmin},{ymin},{xmax},{ymax}"
    except Exception as e:
        xml_errors.append(f"{xml_path.name}: {e}")

if not xml_errors:
    ok(f"All {len(sample_xmls)} sampled XMLs are valid VOC format")
    print(f"  Classes found in sample: {dict(class_counts)}")
else:
    for err in xml_errors:
        fail(err)

# ── Check 7 · Image readability ───────────────────────────────────────────────
section(f"CHECK 7 · Image readability (sample {SAMPLE_SIZE} thermal + {SAMPLE_SIZE} RGB)")

step_t = max(1, len(thermal_files) // SAMPLE_SIZE)
step_r = max(1, len(rgb_files)     // SAMPLE_SIZE)
sample_thermal = thermal_files[::step_t][:SAMPLE_SIZE]
sample_rgb     = rgb_files[::step_r][:SAMPLE_SIZE]

img_errors = []
sizes_seen  = set()

for img_path in sample_thermal + sample_rgb:
    img = cv2.imread(str(img_path))
    if img is None:
        img_errors.append(str(img_path.name))
    else:
        sizes_seen.add(img.shape)

if not img_errors:
    ok(f"All {len(sample_thermal)+len(sample_rgb)} sampled images are readable")
    print(f"  Shapes seen: {sizes_seen}")
else:
    for err in img_errors[:5]:
        fail(f"Unreadable: {err}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  PASS : {ok_count}")
print(f"  FAIL : {fail_count}")
if warnings:
    print(f"  WARN : {len(warnings)}")
    for w in warnings:
        print(f"    • {w}")

if fail_count == 0:
    print("\n  Dataset integrity verified. Ready for Phase 2-3 pipeline.")
else:
    print(f"\n  {fail_count} check(s) failed. Review FAIL items above before proceeding.")
