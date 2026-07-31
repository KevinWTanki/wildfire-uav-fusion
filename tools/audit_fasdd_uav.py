"""
Audit FASDD_UAV before training against it.

Measures, rather than assumes, five things the training pipeline depends on:
  1a. the actual filename-prefix vocabulary (the landing page's
      Fire/Smoke/FireAndSmoke/NeitherFireNorSmoke strings are not what's on
      disk)
  1b. the class-id <-> {fire, smoke} mapping, exhaustively (every label file
      per single-class prefix, not a sample) -- FASDD's own docs don't state
      which integer id is which
  1c. per-split, per-class instance counts, cross-checked against the
      published totals (36,308 fire / 17,222 smoke) -- the dataset is at V9
      and secondary sources have cited stale figures before
  1d. how negative (NeitherFireNorSmoke) images are represented -- present-
      but-empty label files vs. missing files entirely, which Ultralytics
      treats differently in its cache stats (background vs. missing-label)
  1e. split leakage -- exact duplicate filenames and exact duplicate file
      content (MD5) across train/val/test

Findings as of the V9 download used for detection/train_yolov8.py are
recorded in the Commit 1 message that added this script; re-run this module
directly if the dataset is re-downloaded or changes.

Usage:
    python tools/audit_fasdd_uav.py
"""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "FASDD_UAV"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "annotations" / "YOLO_UAV" / "labels"
SPLIT_DIR = ROOT / "annotations" / "YOLO_UAV"
SPLITS = ("train", "val", "test")

PUBLISHED_FIRE_INSTANCES = 36308
PUBLISHED_SMOKE_INSTANCES = 17222


def _stem_of(line: str) -> str:
    return Path(line.strip().split("/")[-1]).stem


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def audit_prefixes(all_images: list[Path]) -> dict[str, int]:
    """1a. Filename prefix vocabulary and counts."""
    _section("1a. Filename prefix vocabulary")
    prefixes = Counter()
    for f in all_images:
        # prefixes are the alphabetic run before "_UAV<digits>"
        stem = f.stem
        idx = stem.rfind("_UAV")
        prefixes[stem[:idx]] += 1 if idx > 0 else 0
    for prefix, count in prefixes.most_common():
        print(f"  {prefix:25s} {count:6d}")
    print(f"  TOTAL images on disk: {len(all_images)}")
    return dict(prefixes)


def audit_class_id_mapping(all_images: list[Path], prefixes: list[str]) -> None:
    """1b. Exhaustive (not sampled) class-id evidence per prefix."""
    _section("1b. Class-id mapping (exhaustive per prefix)")
    for prefix in prefixes:
        ids = Counter()
        n_images = 0
        for f in all_images:
            if not f.stem.startswith(prefix + "_UAV"):
                continue
            n_images += 1
            text = (LABELS_DIR / f"{f.stem}.txt").read_text().strip()
            for line in text.splitlines():
                if line.strip():
                    ids[int(line.split()[0])] += 1
        print(f"  {prefix:20s} n_images={n_images:6d} class_id_counts={dict(ids)}")


def audit_instance_counts() -> None:
    """1c. Per-split, per-class instance counts vs. the published totals."""
    _section("1c. Instance counts per class per split")
    overall = Counter()
    for split in SPLITS:
        stems = [_stem_of(l) for l in (SPLIT_DIR / f"{split}.txt").read_text().strip().splitlines() if l.strip()]
        counts = Counter()
        empty, nonempty = 0, 0
        for stem in stems:
            text = (LABELS_DIR / f"{stem}.txt").read_text().strip()
            if not text:
                empty += 1
                continue
            nonempty += 1
            for line in text.splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
        overall.update(counts)
        print(f"  {split:5s}: n={len(stems):6d}  class_counts={dict(counts)}  "
              f"empty_label_files={empty}  nonempty_label_files={nonempty}")
    print(f"  OVERALL: {dict(overall)}")
    print(f"  Published: fire={PUBLISHED_FIRE_INSTANCES}  smoke={PUBLISHED_SMOKE_INSTANCES}")
    match = (overall.get(0) == PUBLISHED_FIRE_INSTANCES and overall.get(1) == PUBLISHED_SMOKE_INSTANCES)
    print(f"  Matches published totals: {match}")


def audit_negatives(all_images: list[Path], prefixes: list[str]) -> None:
    """1d. How negative images are represented -- empty file vs. no file."""
    _section("1d. Negative-sample representation")
    for prefix in prefixes:
        total = empty = missing = nonempty = 0
        for f in all_images:
            if not f.stem.startswith(prefix + "_UAV"):
                continue
            total += 1
            label_path = LABELS_DIR / f"{f.stem}.txt"
            if not label_path.exists():
                missing += 1
            elif not label_path.read_text().strip():
                empty += 1
            else:
                nonempty += 1
        print(f"  {prefix:20s} total={total:6d} empty_label={empty:6d} "
              f"missing_label={missing:6d} nonempty_label={nonempty:6d}")


def audit_split_leakage(all_images: list[Path]) -> None:
    """1e. Exact duplicate filenames and exact duplicate file content across splits."""
    _section("1e. Split leakage")
    split_stems = {}
    for split in SPLITS:
        split_stems[split] = [_stem_of(l) for l in
                               (SPLIT_DIR / f"{split}.txt").read_text().strip().splitlines() if l.strip()]

    stem_to_splits = defaultdict(set)
    for split, stems in split_stems.items():
        for stem in stems:
            stem_to_splits[stem].add(split)
    dup_filenames = {s: sp for s, sp in stem_to_splits.items() if len(sp) > 1}
    print(f"  Filenames listed in >1 split: {len(dup_filenames)}")

    listed = set().union(*split_stems.values())
    on_disk = {f.stem for f in all_images}
    print(f"  Listed but not on disk: {len(listed - on_disk)}")
    print(f"  On disk but not listed in any split: {len(on_disk - listed)}")

    print("  Hashing image content for exact-duplicate check (may take a while)...")
    stem_to_path = {f.stem: f for f in all_images}
    hash_to_locations = defaultdict(list)
    for split, stems in split_stems.items():
        for stem in stems:
            path = stem_to_path.get(stem)
            if path is None:
                continue
            h = hashlib.md5()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            hash_to_locations[h.hexdigest()].append((split, stem))

    dup_groups = {h: locs for h, locs in hash_to_locations.items() if len(locs) > 1}
    cross_split = {h: locs for h, locs in dup_groups.items() if len({s for s, _ in locs}) > 1}
    print(f"  Total duplicate-content groups: {len(dup_groups)}")
    print(f"  Duplicate-content groups spanning >1 split (leakage): {len(cross_split)}")
    for h, locs in cross_split.items():
        print(f"    {h[:12]}... {locs}")


def main() -> None:
    all_images = sorted(IMAGES_DIR.iterdir())
    prefixes = audit_prefixes(all_images)
    audit_class_id_mapping(all_images, list(prefixes))
    audit_instance_counts()
    audit_negatives(all_images, list(prefixes))
    audit_split_leakage(all_images)
    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
