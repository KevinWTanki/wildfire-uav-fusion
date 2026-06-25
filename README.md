# Wildfire UAV — RGB-Thermal Fusion Detection

Personal research project exploring multi-modal image fusion for wildfire detection from UAV platforms.  
Developed independently as part of a university application portfolio.

---

## Motivation

Wildfires are increasingly difficult to detect early using visible-light cameras alone — smoke occlusion, night conditions, and complex terrain all degrade detection accuracy. Thermal infrared cameras capture heat signatures that persist through these conditions, but lose spatial context that RGB provides. This project investigates how to fuse both modalities to build a more robust detection system.

I chose this problem because it sits at the intersection of computer vision, sensor fusion, and real-world environmental impact — areas I want to pursue at university level.

---

## Project Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Dataset exploration · Alignment pipeline · Overlay verification | ✅ Week 1 complete |
| **2** | Cross-modal registration · ECC / phase-correlation alignment | 🔄 In progress |
| **3** | Fusion model training · YOLOv8 detection on fused frames | ⏳ Planned |
| **4** | UAV deployment simulation · Real-time inference | ⏳ Planned |

---

## Datasets

| Dataset | Content | Size | Use |
|---------|---------|------|-----|
| [LLVIP](https://github.com/bupt-ai-cz/LLVIP) | 15,488 aligned visible + infrared pairs (night, pedestrian) | ~4 GB | Alignment & fusion baseline |
| [FLIR ADAS v2](https://www.flir.com/oem/adas/adas-dataset-form/) | 5,142 aligned RGB-thermal pairs · Pascal VOC annotations | ~2 GB | Detection ground truth |

Datasets are **not** tracked in this repository (file size). See [`docs/setup.md`](docs/setup.md) for download instructions.

---

## Key Decisions & Findings

Technical choices I made and why — full reasoning in [`docs/decisions/`](docs/decisions/).

- **Why ORB homography fails on pre-aligned RGB-IR data** → [Decision 001](docs/decisions/001-orb-homography-failure.md)
- **Why LLVIP needs Identity H, not estimated H** → [Decision 001](docs/decisions/001-orb-homography-failure.md)

---

## Repository Structure

```
wildfire-uav-fusion/
├── week1_pipeline.py       # Phase 1: load → warpPerspective → overlay
├── verify_dataset.py       # FLIR ADAS dataset integrity checker
├── output/                 # Pipeline visualisation outputs (gitignored in bulk)
├── docs/
│   ├── setup.md            # Environment & dataset setup guide
│   ├── decisions/          # Key technical decision records
│   └── journal/            # Weekly progress notes
└── .gitignore
```

---

## Environment

```bash
conda create -n wildfire_uav python=3.11
conda activate wildfire_uav
pip install opencv-python==4.9.0.80 numpy==1.24.0 matplotlib==3.7.0 \
            tqdm==4.65.0 tifffile==2023.4.12 ultralytics==8.0.196 scipy==1.11.0
```

---

## Running the Pipeline

```bash
# Verify FLIR ADAS dataset integrity
python verify_dataset.py

# Run Week 1 fusion pipeline (requires LLVIP dataset)
python week1_pipeline.py
```

---

## Academic Integrity Note

All code in this repository is written by me unless explicitly attributed.  
Where I used AI assistance (Claude Code) for debugging or explanation, the relevant commit messages note this.  
Commit history reflects the actual development timeline — decisions, dead ends, and corrections included.
