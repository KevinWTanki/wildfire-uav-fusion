# Wildfire Detection UAV — RGB/Thermal Fusion

A dual-light (visible + long-wave infrared) detection pipeline for spotting wildfires
early from a UAV platform, with the inference running on-board rather than streamed to
a ground station.

---

## The problem

Early wildfire detection from the air is hard for a single sensor to do well.

A visible-light camera sees smoke and flame in context — it can tell a fire front from
a road — but loses the target under canopy occlusion, in low light, and whenever
smoke and haze look alike. A thermal camera sees the heat signature through most of
that, but at 160×120 it has almost no spatial context: a hot patch of asphalt and a
hot patch of burning brush look similar.

The two failure modes are largely complementary. Fog has no thermal signature; sun-heated
asphalt has no flame texture. That is the basis for fusing them at the pixel level rather
than running two independent detectors, and it is what this project is built around.

---

## Approach

The system is organised as a **core detection path**, with several deeper directions
identified but not scheduled.

**Core path**

```
RGB frame  ──► YOLOv8n detection ──┐
                                   ├──► IoU fusion ──► fire probability ──► alert decision
Thermal frame ──► registration ────┤
                  └► hotspot extraction ──► growth rate (dA/dt)
```

1. **Registration** — a homography maps thermal pixel coordinates into the RGB frame so
   the two streams can be compared spatially.
2. **Thermal analysis** — adaptive thresholding isolates hotspots; connected-component
   analysis extracts temperature, area and shape; a Kalman filter tracks area growth rate
   across frames.
3. **Fusion** — detections and hotspots are matched by spatial overlap, and a weighted
   score combines detector confidence, hotspot temperature and growth rate.

The growth rate is the part that distinguishes an expanding fire from a static heat
source, which a single-frame temperature threshold cannot do.

**Identified directions, not scheduled:** detector quality (class imbalance, local-domain
adaptation), a geolocation error budget converting detections to GPS coordinates, and
hardware calibration once the physical sensors are mounted. Each has a concrete entry
point recorded in the decision notes; none is in progress.

---

## Current status

| Component | State |
|---|---|
| Dataset validation and audit tooling | Working |
| Registration + overlay pipeline | Working on pre-aligned datasets, where the correct homography is Identity |
| Homography estimation | Implemented — cross-modal feature matching found unusable, see Decision 001 |
| Thermal hotspot extraction | Implemented |
| Growth rate tracker | Implemented, exercised on synthetic sequences |
| Detector | Trained on FASDD_UAV — see baseline below |
| IoU fusion | Implemented — not yet driven by the trained detector |
| Hardware calibration | Blocked on hardware; method and calibration target design settled in advance |

Two things about the fusion stage are worth stating because neither is visible from
reading the code. It still generates detection boxes synthetically rather than calling
the trained detector — wiring that in is the immediate next step. And the terrain risk
prior in the fusion score is fixed at a neutral 0.5, standing in for a terrain and
vegetation risk map that does not exist yet.

The growth rate is a real filter output rather than a constant, but it has only been
run on synthetic hotspot sequences. A continuous thermal stream from a real sensor is
what it is waiting on, and absolute calibration of the growth rate against radiometric
ground truth belongs to that same stage.

---

## Detection baseline

The detector exists to replace the simulated boxes in the fusion stage so the pipeline can
run end to end on real detections. Three epochs on CPU is the whole scope of the run —
detector quality is one of the identified, unscheduled directions, not something this
model attempts.

YOLOv8n, first 10 backbone layers frozen from COCO-pretrained weights, 640 px input,
trained and evaluated on FASDD_UAV's own splits.

| | AP@50 |
|---|---|
| flame | 0.621 |
| smoke | 0.853 |
| overall mAP@50 | 0.737 |

On the negative images in the test split: 33 false positive boxes across 1,997 images,
0.017 per image at confidence 0.25.

Classes are reported separately because FASDD_UAV contains roughly twice as many flame
instances as smoke, so an overall figure is carried by the majority class. That matters
here — smoke is the minority class and scores 23 points higher than flame, which is the
opposite of what the instance counts predicted. Notes on that in
[the journal entry](docs/journal/2026-07-30-detector-integration-baseline.md).

---

## Key decisions

Technical choices and the reasoning behind them, including the ones that turned out to be
wrong. Full records in [`docs/decisions/`](docs/decisions/).

- **[Decision 001](docs/decisions/001-orb-homography-failure.md)** — Cross-modal ORB
  feature matching cannot produce a usable homography. Descriptors computed on RGB
  intensity and on thermal radiance describe different physical quantities, so the matcher
  returns confident matches on noise. The failure is silent: a degenerate homography still
  warps without error, producing an output that looks plausible. Detected by measuring the
  fraction of valid output pixels, not by the code raising anything.
- **[Decision 002](docs/decisions/002-yolov8-class-definition.md)** — The detector uses two
  classes, `flame` and `smoke`. An earlier three-class scheme included a
  `normal_vegetation` class, which does not work in a detection framework: background is
  already implicit in the loss, and the class has no definable bounding box. False
  positives are handled with negative-sample images instead.
- **[Decision 003](docs/decisions/003-training-data-gap.md)** — Survey of available
  labelled wildfire data, and what it does not cover. Includes a measured case of
  cross-domain collapse (0.93 → 0.031 precision when a smoke detector trained in one
  region is evaluated in another), which is why locally-representative data matters more
  than raw volume here. An amendment records a dataset the first survey missed, and what
  that changed about the conclusion.

---

## Repository structure

```
wildfire_uav/
├── calibration/
│   ├── homography_estimation.py     # H estimation
│   ├── registration_metrics.py      # reprojection and edge-alignment error
│   └── results/                     # saved H matrices and validation reports
├── detection/                       # RGB detection path
│   ├── dataset_prep.py              # split-list preparation
│   ├── fasdd_uav.yaml               # dataset config: class ids, licence, citation
│   ├── train_yolov8.py              # training
│   ├── evaluate.py                  # per-class AP, negative-set false positives
│   └── inference.py                 # RGB inference wrapper
├── thermal/                         # thermal path
│   ├── hotspot_extraction.py        # adaptive threshold + connected components
│   └── growth_rate_tracker.py       # Kalman filter over hotspot area
├── fusion/                          # consumes both paths
│   └── iou_fusion.py                # spatial matching + weighted scoring
├── pipelines/
│   └── registration_overlay.py      # load → warp → overlay, end to end
├── tools/
│   ├── validate_dataset.py          # dataset integrity and pairing checks
│   └── audit_fasdd_uav.py           # class mapping, splits, negative coverage
├── experiments/
│   └── homography_reference_example.py   # retained demo, not on the production path
├── docs/
│   ├── setup.md                     # environment and dataset setup
│   ├── decisions/                   # technical decision records
│   └── journal/                     # progress notes
├── data/                            # datasets (not tracked — see docs/setup.md)
└── results/                         # evaluation reports
```

`detection/` and `thermal/` are the two parallel perception paths; `fusion/` consumes
both. Modules are named for their function rather than their position in a sequence, so
inserting a stage does not force a round of renames.

---

## Datasets

None are tracked in this repository. See [`docs/setup.md`](docs/setup.md) for download
instructions.

| Dataset | Content | Role | Licence |
|---|---|---|---|
| [FASDD](https://doi.org/10.57760/sciencedb.j00104.00103) (UAV subset) | 25,097 aerial images, `fire` and `smoke` boxes, built-in negatives | Detector training | CC BY-SA 4.0 |
| [LLVIP](https://github.com/bupt-ai-cz/LLVIP) | 15,488 strictly aligned visible/infrared pairs | Registration and fusion development | — |
| [FLIR ADAS v2](https://www.flir.com/oem/adas/adas-dataset-form/) | 5,142 aligned RGB-thermal pairs | Registration validation | — |
| [Boreal Forest Fire](https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9) | 4,954 UAV images, smoke boxes | Smoke supplement | CC BY 4.0 |
| [FLAME 3](https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management) | Radiometric thermal, per-pixel °C | Temperature threshold validation | — |

FASDD_UAV and LLVIP are the two currently in use; the rest belong to work not yet started.

FASDD requires citation as a condition of use: Wang, M., Yue, P., Jiang, L., Yu, D., Tuo,
T., & Li, J. (2025). An open flame and smoke detection dataset for deep learning in remote
sensing based fire detection. *Geo-spatial Information Science*, 28(2), 511–526.

---

## Target hardware

The software is written against this configuration. None of it is required to run the
pipeline on datasets.

| Component | Model |
|---|---|
| Thermal camera | FLIR Lepton 3.5 — 160×120, radiometric, via PureThermal Mini Pro |
| RGB camera | Arducam IMX477, 6 mm CS-mount lens |
| On-board compute | NVIDIA Jetson Nano 4 GB |
| Flight controller | Pixhawk 4 Mini, PX4 |
| Airframe | DJI F450 |

The Jetson Nano's inference budget is the reason the detector is YOLOv8n rather than a
larger variant, and the reason throughput is treated as a hard acceptance criterion rather
than something to optimise later.

---

## Environment

```bash
conda create -n wildfire_uav python=3.11
conda activate wildfire_uav
pip install opencv-python==4.9.0.80 numpy==1.24.0 matplotlib==3.7.0 \
            tqdm==4.65.0 tifffile==2023.4.12 ultralytics==8.0.196 scipy==1.11.0
```

Python 3.10 or 3.11. Newer releases are avoided because the machine-learning
dependencies lag behind them by months; the Jetson deployment target is separately
constrained to 3.8 by JetPack's prebuilt CUDA wheels.

---

## Running

```bash
# Dataset integrity and pairing checks
python tools/validate_dataset.py

# FASDD class mapping, split integrity, negative-sample coverage
python tools/audit_fasdd_uav.py

# Registration overlay, end to end (requires LLVIP)
python pipelines/registration_overlay.py

# Detector training and evaluation (requires FASDD_UAV)
python detection/dataset_prep.py
python detection/train_yolov8.py
python detection/evaluate.py
```
