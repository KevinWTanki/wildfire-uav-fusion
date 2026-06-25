# Environment & Dataset Setup

## Conda environment

```bash
conda create -n wildfire_uav python=3.11 -y
conda activate wildfire_uav

pip install opencv-python==4.9.0.80 \
            numpy==1.24.0 \
            matplotlib==3.7.0 \
            tqdm==4.65.0 \
            tifffile==2023.4.12 \
            ultralytics==8.0.196 \
            scipy==1.11.0
```

## Dataset download

### LLVIP (15,488 visible + infrared pairs)
- Homepage: https://bupt-ai-cz.github.io/LLVIP/
- Google Drive: https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view
- Place extracted folder at: `LLVIP/LLVIP/`

```
LLVIP/LLVIP/
├── visible/
│   ├── train/   (12,025 images)
│   └── test/
├── infrared/
│   ├── train/
│   └── test/
└── Annotations/
```

### FLIR ADAS v2 aligned subset
- Source: https://www.flir.com/oem/adas/adas-dataset-form/
- Place at: `data/flir_adas/aligned/`

```
data/flir_adas/aligned/
├── AnnotatedImages/   (5,142 thermal + 5,142 RGB)
├── Annotations/       (5,142 XML + 5,142 mask)
├── JPEGImages/
├── align_train.txt    (4,129 entries)
└── align_validation.txt (1,013 entries)
```

Run `python verify_dataset.py` to confirm dataset integrity before use.
