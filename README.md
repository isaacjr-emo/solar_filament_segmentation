# Solar Filament Segmentation & Tracking — Streamlit App

Interactive web application for pixel-level instance segmentation **and multi-frame tracking** of solar filaments in H-alpha chromospheric images.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Modes

### 1. Single Image
- Upload one H-alpha image
- Adjust detection parameters
- View overlay, masks, filament table, Dice / Panoptic Quality
- Download binary mask, labeled mask, colour overlay

### 2. Time Series Tracking
- Upload an ordered sequence of images (multi-select)
- Segment every frame, then associate instances across time
- **Overview** — track summary table + lifetime histogram
- **Frame Explorer** — scrub frames with persistent track colours
- **Track Detail** — area evolution plot + appearance thumbnails
- **Export** — CSV of track summaries and full observation tables

### Tracking parameters
- Max centroid distance (px)
- Max frames a track may be “lost”
- Differential-rotation correction (on/off)
- Cadence (minutes between frames)

## Project structure

```
├── app.py                      # Streamlit frontend (both modes)
├── filament_segmentation.py    # Classical CV segmentation pipeline
├── filament_tracking.py        # Multi-object tracker (IoU + Hungarian)
├── requirements.txt
└── README.md
```

## Pipeline overview

**Segmentation:** Disk extraction → Limb darkening → Local contrast → Adaptive threshold → Morphology → Instance labelling

**Tracking:** Per-frame instances → cost matrix (centroid + IoU, optional differential rotation) → Hungarian assignment → track management
