# Driving Gaze Training

A practical toolkit for comparing a user's gaze behavior against model-based driving saliency maps.

This project includes:
- Webcam eye-tracking demos in Python
- Browser-based tracking with WebGazer
- SCOUT/SCOUT+ saliency heatmap generation every 5 seconds
- Overlay comparison between expected (SCOUT) and user gaze heatmaps

## Project Structure

- `webgazer_demo.html` - Browser demo for calibration, 30s recording, and in-page overlay preview
- `webgazer_heatmaps_first30.py` - Convert WebGazer JSON samples to per-5s heatmaps
- `scout_heatmaps_every_5s.py` - Generate SCOUT or SCOUT+ expected heatmaps from video
- `overlay_heatmaps.py` - Overlay reference vs user heatmaps by timestamp
- `quick_gaze_debug.py` - Fast webcam gaze-direction debug tool
- `eyetrax_heatmaps_first30.py` - EyeTrax-based 30s user heatmap generation
- `fullscreen_gaze_heatmap.py` - Fullscreen player with live gaze heatmap

## Requirements

- Python 3.10+ recommended
- macOS tested (camera permission required)
- A local SCOUT repository in `./SCOUT`
- One of these weights in project root:
  - `SCOUT_DReyeVE.pt`
  - `SCOUT+_DReyeVE.pt`
- Driving video:
  - `First Person POV_ Driving.mp4`

## Quick Start

Install core dependencies:

```bash
python3 -m pip install opencv-python mediapipe numpy torch einops
```

## 1) Generate Expected Heatmaps (SCOUT / SCOUT+)

SCOUT:

```bash
python3 scout_heatmaps_every_5s.py \
  --video "First Person POV_ Driving.mp4" \
  --weights "SCOUT_DReyeVE.pt" \
  --scout-repo "SCOUT" \
  --output-dir "scout_heatmaps_5s" \
  --interval-sec 5
```

SCOUT+:

```bash
python3 scout_heatmaps_every_5s.py \
  --video "First Person POV_ Driving.mp4" \
  --weights "SCOUT+_DReyeVE.pt" \
  --scout-repo "SCOUT" \
  --output-dir "scoutplus_heatmaps_5s" \
  --interval-sec 5
```

## 2) Record User Gaze (WebGazer)

Start local server:

```bash
python3 -m http.server 8080
```

Open:
- `http://localhost:8080/webgazer_demo.html`

Flow:
1. Click **Start WebGazer**
2. Click **Calibrate**
3. Click **Record First 30s**
4. (Optional) Click **Export Samples JSON** to save `webgazer_first30_samples.json`

## 3) Convert WebGazer Samples to 5s Heatmaps

```bash
python3 webgazer_heatmaps_first30.py \
  --samples-json webgazer_first30_samples.json \
  --output-dir webgazer_heatmaps_first30
```

## 4) Overlay Expected vs User Heatmaps

SCOUT vs WebGazer:

```bash
python3 overlay_heatmaps.py \
  --reference-dir scout_heatmaps_5s \
  --user-dir webgazer_heatmaps_first30 \
  --user-glob "webgazer_user_heatmap_t*s.png" \
  --output-dir overlay_scout_vs_webgazer_first30
```

SCOUT+ vs WebGazer:

```bash
python3 overlay_heatmaps.py \
  --reference-dir scoutplus_heatmaps_5s \
  --user-dir webgazer_heatmaps_first30 \
  --user-glob "webgazer_user_heatmap_t*s.png" \
  --output-dir overlay_scoutplus_vs_webgazer_first30
```

Output preview:
- `overlay_*/overlay_preview_grid.png`

Color key:
- Blue = model/reference heatmap
- Red = user heatmap
- Magenta = overlap

## Notes

- If webcam fails in browser, verify macOS camera permissions for your browser.
- If page changes do not appear, hard refresh (`Cmd+Shift+R`).
- If SCOUT model load fails, ensure `einops` is installed and SCOUT repo path is correct.
