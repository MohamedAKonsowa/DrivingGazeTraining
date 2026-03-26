# Beginner DR(eye)VE Setup

You found the correct repo. The main challenge is that the original codebase is old:

- Python code was built for **Keras 1 + Theano**
- Some scripts are **Python 2 style**
- The original prediction pipeline expects extra inputs (optical flow + semantic segmentation + temporal stack), not just raw video frames

Because of this, the fastest practical route for your own video is:

1. Use a **converted/exported model** (`.onnx` or TorchScript `.pt`)
2. Run `dreyeve_heatmaps_every_5s.py` on your video

---

## 1) Install deps

If you use ONNX:

```bash
pip install opencv-python numpy onnxruntime
```

If you use TorchScript:

```bash
pip install opencv-python numpy torch
```

---

## 2) Generate one heatmap every 5 seconds

Your video is already the default:
- `First Person POV_ Driving.mp4`

Run:

```bash
python dreyeve_heatmaps_every_5s.py \
  --model-path "/path/to/your_model.onnx" \
  --model-type onnx \
  --interval-sec 5
```

or

```bash
python dreyeve_heatmaps_every_5s.py \
  --model-path "/path/to/your_model.pt" \
  --model-type torchscript \
  --interval-sec 5
```

Outputs are saved to:
- `dreyeve_heatmaps_5s/`

---

## 3) Compare with user gaze (optional)

Use your comparison script:

```bash
python drive_heatmap_compare.py \
  --drive-video "First Person POV_ Driving.mp4" \
  --reference-heatmap-dir "dreyeve_heatmaps_5s" \
  --interval-sec 5 \
  --output-dir "outputs_compare"
```

---

## Important note about the original repo

The original DR(eye)VE repo is useful for research reference and original training/evaluation code, but running it directly for arbitrary modern videos is usually non-trivial due to legacy dependencies and dataset-specific preprocessing.

