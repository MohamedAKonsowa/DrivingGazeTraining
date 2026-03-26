"""
Generate DR(eye)VE-style saliency heatmaps every N seconds from a driving video.

This script is model-runner focused:
- Input: video file
- Model: ONNX or TorchScript saliency model
- Output: one heatmap image every interval (default: 5 seconds)

Default video is set to:
  "First Person POV_ Driving.mp4"

Usage (ONNX):
python dreyeve_heatmaps_every_5s.py \
  --model-path /path/to/dreyeve_model.onnx \
  --model-type onnx

Usage (TorchScript):
python dreyeve_heatmaps_every_5s.py \
  --model-path /path/to/dreyeve_model.pt \
  --model-type torchscript
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def preprocess_bgr(frame_bgr, input_size):
    """Convert BGR frame to normalized tensor input [1, 3, H, W]."""
    h, w = input_size
    resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def normalize_map(saliency):
    saliency = saliency.astype(np.float32)
    saliency -= saliency.min()
    mx = saliency.max()
    if mx > 1e-8:
        saliency /= mx
    return saliency


def save_heatmap(saliency_2d, out_path):
    img = np.clip(saliency_2d * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), color)


class ONNXRunner:
    def __init__(self, model_path, input_name=None, output_name=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.output_name = output_name or self.session.get_outputs()[0].name

    def predict(self, x):
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        return self._to_2d(out)

    @staticmethod
    def _to_2d(out):
        # Expected shapes often include [1,1,H,W] or [1,H,W].
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0, 0]
        elif out.ndim == 3:
            out = out[0]
        elif out.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
        return out


class TorchScriptRunner:
    def __init__(self, model_path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, x):
        with self.torch.no_grad():
            tx = self.torch.from_numpy(x).to(self.device)
            out = self.model(tx)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.detach().cpu().numpy()
        return self._to_2d(out)

    @staticmethod
    def _to_2d(out):
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0, 0]
        elif out.ndim == 3:
            out = out[0]
        elif out.ndim != 2:
            raise RuntimeError(f"Unexpected TorchScript output shape: {out.shape}")
        return out


def build_runner(model_type, model_path, input_name=None, output_name=None):
    if model_type == "onnx":
        return ONNXRunner(model_path, input_name=input_name, output_name=output_name)
    if model_type == "torchscript":
        return TorchScriptRunner(model_path)
    raise ValueError(f"Unsupported model type: {model_type}")


def run(args):
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = build_runner(
        args.model_type,
        args.model_path,
        input_name=args.onnx_input_name,
        output_name=args.onnx_output_name,
    )

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    sample_times = np.arange(0.0, duration_sec, args.interval_sec)
    print(f"Video duration: {duration_sec:.2f}s | samples: {len(sample_times)}")

    for idx, t_sec in enumerate(sample_times, start=1):
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] Could not read frame at {t_sec:.2f}s")
            continue

        inp = preprocess_bgr(frame, (args.input_h, args.input_w))
        sal = runner.predict(inp)
        sal = normalize_map(sal)
        sal = cv2.resize(sal, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        out_name = f"heatmap_t{int(t_sec):05d}s.png"
        save_heatmap(sal, out_dir / out_name)
        print(f"[{idx}/{len(sample_times)}] saved {out_name}")

    cap.release()
    print(f"Done. Heatmaps saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate saliency heatmaps every 5 seconds.")
    p.add_argument(
        "--video-path",
        default="First Person POV_ Driving.mp4",
        help="Path to the driving video.",
    )
    p.add_argument("--model-path", required=True, help="Path to model file (.onnx or .pt).")
    p.add_argument(
        "--model-type",
        required=True,
        choices=["onnx", "torchscript"],
        help="Model runtime type.",
    )
    p.add_argument("--input-h", type=int, default=224, help="Model input height.")
    p.add_argument("--input-w", type=int, default=224, help="Model input width.")
    p.add_argument("--interval-sec", type=float, default=5.0, help="Sample interval in seconds.")
    p.add_argument("--output-dir", default="dreyeve_heatmaps_5s", help="Heatmap output directory.")
    p.add_argument("--onnx-input-name", default=None, help="Optional ONNX input tensor name.")
    p.add_argument("--onnx-output-name", default=None, help="Optional ONNX output tensor name.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
