"""
Generate SCOUT saliency heatmaps every N seconds for a driving video.

Expected layout:
- SCOUT repo cloned at: ./SCOUT
- Weights at: ./SCOUT_DReyeVE.pt
- Video at: ./First Person POV_ Driving.mp4

Example:
python scout_heatmaps_every_5s.py
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch


def preprocess_frame_bgr(frame_bgr, img_h, img_w):
    resized = cv2.resize(frame_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw)


def normalize_map(sal):
    sal = sal.astype(np.float32)
    sal -= sal.min()
    mx = sal.max()
    if mx > 1e-8:
        sal /= mx
    return sal


def make_dummy_task(batch_size, clip_size, device):
    # This checkpoint uses local_context task attributes:
    # ['dist_to_inters', 'inters_priority', 'next_action'].
    return {
        "dist_to_inters": torch.ones((batch_size, 1, clip_size), dtype=torch.float32, device=device) * 10000.0,
        "next_action": torch.zeros((batch_size, 1), dtype=torch.long, device=device),
        "inters_priority": torch.zeros((batch_size, 1), dtype=torch.long, device=device),
    }


def load_scout_model(scout_repo, weights_path, device, img_size=224, clip_size=16):
    scout_repo = Path(scout_repo).resolve()
    if not scout_repo.exists():
        raise FileNotFoundError(f"SCOUT repo not found: {scout_repo}")
    sys.path.insert(0, str(scout_repo))

    # Import from SCOUT repository.
    from model import SCOUT_task  # pylint: disable=import-error

    model = SCOUT_task(
        num_encoder_layers=4,
        use_task=True,
        task_attributes=["local_context"],
        train_backbone=False,
        pretrained_backbone=False,
        transformer_params={
            "add_and_norm": True,
            "fuse_idx": [1, 2],
            "num_att_heads": [2, 2, 2],
        },
        img_size=[img_size, img_size],
        clip_size=clip_size,
    )

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Handle checkpoints saved from DataParallel.
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


def run(args):
    video_path = Path(args.video).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_scout_model(
        scout_repo=args.scout_repo,
        weights_path=str(weights_path),
        device=device,
        img_size=args.img_size,
        clip_size=args.clip_size,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    print(f"Video: {video_path.name} | fps={fps:.2f} | duration={duration_sec:.2f}s")
    print(f"Saving one heatmap every {args.interval_sec:.2f}s to {output_dir}")

    frame_buffer = deque(maxlen=args.clip_size)
    frame_idx = 0
    next_save_time = 0.0
    saved_count = 0
    last_pred = None
    frame_h = frame_w = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        frame_tensor = preprocess_frame_bgr(frame, args.img_size, args.img_size)
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) == args.clip_size:
            clip = torch.stack(list(frame_buffer), dim=0)  # [T, C, H, W]
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]
            task = make_dummy_task(batch_size=1, clip_size=args.clip_size, device=device)

            with torch.no_grad():
                pred = model(clip, task)  # [1, H, W]
                pred = pred.squeeze(0).detach().cpu().numpy()
                pred = normalize_map(pred)
                pred = cv2.resize(pred, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
                last_pred = pred

        t_sec = frame_idx / fps
        if last_pred is not None and t_sec >= next_save_time:
            out_name = f"scout_heatmap_t{int(next_save_time):05d}s.png"
            out_path = output_dir / out_name
            heat_u8 = np.clip(last_pred * 255.0, 0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            cv2.imwrite(str(out_path), heat_color)

            print(f"Saved {out_name}")
            saved_count += 1
            next_save_time += args.interval_sec

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} heatmaps.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate SCOUT heatmaps every N seconds.")
    p.add_argument("--video", default="First Person POV_ Driving.mp4", help="Input video path.")
    p.add_argument("--weights", default="SCOUT_DReyeVE.pt", help="SCOUT checkpoint path.")
    p.add_argument("--scout-repo", default="SCOUT", help="Path to local SCOUT repository.")
    p.add_argument("--output-dir", default="scout_heatmaps_5s", help="Output directory.")
    p.add_argument("--interval-sec", type=float, default=5.0, help="Save interval in seconds.")
    p.add_argument("--img-size", type=int, default=224, help="Model input size (square).")
    p.add_argument("--clip-size", type=int, default=16, help="Temporal clip length.")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
