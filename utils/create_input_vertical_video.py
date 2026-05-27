import os
import re
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

# --------- Configuration ---------
INPUT_ROOT = Path("/SSD1/database/DAVIS/2017/trainval/JPEGImages/480p")
OUTPUT_ROOT = Path("/SSD1/database/vertical")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_H = 96
TARGET_W = 256
FPS = 7
# ---------------------------------


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def resize_and_center_crop(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    orig_w, orig_h = img.size

    scale = max(target_h / orig_h, target_w / orig_w)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    img_cropped = img_resized.crop((left, top, right, bottom))
    return img_cropped


def process_sequence(seq_dir: Path):
    seq_name = seq_dir.name
    frame_paths = sorted(
        [p for p in seq_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda x: natural_key(x.name)
    )

    if not frame_paths:
        print(f"[WARN] No frames found in {seq_dir}")
        return

    out_video_path = OUTPUT_ROOT / f"{seq_name}.mp4"
    print(f"[INFO] Processing sequence: {seq_name} -> {out_video_path}")

    writer = imageio.get_writer(out_video_path, fps=FPS)

    try:
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")

            img_proc = resize_and_center_crop(img, TARGET_H, TARGET_W)

            img_rot = img_proc.transpose(Image.ROTATE_90)

            frame_np = np.array(img_rot)
            writer.append_data(frame_np)
    finally:
        writer.close()

    print(f"[OK] Saved: {out_video_path}")


if __name__ == "__main__":
    for seq_dir in sorted(INPUT_ROOT.iterdir()):
        if not seq_dir.is_dir():
            continue
        process_sequence(seq_dir)
