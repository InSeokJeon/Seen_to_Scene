import re
import math
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

# --------- Configuration ---------
INPUT_ROOT = Path("/SSD1/database/DAVIS/2017/trainval/JPEGImages/480p")
OUTPUT_ROOT = Path("/SSD1/database/challenge_vo_input")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_H = 256
TARGET_W = 256
FPS = 7

RATIOS = [0.75, 0.80, 0.85, 0.90]
SNAP_TO_8 = 16
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
    
    return img_resized.crop((left, top, left + target_w, top + target_h))


def keep_width_from_ratio(W: int, r: float, snap_to: int = 16) -> int:
    
    keep = (1.0 - r) * W
    keep_w = int(round(keep))
    keep_w = max(1, min(W, keep_w))

    keep_w = int(math.ceil(keep_w / snap_to) * snap_to)
    keep_w = max(snap_to, min(W, keep_w))

    return keep_w



def center_crop_width(img_rgb: Image.Image, keep_w: int) -> Image.Image:
    
    W, H = img_rgb.size
    assert (W, H) == (TARGET_W, TARGET_H)

    left = (W - keep_w) // 2
    right = left + keep_w
    
    return img_rgb.crop((left, 0, right, H))


def process_sequence(seq_dir: Path):
    seq_name = seq_dir.name
    frame_paths = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
                         key=lambda x: natural_key(x.name))
    
    if not frame_paths:
        print(f"[WARN] No frames found in {seq_dir}")
        return

    for r in RATIOS:
        keep_w = keep_width_from_ratio(TARGET_W, r, snap_to=SNAP_TO_8)

        out_dir = OUTPUT_ROOT / f"ratio_{r:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_video = out_dir / f"{seq_name}.mp4"
        writer = imageio.get_writer(out_video, fps=FPS)

        try:
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img256 = resize_and_center_crop(img, TARGET_H, TARGET_W)

                cropped = center_crop_width(img256, keep_w)
                writer.append_data(np.array(cropped))
        finally:
            writer.close()

        actual_r = 1.0 - (keep_w / TARGET_W)
        print(f"[OK] {seq_name} target r={r:.2f} -> keep_w={keep_w} (actual r={actual_r:.4f}) saved {out_video}")


if __name__ == "__main__":
    for seq_dir in sorted(INPUT_ROOT.iterdir()):
        if seq_dir.is_dir():
            process_sequence(seq_dir)
