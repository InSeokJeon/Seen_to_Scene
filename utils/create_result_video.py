import os
import shutil
from pathlib import Path

src_dir = Path("/SSD1/cvpr2026/Final_Final/mask_025_videos/no_ref/comp")
dst_dir = Path("/SSD1/cvpr2026/Final_Final/result_025/no_ref")

dst_dir.mkdir(parents = True, exist_ok = True)

for mp4_path in src_dir.glob("*_comp.mp4"):
    dst_path = dst_dir / mp4_path.name
    print(f"Copy: {mp4_path} -> {dst_path}")
    shutil.copy2(mp4_path, dst_path)

print("Done.")
