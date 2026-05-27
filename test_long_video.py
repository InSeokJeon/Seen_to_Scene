import os
import glob
import csv
import subprocess
from pathlib import Path

import test

DATA_ROOT = "data path" 
PRETRAINED_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

BASE_OUTPUT_DIR = "./output_pathg"
BASE_VIDEO_DIR = "./input_video_path"

WIDTH = 256
HEIGHT = 256
PAD_DIRECTION = "horizontal"

WINDOW_SIZE = 25
STRIDE = 5
NUM_INFERENCE_STEPS = 25
MIN_GUIDANCE = 1.0
MAX_GUIDANCE = 3.0
FPS = 7
MOTION_BUCKET_ID = 127
NOISE_AUG = 0.02
SEED = 2026
DECODE_CHUNK_SIZE = 8
FRAME_STRIDE = 1
MAX_FRAMES = None

USE_BLUR_COMP = True
SAVE_MP4 = True

REF_WINDOWS = [4]
# ----------------------


def main():
    video_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.mp4")))

    if not video_files:
        print(f"[WARN] No videos found in {DATA_ROOT}")
        return

    ref_stats = {rw: [] for rw in REF_WINDOWS if rw is not None}

    csv_path = "ref_stats.csv"
    csv_f = open(csv_path, "w", newline = "")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["seq_name", "num_frames", "ref_window", "num_refs"])

    for ref_w in REF_WINDOWS:
        if ref_w is None:
            setting_name = "no_ref"
        else:
            setting_name = f"ref_w{ref_w}"

        output_dir = os.path.join(BASE_OUTPUT_DIR, setting_name)
        video_dir = os.path.join(BASE_VIDEO_DIR, setting_name)

        os.makedirs(output_dir, exist_ok = True)
        os.makedirs(video_dir, exist_ok = True)

        print(f"\n===== Setting: {setting_name} =====")
        print(f"Output dir: {output_dir}")
        print(f"Video dir : {video_dir}")

        for vid_path in video_files:
            vid_path = Path(vid_path)
            seq_name = vid_path.stem

            if ref_w is not None:
                images = test.load_video(str(vid_path), frame_stride = FRAME_STRIDE, max_frames = MAX_FRAMES)
                T = len(images)
                ref_indices = test.select_reference_frame_indices(images, reference_window_size = ref_w)
                num_refs = len(ref_indices)
                ref_stats[ref_w].append(num_refs)
                csv_writer.writerow([seq_name, T, ref_w, num_refs])

                print(f"[STATS] {seq_name}: T = {T}, ref_window = {ref_w}, num_refs = {num_refs}")
            else:
                images = test.load_video(str(vid_path), frame_stride = FRAME_STRIDE, max_frames = MAX_FRAMES)
                T = len(images)
                print(f"[STATS] {seq_name}: T={T}, ref_window=None (no_ref)")

            print(f"[RUN] {setting_name} / {vid_path.name}")

            cmd = [
                "python", "test.py",
                "--pretrained_model_name_or_path", PRETRAINED_MODEL,
                "--input_video", str(vid_path),
                "--output_dir", output_dir,
                "--video_dir", video_dir,
                "--width", str(WIDTH),
                "--height", str(HEIGHT),
                "--window_size", str(WINDOW_SIZE),
                "--stride", str(STRIDE),
                "--num_inference_steps", str(NUM_INFERENCE_STEPS),
                "--min_guidance_scale", str(MIN_GUIDANCE),
                "--max_guidance_scale", str(MAX_GUIDANCE),
                "--fps", str(FPS),
                "--motion_bucket_id", str(MOTION_BUCKET_ID),
                "--noise_aug_strength", str(NOISE_AUG),
                "--seed", str(SEED),
                "--decode_chunk_size", str(DECODE_CHUNK_SIZE),
                "--frame_stride", str(FRAME_STRIDE),
                "--pad_direction", PAD_DIRECTION,
            ]

            if MAX_FRAMES is not None:
                cmd += ["--max_frames", str(MAX_FRAMES)]

            if USE_BLUR_COMP:
                cmd += ["--use_blur_comp"]

            if SAVE_MP4:
                cmd += ["--save_mp4"]

            if ref_w is not None:
                cmd += ["--use_ref_selection", "--ref_window", str(ref_w)]

            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

        if ref_w is not None and len(ref_stats[ref_w]) > 0:
            avg_refs = sum(ref_stats[ref_w]) / len(ref_stats[ref_w])
            print(f"\n[SUMMARY] ref_window={ref_w}: "
                  f"mean num_refs over {len(ref_stats[ref_w])} sequences = {avg_refs:.2f}")

    csv_f.close()
    print(f"\n[OK] Saved ref statistics to {csv_path}")


if __name__ == "__main__":
    main()
