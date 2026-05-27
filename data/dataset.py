import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.util import create_video_outpainting_mask


class OutpaintSequenceDataset(Dataset):
    def __init__(self, data_root, width = 1024, height = 576, sample_frames = 25, mask_ratio = 0.33):        
        self.num_samples = 100000
        self.base_folder = data_root
        self.img_folders = []
        self.mask_folders = []

        for (path, dir, files) in os.walk(self.base_folder, 'JPEGImages'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    if path not in self.img_folders:
                        if len(os.listdir(path)) > sample_frames:
                            self.img_folders.append(path)
        
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return self.num_samples
    
    def resize_center_crop(self, img: Image.Image, target_w: int, target_h: int):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        left   = (new_w - target_w) // 2
        top    = (new_h - target_h) // 2
        right  = left + target_w
        bottom = top  + target_h
        
        return resized.crop((left, top, right, bottom))
    
    
    def __getitem__(self, idx):
        chosen_img_folder = random.choice(self.img_folders)
        img_folder_path = os.path.join(self.base_folder, chosen_img_folder)
        img_frames = os.listdir(img_folder_path)
        img_frames.sort()

        img_start_idx = random.randint(0, len(img_frames) - self.sample_frames)
        selected_img_frames = img_frames[img_start_idx:img_start_idx + self.sample_frames]

        masks = create_video_outpainting_mask(len(selected_img_frames), imageHeight = self.height, imageWidth = self.width, mask_ratio = self.mask_ratio)

        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        mask_values = torch.empty((self.sample_frames, 1, self.height, self.width))

        for i, frame_name in enumerate(selected_img_frames):
            img_frame_path = os.path.join(img_folder_path, frame_name)
            
            img = Image.open(img_frame_path)
            img_resized = self.resize_center_crop(img, 256, 256)
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            img_normalized = img_tensor / 127.5 - 1

            mask = masks[i]
            mask_tensor = torch.from_numpy(np.array(mask)).float()
            mask_tensor = mask_tensor / 255.0

            if self.channels == 3:
                img_normalized = img_normalized.permute(2, 0, 1)
            elif self.channels == 1:
                img_normalized = img_normalized.mean(dim=2, keepdim=True)

            pixel_values[i] = img_normalized
            mask_values[i] = mask_tensor.unsqueeze(0)

        return {'pixel_values': pixel_values, 'mask_values': mask_values}