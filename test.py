import os
import re
import cv2
import inspect
import argparse
import PIL.Image
import numpy as np
import imageio.v2 as imageio

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.utils import BaseOutput, logging
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from models.bidirectional_flow_raft import RAFT_bi
from models.latent_warping import LatentPropagation
from models.unet import UNetSpatioTemporalConditionModel
from models.recurrent_flow_completion import RecurrentFlowCompleteNet

from decord import VideoReader, cpu
from utils.util import _resize_with_antialiasing, _visualize_flow

import time

from contextlib import contextmanager
from dataclasses import dataclass, asdict

@dataclass
class ProfileRec:
    name: str
    time_ms: float
    peak_mem_gb: float
    params_m: float = 0.0
    note: str = ""

def count_params_m(module: torch.nn.Module, trainable_only: bool = False) -> float:
    if module is None:
        return 0.0
    if trainable_only:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in module.parameters())
    return n / 1e6

@contextmanager
def profile_cuda_section(records: list, name: str, note: str = "", sync: bool = True, reset_peak: bool = True):
    if torch.cuda.is_available():
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        if sync:
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        if sync:
            torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        t0 = time.time()
        yield
        ms = (time.time() - t0) * 1000.0
        peak_gb = 0.0

    records.append(ProfileRec(name=name, time_ms=float(ms), peak_mem_gb=float(peak_gb), note=note))

def print_profile_table(records: list):
    print("\n================ Profiling Results ================")
    print(f"{'Module':40s} | {'Time (ms)':>10s} | {'PeakMem (GB)':>12s} | {'Params (M)':>10s} | Note")
    print("-"*100)
    for r in records:
        print(f"{r.name:40s} | {r.time_ms:10.2f} | {r.peak_mem_gb:12.3f} | {r.params_m:10.2f} | {r.note}")
    print("="*100+"\n")


logger = logging.get_logger(__name__)


def _append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []

    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def save_video_or_gif(frames_pil, out_path: Path, fps: int, use_mp4: bool):
    out_path.parent.mkdir(parents = True, exist_ok = True)

    if use_mp4:
        try:
            import imageio_ffmpeg
            imageio.mimsave(out_path, [np.array(f) for f in frames_pil], fps = fps, codec = "libx264", quality = 8)
            return
        except Exception as e:
            print(f"[WARN] Not Installed Tool: {e}\n -> Save GIF.")
            out_path = out_path.with_suffix(".gif")

    imageio.mimsave(out_path, [np.array(f) for f in frames_pil], fps = fps)


def load_video(video_path, frame_stride = 1, max_frames = None):
    vr = VideoReader(video_path, ctx = cpu(0))
    num_frames_total = len(vr)

    indices = list(range(0, num_frames_total, frame_stride))

    images = []
    for idx in indices:
        if max_frames is not None and len(images) >= max_frames:
            break

        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame)
        images.append(img)

    return images


def pad_image_and_mask(image, target_width, target_height, pad_direction = "horizontal"):
    orig_w, orig_h = image.size

    if pad_direction == "horizontal":
        if orig_h != target_height:
            raise ValueError(
                f"[Error] Orignal Height and Target Height should be same in Horizontal Outpainting Mode")
        if orig_w > target_width:
            raise ValueError(
                f"[Error] Original Width should be smaller than Target Width")

        new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        x_offset = (target_width - orig_w) // 2
        new_img.paste(image, (x_offset, 0))

        mask_np = np.zeros((target_height, target_width), dtype = np.uint8)
        if x_offset > 0:
            mask_np[:, :x_offset] = 255
            mask_np[:, x_offset + orig_w:] = 255

    elif pad_direction == "vertical":
        if orig_w != target_width:
            raise ValueError(
                f"[Error] Orignal Width and Target Width should be same in Vertical Outpainting Mode")
        if orig_h > target_height:
            raise ValueError(
                f"[Error] Original Height should be smaller than Target Height")

        new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        y_offset = (target_height - orig_h) // 2
        new_img.paste(image, (0, y_offset))

        mask_np = np.zeros((target_height, target_width), dtype = np.uint8)
        if y_offset > 0:
            mask_np[:y_offset, :] = 255
            mask_np[y_offset + orig_h:, :] = 255
    else:
        raise ValueError(f"Padding Direction must be 'Horizontal' or 'Vertical', Got {pad_direction}")

    mask_img = Image.fromarray(mask_np, mode = "L")

    return new_img, mask_img


def pad_images_and_make_masks(images, target_width, target_height, pad_direction):
    masks = []
    padded_images = []
    for img in images:
        p_img, m_img = pad_image_and_mask(img, target_width, target_height, pad_direction)
        padded_images.append(p_img)
        masks.append(m_img)

    return padded_images, masks


def compute_structure_term(im1_gray, im2_gray):
    x = im1_gray.astype(np.float32) / 255.0
    y = im2_gray.astype(np.float32) / 255.0

    mu_x = x.mean()
    mu_y = y.mean()

    x_c = x - mu_x
    y_c = y - mu_y

    sigma_x = np.sqrt((x_c ** 2).mean() + 1e-8)
    sigma_y = np.sqrt((y_c ** 2).mean() + 1e-8)
    sigma_xy = (x_c * y_c).mean()

    C3 = 1e-4
    denom = (sigma_x * sigma_y + C3)

    if denom == 0:
        return 0.0

    return float((sigma_xy + C3) / denom)


def select_reference_frame_indices(images, reference_window_size):
    T = len(images)

    if T == 0:
        return []

    grays = [np.array(img.convert("L")) for img in images]

    selected = [0]
    current = 0

    while current < T - 1:
        remaining = (T - 1) - current

        if remaining < reference_window_size:
            selected.append(T - 1)
            break

        start = current + 1
        end = current + reference_window_size

        best_idx = None
        best_score = None

        for j in range(start, end + 1):
            s = compute_structure_term(grays[current], grays[j])
            if best_score is None or s < best_score:
                best_score = s
                best_idx = j

        if best_idx is None:
            best_idx = T - 1

        selected.append(best_idx)
        current = best_idx

    if selected[-1] != T - 1:
        selected.append(T - 1)

    return selected


def build_pairs_for_all_frames(T, ref_indices):
    refs = sorted(ref_indices)
    R = len(refs)
    assert R >= 1, "At Least One Reference Frame Needed"

    pairs_chain = []

    for j in range(R-1, 0, -1):
        s = refs[j] 
        t = refs[j-1]
        pairs_chain.append((s, t))

    pairs_to_frame = []
    for t in range(T):
        seg_ref = None
        for r in refs:
            if r >= t:
                seg_ref = r
                break
            
        if seg_ref is None:
            seg_ref = refs[-1]

        if seg_ref == t:
            continue

        pairs_to_frame.append((seg_ref, t))

    return pairs_chain, pairs_to_frame


def get_views(T, window_size = 25, stride = 10):
    if T <= window_size:
        return [(0, T)]

    views = []
    start = 0
    while start + window_size <= T:
        views.append((start, start + window_size))
        start += stride

    if views[-1][1] < T:
        views.append((T - window_size, T))

    seen = set()
    views_unique = []
    for st, ed in views:
        if (st, ed) not in seen:
            views_unique.append((st, ed))
            seen.add((st, ed))

    return views_unique


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, vae, image_encoder, unet, scheduler, feature_extractor, fix_raft, vo_flow_complete, lat_bi_propagator):
        super().__init__()
        self.register_modules(vae = vae, image_encoder = image_encoder, unet = unet, fix_raft = fix_raft, vo_flow_complete = vo_flow_complete, scheduler = scheduler, feature_extractor = feature_extractor, lat_bi_propagator = lat_bi_propagator)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor = self.vae_scale_factor)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype
        
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0
            image = self.feature_extractor(images = image, do_normalize = True, do_center_crop = False, do_resize = False, do_rescale = False, return_tensors = "pt").pixel_values
        image = image.to(device = device, dtype = dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        image = image.to(device = device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])

        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, dtype, batch_size, num_videos_per_prompt, do_classifier_free_guidance):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created.")

        add_time_ids = torch.tensor([add_time_ids], dtype = dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size = 14):
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}

            if accepts_num_frames:
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)

        frames = torch.cat(frames, dim = 0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()

        return frames

    def check_inputs(self, image, height, width):
        if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]`")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator, latents = None):
        shape = (batch_size, num_frames, num_channels_latents // 2, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"list of generators length {len(generator)} != batch size {batch_size}")

        if latents is None:
            latents = randn_tensor(shape, generator = generator, device = device, dtype = dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale

        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def propagate_latents_global(self, cond_latents_all, flows_bi, masks_all, flow_pairs_info = None):
        flows_fw, flows_bw = flows_bi
        
        _, _, cond_latents_all = self.lat_bi_propagator(cond_latents_all, flows_fw, flows_bw, masks_all, flow_pairs_info = flow_pairs_info)
        
        return cond_latents_all

    @torch.no_grad()
    def ddim_inversion_global(self, latents_global, cond_latents_all, masks_all, flows_bi, cond, added_time_ids):
        timesteps = reversed(self.scheduler.timesteps)

        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = (torch.cat([latents_global, latents_global]) if latents_global.shape[0] == 1 else latents_global)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, cond_latents_all], dim = 2)
                eps = self.unet(latent_model_input, t, encoder_hidden_states = cond, added_time_ids = added_time_ids).sample

                idx = i
                alpha_prod_t = self.scheduler.alphas_cumprod[idx]
                alpha_prod_t_prev = (self.scheduler.alphas_cumprod[idx - 1] if idx > 0 else self.scheduler.alphas_cumprod[0])
                mu, mu_prev = alpha_prod_t.sqrt(), alpha_prod_t_prev.sqrt()
                sigma, sigma_prev = (1 - alpha_prod_t).sqrt(), (1 - alpha_prod_t_prev).sqrt()
                pred_x0 = (latents_global - sigma_prev * eps) / mu_prev
                latents_global = mu * pred_x0 + sigma * eps

        return latents_global

    @torch.no_grad()
    def __call_seen_to_scene__(
        self,
        images,
        masks,
        *,
        refer_idx = None, 
        height = 576,
        width = 1024,
        window_size = 16,
        stride = 4,
        num_inference_steps = 25,
        min_guidance_scale = 1.0,
        max_guidance_scale = 1.0,
        fps = 7,
        motion_bucket_id = 127,
        noise_aug_strength = 0.02,
        decode_chunk_size = None,
        num_videos_per_prompt = 1,
        generator = None,
        output_type = "pil",
        return_dict = True):

        profiles = []

        module_params = {"ImageEncoder(CLIP)": count_params_m(self.image_encoder),
                         "VAE": count_params_m(self.vae),
                         "UNet": count_params_m(self.unet),
                         "RAFT(flow)": count_params_m(self.fix_raft),
                         "FCNet(flow_complete)": count_params_m(self.vo_flow_complete),
                         "LatentPropagator": count_params_m(self.lat_bi_propagator)}

        device = torch.device("cuda")
        self.check_inputs(images[0], height, width)
        batch_size = 1
        T = len(images)
        decode_chunk_size = decode_chunk_size or window_size

        self._guidance_scale = max_guidance_scale
        
        with profile_cuda_section(profiles, "CLIP encode (image_embeddings)", note="encode first frame"):
            image_embeddings = self._encode_image(images[0], device, num_videos_per_prompt, self.do_classifier_free_guidance)
        profiles[-1].params_m = module_params["ImageEncoder(CLIP)"]
        
        total_images = []
        for im in images:
            x = self.image_processor.pil_to_numpy(im)
            x = self.image_processor.numpy_to_pt(x)
            total_images.append(x.unsqueeze(1))
        total_images = torch.cat(total_images, dim = 1).to(device, dtype = image_embeddings.dtype)
        total_images = total_images * 2.0 - 1.0

        total_masks = []
        for mk in masks:
            m = self.image_processor.pil_to_numpy(mk)
            m = self.image_processor.numpy_to_pt(m)
            total_masks.append(m.unsqueeze(1))
        total_masks = torch.cat(total_masks, dim = 1).to(device, dtype = image_embeddings.dtype)
        total_masks = total_masks[:, :, 0:1, :, :] 

        with torch.autocast(device_type = "cuda", enabled = False):
            imgs_raft = total_images.float()
            masks_raft = total_masks.float()
            
            Bm, Tm, _, Hm, Wm = masks_raft.shape
            m_flat = masks_raft.view(Bm * Tm, 1, Hm, Wm)
            m_dil = F.max_pool2d(m_flat, kernel_size = 3, stride = 1, padding = 1)
            masks_dil = m_dil.view(Bm, Tm, 1, Hm, Wm)

            if (refer_idx is None) or (len(refer_idx) == 0):
                with profile_cuda_section(profiles, "RAFT flow", note="all pairs"):
                    gt_flows_bi = self.fix_raft(imgs_raft, iters = 20)
                flow_pairs_info = None
            else:
                with profile_cuda_section(profiles, "RAFT flow", note="selected pairs"):
                    gt_flows_bi = self.fix_raft.forward_pairs(imgs_raft, pairs = refer_idx, iters = 20, bidirectional = True)
                flow_pairs_info = refer_idx
            profiles[-1].params_m = module_params["RAFT(flow)"]

            with profile_cuda_section(profiles, "FCNet flow completion", note="forward_bidirect_flow + combine"):
                pred_flows_bi, _ = self.vo_flow_complete.forward_bidirect_flow(gt_flows_bi, masks_dil)
                pred_flows_bi = self.vo_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, masks_dil)
                flows_bi = pred_flows_bi
            profiles[-1].params_m = module_params["FCNet(flow_complete)"]
            
        needs_upcasting = (self.vae.dtype == torch.float16) and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype = torch.float32)

        with profile_cuda_section(profiles, "VAE encode (cond_latents_all)", note=f"T={T}"):
            cond_list = []
            for idx in range(T):
                image_pt = self.image_processor.preprocess(images[idx], height = height, width = width).to(device)
                mask_pt = self.image_processor.preprocess(masks[idx], height = height, width = width).to(device)
                mask_pt = (mask_pt + 1.0) / 2.0

                masked_image = image_pt * (1 - mask_pt)
                noise = randn_tensor(masked_image.shape, generator = generator, device = device, dtype = masked_image.dtype)
                masked_image = masked_image + noise_aug_strength * noise

                cond_lat = self._encode_vae_image(masked_image, device = device, num_videos_per_prompt = num_videos_per_prompt, do_classifier_free_guidance = self.do_classifier_free_guidance)
                cond_list.append(cond_lat.unsqueeze(1))
            cond_latents_all = torch.cat(cond_list, dim = 1).to(image_embeddings.dtype)
        profiles[-1].params_m = module_params["VAE"]
        
        fps_ = fps - 1
        added_time_ids = self._get_add_time_ids(fps_, motion_bucket_id, noise_aug_strength, image_embeddings.dtype, batch_size, num_videos_per_prompt, self.do_classifier_free_guidance).to(device)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        with profile_cuda_section(profiles, "Reference-guided latent propagation", note="lat_bi_propagator"):
            cond_latents_all = self.propagate_latents_global(cond_latents_all, flows_bi, total_masks, flow_pairs_info = flow_pairs_info)
        profiles[-1].params_m = module_params["LatentPropagator"]
        
        self.scheduler.set_timesteps(num_inference_steps, device = device)
        num_channels_latents = self.unet.config.in_channels

        latents_global = self.prepare_latents(batch_size * num_videos_per_prompt, T, num_channels_latents, height, width, image_embeddings.dtype, device, generator, latents = None)
        guidance_scale_global = torch.linspace(min_guidance_scale, max_guidance_scale, T, device = device, dtype = latents_global.dtype)
        guidance_scale_global = guidance_scale_global.unsqueeze(0).repeat(batch_size * num_videos_per_prompt, 1)
        self._guidance_scale = _append_dims(guidance_scale_global, latents_global.ndim)

        latents_global = self.ddim_inversion_global(latents_global, cond_latents_all, total_masks, flows_bi, image_embeddings, added_time_ids)

        latents = latents_global
        cond_lat = cond_latents_all

        views = get_views(T, window_size = window_size, stride = stride)

        self.scheduler.set_timesteps(num_inference_steps, device = device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with profile_cuda_section(profiles, "Diffusion denoising", note=f"steps={num_inference_steps}, views={len(views)}"):
            with self.progress_bar(total = num_inference_steps) as pbar:
                for i, t in enumerate(timesteps):
                    values_tot = torch.zeros_like(latents)
                    counts_tot = torch.zeros_like(latents)

                    for (st, ed) in views:
                        latents_win = latents[:, st:ed]
                        cond_win = cond_lat[:, st:ed]
                        guidance_scale_win = self._guidance_scale[:, st:ed]

                        latent_model_input = self.scheduler.scale_model_input(latents_win, t)
                        latent_model_input = torch.cat([latent_model_input, cond_win], dim = 2)

                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states = image_embeddings, added_time_ids = added_time_ids, return_dict = False)[0]

                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale_win * (noise_pred_cond - noise_pred_uncond)

                        values_tot[:, st:ed] += noise_pred
                        counts_tot[:, st:ed] += 1

                    noise_pred_full = values_tot / counts_tot
                    latents = self.scheduler.step(noise_pred_full, t, latents).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        pbar.update()
        profiles[-1].params_m = module_params["UNet"]

        with profile_cuda_section(profiles, "VAE decode", note=f"decode_chunk={decode_chunk_size}"):
            latents_to_decode = latents[0:1]
            frames_all = self.decode_latents(latents_to_decode, num_frames = T, decode_chunk_size = decode_chunk_size)[0]
        profiles[-1].params_m = module_params["VAE"]
        
        frames_batched = frames_all.unsqueeze(0).permute(0, 1, 2, 3, 4)
        frames = tensor2vid(frames_batched, self.image_processor, output_type = output_type)[0]
        
        print_profile_table(profiles)
        
        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type = str, required = True)
    parser.add_argument("--input_video", type = str, required = True)
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--video_dir", type = str, required = True)
    parser.add_argument("--width", type = int, default = 256)
    parser.add_argument("--height", type = int, default = 256)
    parser.add_argument("--window_size", type = int, default = 25)
    parser.add_argument("--stride", type = int, default = 16)
    parser.add_argument("--num_inference_steps", type = int, default = 25)
    parser.add_argument("--min_guidance_scale", type = float, default = 1.0)
    parser.add_argument("--max_guidance_scale", type = float, default = 3.0)
    parser.add_argument("--fps", type = int, default = 7)
    parser.add_argument("--motion_bucket_id", type = int, default = 127)
    parser.add_argument("--noise_aug_strength", type = float, default = 0.02)
    parser.add_argument("--seed", type = int, default = 2026)
    parser.add_argument("--save_mp4", action = "store_true")
    parser.add_argument("--decode_chunk_size", type = int, default = 8)
    parser.add_argument("--frame_stride", type = int, default = 1)
    parser.add_argument("--max_frames", type = int, default = None)
    parser.add_argument("--pad_direction", type = str, default = "vertical", choices = ["horizontal", "vertical"])
    parser.add_argument("--use_ref_selection", action = "store_true")
    parser.add_argument("--ref_window", type = int, default = 4)
    parser.add_argument("--use_blur_comp", action = "store_true")
    parser.add_argument("--blur_kernel_size", type = int, default = 11)
    parser.add_argument("--blur_sigma", type = float, default = 10.0)
    parser.add_argument("--input_strength", type = float, default = 0.5)
    args = parser.parse_args()

    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder = "feature_extractor")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder = "vae", torch_dtype = torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder = "image_encoder", torch_dtype = torch.float16)

    fix_raft = RAFT_bi("./weights/raft-things.pth")

    unet_check_path = "./output/checkpoint-100000/unet"
    unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_check_path, subfolder = "unet", torch_dtype = torch.float16)

    propagator_check_path = "./output/checkpoint-100000/"
    vo_flow_complete = RecurrentFlowCompleteNet()
    vo_flow_complete.load_state_dict(torch.load(f"{propagator_check_path}/fix_flow_complete/pytorch_model.bin", map_location = "cpu"))
    lat_bi_propagator = LatentPropagation()
    lat_bi_propagator.load_state_dict(torch.load(f"{propagator_check_path}/lat_bi_propagator/pytorch_model.bin", map_location = "cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device = device).manual_seed(args.seed)

    weight_dtype = torch.float32
    image_encoder.to(device, dtype = weight_dtype)
    vae.to(device, dtype = weight_dtype)
    fix_raft.to(device, dtype = weight_dtype)
    vo_flow_complete.to(device, dtype = weight_dtype)

    pipe = StableVideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, unet = unet, vae = vae, fix_raft = fix_raft, vo_flow_complete = vo_flow_complete, lat_bi_propagator = lat_bi_propagator, torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    pipe.set_progress_bar_config(disable = False)

    input_video_path = Path(args.input_video)
    seq_name = input_video_path.stem

    out_root = Path(args.output_dir)
    out_root.mkdir(parents = True, exist_ok = True)
    seq_out_dir = out_root / seq_name

    video_root = Path(args.video_dir)
    video_root.mkdir(parents = True, exist_ok = True)
    video_out_dir_pred = video_root / "pred"
    video_out_dir_comp = video_root / "comp"
    video_out_dir_pred.mkdir(parents = True, exist_ok = True)
    video_out_dir_comp.mkdir(parents = True, exist_ok = True)

    frames_pred_dir = seq_out_dir / "frames_pred"
    frames_comp_dir = seq_out_dir / "frames_comp"
    frames_comp_blur_dir = seq_out_dir / "frames_comp_blur"
    frames_pred_dir.mkdir(parents = True, exist_ok = True)
    frames_comp_dir.mkdir(parents = True, exist_ok = True)
    if args.use_blur_comp:
        frames_comp_blur_dir.mkdir(parents = True, exist_ok = True)

    images = load_video(str(input_video_path), frame_stride = args.frame_stride, max_frames = args.max_frames)
    T = len(images)

    if T == 0:
        raise RuntimeError(f"[Error] Can not read frames in : {input_video_path}")

    print(f"[INFO] Loaded {T} frames from video: {input_video_path}")

    refer_idx = None
    if args.use_ref_selection:
        ref_indices = select_reference_frame_indices(images, reference_window_size = args.ref_window)
        pairs_chain, pairs_to_frame = build_pairs_for_all_frames(T, ref_indices)
        pairs_all = pairs_chain + pairs_to_frame
        refer_idx = pairs_all
        
        print(f"[INFO] Reference frame indices (Len = {len(ref_indices)}): {ref_indices}")
    else:
        print("[INFO] Reference selection OFF (Use All Frames)")

    target_height = args.height
    target_width = args.width

    if target_height % 8 != 0 or target_width % 8 != 0:
        raise ValueError(
            f"Target Height ({target_height}) and Width ({target_width}) Should devide into 8")

    padded_images, masks = pad_images_and_make_masks(images, target_width = target_width, target_height = target_height, pad_direction = args.pad_direction)

    print(f"[INFO] After padding, sequence length: {T}, size: {target_width}x{target_height}")

    stt = time.time()
    with torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = (device.type == "cuda")):
        out = pipe.__call_seen_to_scene__(
            images = padded_images,
            masks = masks,
            height = target_height,
            width = target_width,
            refer_idx = refer_idx,
            window_size = args.window_size,
            stride = args.stride,
            num_inference_steps = args.num_inference_steps,
            min_guidance_scale = args.min_guidance_scale,
            max_guidance_scale = args.max_guidance_scale,
            fps = args.fps,
            motion_bucket_id = args.motion_bucket_id,
            noise_aug_strength = args.noise_aug_strength,
            decode_chunk_size = args.decode_chunk_size,
            num_videos_per_prompt = 1,
            generator = generator,
            output_type = "pil",
            return_dict = True)

    frames = out.frames

    print("========")
    print(time.time() - stt)
    comp_frames = []
    comp_blur_frames = []
    blur_k = args.blur_kernel_size
    
    if blur_k % 2 == 0:
        blur_k += 1
        
    for i, pred_pil in enumerate(frames):
        gt_pil = padded_images[i]
        mask_pil = masks[i]

        gt = np.array(gt_pil, dtype = np.uint8)
        pred = np.array(pred_pil, dtype = np.uint8)
        m = np.array(mask_pil.convert("L"), dtype = np.uint8) / 255.0
        m = m[..., None]

        comp = (gt.astype(np.float32) * (1.0 - m) + pred.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
        comp_pil = Image.fromarray(comp)

        pred_pil.save(frames_pred_dir / f"frame_{i:04d}.png")
        comp_pil.save(frames_comp_dir / f"frame_{i:04d}.png")

        comp_frames.append(comp_pil)
        
        if args.use_blur_comp:
            gt_f = gt.astype(np.float32)
            pred_f = pred.astype(np.float32)

            mask_gray = np.array(mask_pil.convert("L"), dtype = np.uint8)

            blurred = cv2.GaussianBlur(mask_gray, (blur_k, blur_k), args.blur_sigma)
            blurred = blurred.astype(np.float32) / 255.0

            alpha_in = (1.0 - blurred) * float(args.input_strength)
            alpha_gen = 1.0 - alpha_in

            alpha_in_3 = alpha_in[..., None]
            alpha_gen_3 = alpha_gen[..., None]

            comp_blur = gt_f * alpha_in_3 + pred_f * alpha_gen_3
            comp_blur = np.clip(comp_blur, 0, 255).astype(np.uint8)

            comp_blur_pil = Image.fromarray(comp_blur)
            comp_blur_pil.save(frames_comp_blur_dir / f"frame_{i:04d}.png")
            comp_blur_frames.append(comp_blur_pil)

    video_name_pred = f"{seq_name}_pred.mp4" if args.save_mp4 else f"{seq_name}_pred.gif"
    video_name_comp = f"{seq_name}_comp.mp4" if args.save_mp4 else f"{seq_name}_comp.gif"

    save_video_or_gif(frames, seq_out_dir / video_name_pred, fps = args.fps, use_mp4 = args.save_mp4)
    save_video_or_gif(comp_frames, seq_out_dir / video_name_comp, fps = args.fps, use_mp4 = args.save_mp4)

    save_video_or_gif(frames, video_out_dir_pred / video_name_pred, fps = args.fps, use_mp4 = args.save_mp4)
    save_video_or_gif(comp_frames, video_out_dir_comp / video_name_comp, fps = args.fps, use_mp4 = args.save_mp4)

    if args.use_blur_comp and len(comp_blur_frames) > 0:
        video_name_comp_blur = f"{seq_name}_comp_blur.mp4" if args.save_mp4 else f"{seq_name}_comp_blur.gif"
        save_video_or_gif(comp_blur_frames, seq_out_dir / video_name_comp_blur, fps = args.fps, use_mp4 = args.save_mp4)
        save_video_or_gif(comp_blur_frames, video_out_dir_comp / video_name_comp_blur, fps = args.fps, use_mp4 = args.save_mp4)

    print(f"[OK] saved: {seq_out_dir} (pred, composition, blur_comp = {args.use_blur_comp})")
