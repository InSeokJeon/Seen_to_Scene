import inspect
import PIL.Image
import numpy as np

from tqdm import tqdm
from einops import rearrange
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.utils import BaseOutput, logging
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from utils.util import _resize_with_antialiasing, _visualize_flow

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
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):

    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, vae, image_encoder, unet, scheduler, feature_extractor, fix_raft, fix_flow_complete, lat_bi_propagator):
        super().__init__()

        self.register_modules(vae = vae, image_encoder = image_encoder, unet = unet, fix_raft = fix_raft, fix_flow_complete = fix_flow_complete, scheduler = scheduler, feature_extractor = feature_extractor, lat_bi_propagator = lat_bi_propagator)
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
            raise ValueError(f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.")

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
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
            
        frames = torch.cat(frames, dim = 0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        
        return frames

    def check_inputs(self, image, height, width):
        if (not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list)):
            raise ValueError("`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator, latents = None):
        shape = (batch_size, num_frames, num_channels_latents // 2, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.")
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
    def ddim_inversion(self, latent, image_latents, masks, flows, cond, added_time_ids):
        timesteps = reversed(self.scheduler.timesteps)
        
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                alpha_prod_t = self.scheduler.alphas_cumprod[i]
                alpha_prod_t_prev = (self.scheduler.alphas_cumprod[i - 1] if i > 0 else self.scheduler.alphas_cumprod[0])

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                if latent.shape[0] == 1:
                    latent_model_input = torch.cat([latent] * 2)
                else:
                    latent_model_input = latent

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                out_b, out_f, cond_latents = self.lat_bi_propagator(latent_model_input, image_latents, flows[0], flows[1], masks) 
                latent_model_input = torch.cat([latent_model_input, cond_latents], dim = 2)
                eps = self.unet(latent_model_input, t, encoder_hidden_states = cond, added_time_ids = added_time_ids).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                
        return latent

    @torch.no_grad()
    def __call__(self, images, masks, height = 576, width = 1024, num_frames = None, num_inference_steps = 25, min_guidance_scale = 1.0, max_guidance_scale = 3.0, fps = 7, motion_bucket_id = 127, noise_aug_strength = 0.02, decode_chunk_size = None, num_videos_per_prompt = 1, generator = None, latents = None, output_type = "pil", callback_on_step_end = None, callback_on_step_end_tensor_inputs = ["latents"], return_dict = True, g_step = None):
        image = images[0]
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        total_images = []
        for input_idx in range(len(images)):
            input_img = images[input_idx]
            input_img = self.image_processor.pil_to_numpy(input_img)
            input_img = self.image_processor.numpy_to_pt(input_img)

            total_images.append(input_img.unsqueeze(1))
        total_images = torch.cat(total_images, dim = 1).to(device, dtype = image_embeddings.dtype)
        total_images = total_images * 2.0 - 1.0

        total_masks = []
        for input_idx in range(len(masks)):
            input_mask = masks[input_idx]
            input_mask = self.image_processor.pil_to_numpy(input_mask)
            input_mask = self.image_processor.numpy_to_pt(input_mask)

            total_masks.append(input_mask.unsqueeze(1))
        total_masks = torch.cat(total_masks, dim = 1).to(device, dtype = image_embeddings.dtype)
        total_masks = total_masks[:, :, 0:1, :, :]
        
        gt_flows_bi = self.fix_raft(total_images, iters = 20)
        pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, total_masks)
        pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, total_masks)
        _visualize_flow("./result", pred_flows_bi, gt_flows_bi, g_step)
        
        fps = fps - 1

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        condition_latents = []
        for input_idx in range(0, len(images)):
            image = images[input_idx]
            mask = masks[input_idx]

            image = self.image_processor.preprocess(image, height = height, width = width).to(device)
            mask = self.image_processor.preprocess(mask, height = height, width = width).to(device)
            mask = (mask + 1.0) / 2.0

            masked_image = image * (1 - mask)

            noise = randn_tensor(masked_image.shape, generator = generator, device = device, dtype = masked_image.dtype)
            masked_image = masked_image + noise_aug_strength * noise

            condition_latent = self._encode_vae_image(masked_image, device = device, num_videos_per_prompt = num_videos_per_prompt, do_classifier_free_guidance = self.do_classifier_free_guidance)
            condition_latents.append(condition_latent.unsqueeze(1))

        image_latents = torch.cat(condition_latents, dim = 1)
        image_latents = image_latents.to(image_embeddings.dtype)
                        
        added_time_ids = self._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype, batch_size, num_videos_per_prompt, self.do_classifier_free_guidance)
        added_time_ids = added_time_ids.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device = device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_videos_per_prompt, num_frames, num_channels_latents, height, width, image_embeddings.dtype, device, generator, latents)

        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        latents = self.ddim_inversion(latents, image_latents, total_masks, pred_flows_bi, image_embeddings, added_time_ids)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        with self.progress_bar(total = num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                                
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                out_b, out_f, cond_latents = self.lat_bi_propagator(latent_model_input, image_latents, pred_flows_bi[0], pred_flows_bi[1], total_masks) 
                latent_model_input = torch.cat([latent_model_input, cond_latents], dim = 2)   
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states = image_embeddings, added_time_ids = added_time_ids, return_dict = False)[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype = torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type = output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames = frames)


