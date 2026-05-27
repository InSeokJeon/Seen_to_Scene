import os
import math
import yaml
import shutil
import random
import logging
import argparse
from pathlib import Path
from urllib.parse import urlparse

import cv2
import PIL
import numpy as np
from PIL import Image

import torch
import torchvision as tv
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from tqdm.auto import tqdm
from einops import rearrange
from packaging import version
from huggingface_hub import create_repo, upload_folder

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version, deprecate, load_image

from data.dataset import OutpaintSequenceDataset
from pipeline import StableVideoDiffusionPipeline
from models.bidirectional_flow_raft import RAFT_bi
from models.latent_warping import LatentPropagation
from models.unet import UNetSpatioTemporalConditionModel
from models.recurrent_flow_completion import RecurrentFlowCompleteNet
from utils.util import _resize_with_antialiasing, rand_log_normal, export_to_gif, _visualize_flow

from utils.loss import FlowLoss


check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level = "INFO")


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f = video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def merge_args_with_config(args, config):
    for key, value in config.items():
        setattr(args, key, value)
    return args


def _module_subfolder(model):
    cls = type(model).__name__.lower()
    if "unet" in cls:
        return "unet"
    if "vae" in cls:
        return "vae"
    if "latentpropagation" in cls or "lat_bi_propagator" in cls:
        return "lat_bi_propagator"
    if "raft" in cls:
        return "fix_raft"
    if "flow" in cls and "complete" in cls:
        return "fix_flow_complete"
    return cls


def parse_args():
    parser = argparse.ArgumentParser(description = "Script to train Video Outpainting Model Based on Stable Video Diffusion.")
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = "stabilityai/stable-video-diffusion-img2vid-xt-1-1", help = "Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type = str, default = None, required = False, help = "Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--num_frames", type = int, default = 25)
    parser.add_argument("--width", type = int, default = 1024)
    parser.add_argument("--height", type = int, default = 576)
    parser.add_argument("--one_side_mask_ratio", type = int, default = 0.33)
    parser.add_argument("--num_validation_images", type = int, default = 1, help = "Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_steps", type = int, default = 500, help = "Run fine-tuning validation every X epochs.")
    parser.add_argument("--output_dir", type = str, default = "./outputs", help = "The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type = int, default = None, help = "A seed for reproducible training.")
    parser.add_argument("--per_gpu_batch_size", type = int, default = 1, help = "Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type = int, default = 100)
    parser.add_argument("--max_train_steps", type = int, default = None, help = "Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1, help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action = "store_true", help = "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type = float, default = 1e-4, help = "Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action = "store_true", default = False, help = "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type = str, default = "constant", help = ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'))
    parser.add_argument("--lr_warmup_steps", type = int, default = 500, help = "Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--conditioning_dropout_prob", type = float, default = 0.1, help = "Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.")
    parser.add_argument("--use_8bit_adam", action = "store_true", help = "Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--allow_tf32", action = "store_true", help = ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"))
    parser.add_argument("--use_ema", action = "store_true", help = "Whether to use EMA model.")
    parser.add_argument("--non_ema_revision", type = str, default = None, required = False, help = ("Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or remote repository specified with --pretrained_model_name_or_path."))
    parser.add_argument("--num_workers", type = int, default = 8, help = ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process." ))
    parser.add_argument("--adam_beta1", type = float, default = 0.9, help = "The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type = float, default = 0.999, help = "The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type = float, default = 1e-2, help = "Weight decay to use.")
    parser.add_argument("--adam_epsilon", type = float, default = 1e-08, help = "Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default = 1.0, type = float, help = "Max gradient norm.")
    parser.add_argument("--push_to_hub", action = "store_true", help = "Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type = str, default = None, help = "The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type = str, default = None, help = "The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type = str, default = "logs", help = ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))
    parser.add_argument("--mixed_precision", type = str, default = None, choices = ["no", "fp16", "bf16"], help = "Whether to use mixed precision.")
    parser.add_argument("--report_to", type = str, default = "tensorboard", help = "The integration to report the results and logs to. Supported platforms are tensorboard")
    parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type = int, default = 500, help = "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming")
    parser.add_argument("--checkpoints_total_limit", type = int, default = 10, help = ("Max number of checkpoints to store."))
    parser.add_argument("--resume_from_checkpoint", type = str, default = None, help = "Whether training should be resumed from a previous checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action = "store_true", help = "Whether or not to use xformers.")
    parser.add_argument("--pretrain_unet", type = str, default = None, help = "use weight for unet block")
    
    args = parser.parse_args()
    config = load_config("./config.yaml")
    args = merge_args_with_config(args, config)
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    
    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate("non_ema_revision!=None", "0.15.0", message = ("Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to use `--variant=non_ema` instead."))

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir = args.output_dir, logging_dir = logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps, mixed_precision = args.mixed_precision, log_with = args.report_to, project_config = accelerator_project_config)
    
    generator = torch.Generator(device = accelerator.device).manual_seed(args.seed)
    
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt = "%m/%d/%Y %H:%M:%S", level = logging.INFO)
    logger.info(accelerator.state, main_process_only = False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok = True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id = args.hub_model_id or Path(args.output_dir).name, exist_ok = True, token = args.hub_token).repo_id
    
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder = "feature_extractor", revision = args.revision)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder = "image_encoder", revision = args.revision, variant = "fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder = "vae", revision = args.revision, variant = "fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet, subfolder = "unet", low_cpu_mem_usage = False, ignore_mismatched_sizes = True)

    ckpt_path = "./weights/raft-things.pth"
    fix_raft = RAFT_bi(ckpt_path)
    
    ckpt_path = "./weights/recurrent_flow_completion.pth"
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    
    inp_cond_lat_channel = 4
    lat_bi_propagator = LatentPropagation(inp_cond_lat_channel)
    
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    fix_raft.requires_grad_(False)
    fix_flow_complete.requires_grad_(False)
    lat_bi_propagator.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    image_encoder.to(accelerator.device, dtype = weight_dtype)
    vae.to(accelerator.device, dtype = weight_dtype)
    fix_raft.to(accelerator.device, dtype = weight_dtype)
    # fix_flow_complete.to(accelerator.device, dtype = weight_dtype)
    
    flow_loss = FlowLoss()
    
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                sub = _module_subfolder(model)
                save_dir = os.path.join(output_dir, sub)
                os.makedirs(save_dir, exist_ok = True)
                
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(save_dir)
                else:
                    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                
                if len(weights) > 0:
                    weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                sub = _module_subfolder(model)
                load_dir = os.path.join(input_dir, sub)

                if hasattr(model, "from_pretrained"):
                    try:
                        loaded = type(model).from_pretrained(load_dir)
                        model.load_state_dict(loaded.state_dict(), strict = True)
                        del loaded
                        continue
                    except Exception:
                        pass

                bin_path = os.path.join(load_dir, "pytorch_model.bin")
                if os.path.exists(bin_path):
                    sd = torch.load(bin_path, map_location = "cpu")
                    missing, unexpected = model.load_state_dict(sd, strict = False)
                    if missing or unexpected:
                        print(f"[load_model_hook] ({sub}) missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    print(f"[load_model_hook] WARN: {bin_path} not found for {type(model).__name__}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size * accelerator.num_processes)

    optimizer_cls = torch.optim.AdamW

    parameters_list = []
    unet.requires_grad_(True)
    for name, param in unet.named_parameters():
        if 'temporal_transformer_block' in name or 'noise_refiner' in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    lat_bi_propagator.requires_grad_(True)
    for p in lat_bi_propagator.parameters():
        p.requires_grad = True
    parameters_list += list(lat_bi_propagator.parameters())
    
    fix_flow_complete.requires_grad_(True)
    for name, param in fix_flow_complete.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
    
    optimizer = optimizer_cls(parameters_list, lr = args.learning_rate, betas = (args.adam_beta1, args.adam_beta2), weight_decay = args.adam_weight_decay, eps = args.adam_epsilon)
    
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        for name, para in lat_bi_propagator.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        for name, para in fix_flow_complete.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
            
        rec_txt1.close()
        rec_txt2.close()
    
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    
    data_root = args.data_root
    train_dataset = OutpaintSequenceDataset(data_root, width = args.width, height = args.height, mask_ratio = args.one_side_mask_ratio, sample_frames = args.num_frames)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler = sampler, batch_size = args.per_gpu_batch_size, num_workers = args.num_workers)
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer = optimizer, num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes, num_training_steps = args.max_train_steps * accelerator.num_processes)
    
    unet, lat_bi_propagator, fix_flow_complete, flow_loss, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(unet, lat_bi_propagator, fix_flow_complete, flow_loss, optimizer, lr_scheduler, train_dataloader)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config = vars(args))
    
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("#################### Running training ####################")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        pixel_values = feature_extractor(images = pixel_values, do_normalize = True, do_center_crop = False, do_resize = False, do_rescale = False, return_tensors = "pt").pixel_values
        pixel_values = pixel_values.to(device = accelerator.device, dtype = weight_dtype)
        
        image_embeddings = image_encoder(pixel_values).image_embeds
        
        return image_embeddings
    
    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.module.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.")

        add_time_ids = torch.tensor([add_time_ids], dtype = dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        
        return add_time_ids
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key = lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable = not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        lat_bi_propagator.train()
        fix_flow_complete.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet, fix_flow_complete, lat_bi_propagator):
                pixel_values = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking = True)
                mask_values = batch["mask_values"].to(weight_dtype).to(accelerator.device, non_blocking = True)    
                masked_pixel_values = pixel_values * (1 - mask_values)

                gt_flows_bi = fix_raft(pixel_values, iters = 20)    # 1 x 24 x 2 x 256 x 256
                pred_flows_bi, _ = fix_flow_complete.module.forward_bidirect_flow(gt_flows_bi, mask_values)
                pred_flows_bi = fix_flow_complete.module.combine_flow(gt_flows_bi, pred_flows_bi, mask_values) 
                
                # _visualize_flow("./result", pred_flows_bi, gt_flows_bi, global_step)
                
                latents = tensor_to_vae_latent(pixel_values, vae)
                
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                
                cond_sigmas = rand_log_normal(shape = [bsz,], loc = -3.0, scale = 0.5).to(latents)
                noise_aug_strength = cond_sigmas[0]
                cond_sigmas = cond_sigmas[:, None, None, None, None]

                conditional_pixel_values = torch.randn_like(masked_pixel_values) * cond_sigmas + masked_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)
                conditional_latents = conditional_latents / vae.config.scaling_factor

                out_b, out_f, conditional_latents = lat_bi_propagator(orig_lats = latents, cond_lat = conditional_latents, flow_fw = pred_flows_bi[0], flow_bw = pred_flows_bi[1], masks = mask_values)
                
                sigmas = rand_log_normal(shape = [bsz,], loc = 0.7, scale = 1.6).to(latents.device)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = noise * sigmas + latents
                
                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                
                encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].float())

                added_time_ids = _get_add_time_ids(7, 127, noise_aug_strength, encoder_hidden_states.dtype, bsz)
                added_time_ids = added_time_ids.to(latents.device)
                
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device = latents.device, generator = generator)
                    
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))

                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    
                    conditional_latents = image_mask * conditional_latents
                
                inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim = 2)

                target = latents

                model_pred = unet(inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids = added_time_ids).sample
                
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                diff_loss = torch.mean((weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim = 1)
                
                f_loss, warp_loss = flow_loss(pred_flows_bi, gt_flows_bi, mask_values, pixel_values)
                
                loss = diff_loss + f_loss + warp_loss
                
                loss = loss.mean()
                
                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step = global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if ((global_step % args.validation_steps == 0) or (global_step == 1)):
                        logger.info(f"Running validation... \n Generating {args.num_validation_images} videos.")

                        pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet = accelerator.unwrap_model(unet),
                            fix_raft = accelerator.unwrap_model(fix_raft),
                            fix_flow_complete = accelerator.unwrap_model(fix_flow_complete),
                            vae = accelerator.unwrap_model(vae),
                            lat_bi_propagator = accelerator.unwrap_model(lat_bi_propagator),
                            revision = args.revision,
                            torch_dtype = weight_dtype)
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable = True)

                        val_save_dir = os.path.join(args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        sub_folders = [name for name in os.listdir("./demo/mask") if os.path.isdir(os.path.join("./demo/mask", name))]

                        for num_demos in range(len(sub_folders)):
                            demo_paths = []
                            demo_mask_paths = []

                            print(sub_folders[num_demos])

                            for (path, dir, files) in os.walk("./demo/img/{}".format(sub_folders[num_demos])):
                                for filename in files:
                                    ext = os.path.splitext(filename)[-1]
                                    if ext == '.jpg':
                                        demo_paths.append("%s/%s" % (path, filename))

                            for (path, dir, files) in os.walk("./demo/mask/{}".format(sub_folders[num_demos])):
                                for filename in files:
                                    ext = os.path.splitext(filename)[-1]
                                    if ext == '.png':
                                        demo_mask_paths.append("%s/%s" % (path, filename))
                            
                            demo_paths.sort()
                            demo_paths = demo_paths[:args.num_frames]

                            demo_mask_paths.sort()
                            demo_mask_paths = demo_mask_paths[:args.num_frames]

                            demo_imgs = []
                            for demo_path in demo_paths:
                                demo_img = load_image(demo_path)
                                
                                def resize_center_crop(img: Image.Image, target_w: int, target_h: int):
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
    
                                demo_img = resize_center_crop(demo_img, args.width, args.height)
                                
                                demo_imgs.append(demo_img)
                            
                            demo_masks = []
                            for demo_mask_path in demo_mask_paths:
                                demo_mask = load_image(demo_mask_path).resize((args.width, args.height))
                                demo_masks.append(demo_mask)
                        
                            with torch.autocast(str(accelerator.device).replace(":0", ""), enabled = accelerator.mixed_precision == "fp16"):
                                for val_img_idx in range(args.num_validation_images):
                                    num_frames = args.num_frames
                                    video_frames = pipeline(
                                        demo_imgs,
                                        demo_masks,
                                        height = args.height,
                                        width = args.width,
                                        num_frames = num_frames,
                                        decode_chunk_size = 8,
                                        motion_bucket_id = 127,
                                        fps = 7,
                                        noise_aug_strength = 0.02,
                                        g_step = global_step
                                        # generator=generator,
                                    ).frames[0]

                                    out_file = os.path.join(
                                        val_save_dir,
                                        f"{sub_folders[num_demos]}.mp4",
                                    )

                                    for i in range(num_frames):
                                        gt = demo_imgs[i]
                                        mask = demo_masks[i]
                                        mask = mask.convert("RGB")
                                        pred = video_frames[i]

                                        gt = np.array(gt)
                                        mask = np.array(mask)
                                        mask = mask / 255.0
                                        pred = np.array(pred)

                                        masked_in = (gt.astype(np.float32) * (1.0 - mask)).clip(0, 255).astype(np.uint8)
                                        
                                        comp_img = gt * (1.0 - mask) + pred * mask
                                        comp_img = comp_img.astype(np.uint8)
                                        
                                        mask = mask * 255.0
                                        mask = mask.astype(np.uint8)
                                        concat_img = np.concatenate([gt, mask, masked_in, pred, comp_img], axis=1)

                                        video_frames[i] = concat_img
                                    export_to_gif(video_frames, out_file, 15)

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "diff_loss": diff_loss.detach().item(), "f_loss": f_loss.detach().item(), "warp_loss": warp_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
