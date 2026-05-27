import os
import cv2 

import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

def create_video_outpainting_mask(video_length, imageHeight = 240, imageWidth = 432, mask_ratio = 0.33):
        r = float(mask_ratio)
        r = max(0.0, min(r, 0.5))
        
        left_idx = int(round(r * imageWidth))
        right_idx = int(round((1.0 - r) * imageWidth))
        
        masks = []
        
        for _ in range(video_length):
            arr = np.zeros((imageHeight, imageWidth), dtype = np.uint8)

            if left_idx > 0:
                arr[:, :left_idx] = 255
            if right_idx < imageWidth:
                arr[:, right_idx:] = 255

            masks.append(Image.fromarray(arr, mode = 'L'))
            
        return masks
    
    
def rand_log_normal(shape, loc = 0., scale = 1., device = 'cpu', dtype = torch.float32):
    u = torch.rand(shape, dtype = dtype, device = device) * (1 - 2e-7) + 1e-7
    
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def _resize_with_antialiasing(input, size, interpolation = "bicubic", align_corners = True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(input, size = size, mode = interpolation, align_corners = align_corners)
    
    return output


def _compute_padding(kernel_size):
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)

    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    
    tmp_kernel = kernel[:, None, ...].to(device = input.device, dtype = input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode = "reflect")

    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    output = torch.nn.functional.conv2d(input, tmp_kernel, groups = tmp_kernel.size(0), padding = 0, stride = 1)
    out = output.view(b, c, h, w)
    
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device = sigma.device, dtype = sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim = True)


def _gaussian_blur2d(input, kernel_size, sigma):
    
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype = input.dtype)
    else:
        sigma = sigma.to(dtype = input.dtype)

    bs = sigma.shape[0]
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'), format = 'GIF', append_images = pil_frames[1:], save_all = True, duration = 500, loop = 0)


import numpy as np

def _make_colorwheel_cw():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY); col += RY
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255; col += YG
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC); col += GC
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255; col += CB
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM); col += BM
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    
    return colorwheel


def _flow_uv_to_colors_cw(u, v, convert_to_bgr=False):
    H, W = u.shape
    flow_image = np.zeros((H, W, 3), np.uint8)
    colorwheel = _make_colorwheel_cw()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(u*u + v*v)
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1); k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
        
    return flow_image


def flow_to_color_cw(flow_uv, convert_to_bgr = False):
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    u = flow_uv[:, :, 0]; v = flow_uv[:, :, 1]
    rad = np.sqrt(u*u + v*v)
    rad_max = np.max(rad)
    eps = 1e-5
    u = u / (rad_max + eps)
    v = v / (rad_max + eps)
    
    return _flow_uv_to_colors_cw(u, v, convert_to_bgr = convert_to_bgr)


def flow_to_numpy(flow_tensor):
    if flow_tensor.dim() == 4:
        flow_tensor = flow_tensor[0]
    flow = flow_tensor.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
    
    return flow


def _visualize_flow(output_root, fcnet_comp_flows, gt, global_step): 
    K = 24
    vis_flow_root = os.path.join(output_root, "completed_flow", str(global_step)) 
    os.makedirs(vis_flow_root, exist_ok = True) 
    
    for t in range(K): 
        
        f_fw = fcnet_comp_flows[0][:, t] 
        f_bw = fcnet_comp_flows[1][:, t] 
        
        g_fw = gt[0][:, t]
        g_bw = gt[1][:, t]
        
        f_fw = flow_to_color_cw(flow_to_numpy(f_fw), convert_to_bgr = True) 
        f_bw = flow_to_color_cw(flow_to_numpy(f_bw), convert_to_bgr = True)
        
        g_fw = flow_to_color_cw(flow_to_numpy(g_fw), convert_to_bgr = True) 
        g_bw = flow_to_color_cw(flow_to_numpy(g_bw), convert_to_bgr = True) 
        
        cv2.imwrite(os.path.join(vis_flow_root, f"{t:05d}_to_{t+1:05d}_complete_fw.png"), f_fw) 
        cv2.imwrite(os.path.join(vis_flow_root, f"{t+1:05d}_to_{t:05d}_complete_bw.png"), f_bw) 
        cv2.imwrite(os.path.join(vis_flow_root, f"{t:05d}_to_{t+1:05d}_gt_fw.png"), g_fw) 
        cv2.imwrite(os.path.join(vis_flow_root, f"{t+1:05d}_to_{t:05d}_gt_bw.png"), g_bw) 
        
def latent_to_rgb(lat, method = "pca"):
    assert lat.ndim == 3, f"expect [C,H,W], got {lat.shape}"
    C, H, W = lat.shape
    x = lat.detach().float().clone()

    x = x.view(C, -1)
    x = x - x.mean(dim = 1, keepdim = True)

    if method == "pca" and C >= 3:
        cov = (x @ x.t()) / (x.shape[1] - 1 + 1e-6)
        evals, evecs = torch.linalg.eigh(cov)
        proj = (evecs[:, -3:].t() @ x).view(3, H, W)
        rgb = proj
    else:
        if C < 3:
            pad = torch.zeros(3 - C, H, W, device=lat.device, dtype=lat.dtype)
            rgb = torch.cat([lat, pad], dim=0)
        else:
            rgb = lat[:3]

    rgb = rgb.clone()
    for c in range(3):
        vmin = rgb[c].min()
        vmax = rgb[c].max()
        if float(vmax - vmin) < 1e-8:
            rgb[c].zero_()
        else:
            rgb[c] = (rgb[c] - vmin) / (vmax - vmin + 1e-8)
    rgb = (rgb.permute(1,2,0).cpu().clamp(0,1).numpy() * 255.0).astype("uint8")
    
    return rgb


def map01(t):
    t = t.detach().float()
    tmin = t.amin(dim = (-2,-1), keepdim = True)
    tmax = t.amax(dim = (-2,-1), keepdim = True)
    
    return (t - tmin) / (tmax - tmin + 1e-8)

def tensor_to_gray_image(t):
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    t = map01(t)
    return (t.cpu().numpy() * 255.0).astype("uint8")

def save_panel_for_frame(
    save_path,
    x_i,              
    warped_fw_i,     
    warped_bw_i,  
    fw_w_i,  
    bw_w_i,   
    mask_i,              
    pulled_mask_i=None 
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    im_orig = latent_to_rgb(x_i)
    im_fw   = latent_to_rgb(warped_fw_i)
    im_bw   = latent_to_rgb(warped_bw_i)

    diff_fw = (x_i - warped_fw_i).abs().sum(dim = 0, keepdim = True)  # [1,H,W]
    diff_bw = (x_i - warped_bw_i).abs().sum(dim = 0, keepdim = True)

    im_fw_w   = tensor_to_gray_image(fw_w_i)
    im_bw_w   = tensor_to_gray_image(bw_w_i)
    im_mask   = tensor_to_gray_image(mask_i)
    im_dff_fw = tensor_to_gray_image(diff_fw)
    im_dff_bw = tensor_to_gray_image(diff_bw)
    if pulled_mask_i is not None:
        im_pulled = tensor_to_gray_image(pulled_mask_i)
    else:
        im_pulled = None

    cols = 5 if im_pulled is not None else 4
    fig, axs = plt.subplots(2, cols, figsize=(3.5*cols, 7))

    axs[0,0].imshow(im_orig); axs[0,0].set_title("orig latent (RGB)"); axs[0,0].axis('off')
    axs[0,1].imshow(im_fw);   axs[0,1].set_title("warped from future (RGB)"); axs[0,1].axis('off')
    axs[0,2].imshow(im_bw);   axs[0,2].set_title("warped from past (RGB)");   axs[0,2].axis('off')
    axs[0,3].imshow(im_mask, cmap='gray'); axs[0,3].set_title("mask (1=hole)"); axs[0,3].axis('off')
    if cols == 5:
        axs[0,4].imshow(im_pulled, cmap='gray'); axs[0,4].set_title("pulled mask (1=uncovered)"); axs[0,4].axis('off')

    axs[1,0].imshow(im_dff_fw, cmap='inferno'); axs[1,0].set_title("|orig - fw| (L1)"); axs[1,0].axis('off')
    axs[1,1].imshow(im_dff_bw, cmap='inferno'); axs[1,1].set_title("|orig - bw| (L1)"); axs[1,1].axis('off')
    axs[1,2].imshow(im_fw_w, cmap='viridis');   axs[1,2].set_title("fw weight"); axs[1,2].axis('off')
    axs[1,3].imshow(im_bw_w, cmap='viridis');   axs[1,3].set_title("bw weight"); axs[1,3].axis('off')
    if cols == 5:
        axs[1,4].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


    # idxs = [1, T//2, T-2]
    # for i in idxs:
    #     save_panel_for_frame(
    #     save_path=f"./debug_latent/i_{i:02d}.png",
    #     x_i=orig_lat[:, i][0].detach().cpu(),                                   # [C,H,W]
    #     warped_fw_i=warped_lats_fw[:, i][0].detach().cpu(),              # [C,H,W]
    #     warped_bw_i=warped_lats_bw[:, i][0].detach().cpu(),              # [C,H,W]
    #     fw_w_i=fw_w[:, i][0].detach().cpu(),                             # [1,H,W]
    #     bw_w_i=bw_w[:, i][0].detach().cpu(),                             # [1,H,W]
    #     mask_i=masks[:, i][0].detach().cpu(),                            # [1,H,W]
    #     pulled_mask_i=pulled_masks[:, i][0].detach().cpu() if 'pulled_masks' in locals() else None
    # )
    
    
