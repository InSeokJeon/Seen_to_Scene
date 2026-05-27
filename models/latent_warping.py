import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.deformable_conv import ModulatedDeformConv2d
from utils.util import save_panel_for_frame

def backward_warp(x, flow, align_corners = True):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device = device, dtype = dtype),
        torch.arange(0, W, device = device, dtype = dtype),
        indexing = "ij")
    
    base = torch.stack([grid_x, grid_y], dim = -1)
    base = base.unsqueeze(0).expand(B, H, W, 2)
    
    grid = base + flow.permute(0, 2, 3, 1)
    grid_x = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0
    norm_grid = torch.stack([grid_x, grid_y], dim = -1)

    return F.grid_sample(x, norm_grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = align_corners)


def rescale_flow(flow, target_h, target_w):
    B, _, Hf, Wf = flow.shape
    flow_r = F.interpolate(flow, size = (target_h, target_w), mode = 'bilinear', align_corners = True)
    sx = target_w / max(Wf, 1)
    sy = target_h / max(Hf, 1)
    scale = torch.tensor([sx, sy], device = flow.device, dtype = flow.dtype).view(1, 2, 1, 1)
    
    return flow_r * scale


def compose_flow(curr_flow, acc_flow, warp_fn = backward_warp):
    return acc_flow + warp_fn(curr_flow, acc_flow)


def length_sq(x):
    return (x * x).sum(dim = 1, keepdim = True)


def fb_consistency_mask(flow_fw, flow_bw, warp_fn = backward_warp, alpha1 = 0.01, alpha2 = 0.5):
    bw_warped = warp_fn(flow_bw, flow_fw)
    flow_diff = flow_fw + bw_warped
    mag = length_sq(flow_fw) + length_sq(bw_warped)
    thr = alpha1 * mag + alpha2
    
    return (length_sq(flow_diff) < thr).float()


class _DtypeDeviceMixin:
    @property
    def dtype(self):
        for p in self.parameters(recurse = True):
            return p.dtype
        for b in self.buffers(recurse = True):
            return b.dtype
        return torch.get_default_dtype()

    @property
    def device(self):
        for p in self.parameters(recurse = True):
            return p.device
        for b in self.buffers(recurse = True):
            return b.device
        return torch.device("cpu")
    
    
class DeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)
        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1)
        )
        self.init_offset()
    
    def constant_init(self, module, val, bias = 0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_offset(self):
        self.constant_init(self.conv_offset[-1], val = 0, bias = 0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim = 1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim = 1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


class LatentPropagation(_DtypeDeviceMixin, nn.Module):
    def __init__(self, channels = 4, learnable = True, use_fb_check = True, deform_align_class = DeformableAlignment):
        super().__init__()
        self.C = channels
        self.learnable = learnable
        self.use_fb_check = use_fb_check
        
        if self.learnable:
            self.align_backward = deform_align_class(channels, channels, 3, deform_groups = 4)
            self.align_forward  = deform_align_class(channels, channels, 3, deform_groups = 4)

            self.refine_backward = nn.Sequential(
                nn.Conv2d(2*channels + 1, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )
            self.refine_forward = nn.Sequential(
                nn.Conv2d(2*channels + 1, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(2*channels + 1, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )
            
        self._grid_cache = {}
        
    def _norm_base_grid(self, device, dtype, H, W):
        key = (device, dtype, H, W)
        cached = self._grid_cache.get(key, None)
        if cached is not None:
            return cached
        
        yy, xx = torch.meshgrid(
            torch.arange(H, device = device, dtype = dtype),
            torch.arange(W, device = device, dtype = dtype),
            indexing = "ij"
        )
        gx = 2.0 * xx / max(W - 1, 1) - 1.0
        gy = 2.0 * yy / max(H - 1, 1) - 1.0
        base = torch.stack([gx, gy], dim = -1).unsqueeze(0)
        self._grid_cache[key] = base
        
        return base

    def backward_warp_cached(self, x, flow, align_corners = True):
        B, C, H, W = x.shape
        base = self._norm_base_grid(x.device, x.dtype, H, W)
        base.requires_grad =False
        
        fx = 2.0 * flow[:, 0] / max(W - 1, 1)
        fy = 2.0 * flow[:, 1] / max(H - 1, 1)
        grid = torch.stack([fx, fy], dim = -1) + base
        
        return F.grid_sample(x, grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = align_corners)
    
    def forward(self, cond_lat, flow_fw, flow_bw, masks, flow_pairs_info = None):        
        B, T, C, H, W = cond_lat.shape      # 1 x 25 x 4 x 32 x 32

        batch_size_mask = masks.shape[0]
        if B != batch_size_mask:
            masks = masks.repeat(B, 1, 1, 1, 1)
            flow_fw = flow_fw.repeat(B, 1, 1, 1, 1)
            flow_bw = flow_bw.repeat(B, 1, 1, 1, 1)
        
        masks = rearrange(masks, "b t c h w -> (b t) c h w")
        masks = F.interpolate(masks, size = cond_lat.shape[-2:], mode = "nearest")
        masks = rearrange(masks, "(b t) c h w -> b t c h w", b = B)
        masks = (masks != 0).float()        # 1 x 25 x 1 x 32 x 32
        cnts  = 1.0 - masks                 # 1 x 25 x 1 x 32 x 32
 
        flow_fw_r = torch.stack([rescale_flow(flow_fw[:, k], H, W) for k in range(T - 1)], dim = 1)     # 1 x 24 x 2 x 32 x 32
        flow_bw_r = torch.stack([rescale_flow(flow_bw[:, k], H, W) for k in range(T - 1)], dim = 1)     # 1 x 24 x 2 x 32 x 32
        
        if flow_pairs_info is None:
            print("Propagation with the All Frames")
            return self._forward_sequential(cond_lat, masks, cnts, flow_fw_r, flow_bw_r)
        
        else:
            print("Propagation with the Reference Frames")
            return self._forward_reference_pairs(cond_lat, masks, cnts, flow_fw_r, flow_bw_r, flow_pairs_info)


    def _forward_sequential(self, cond_lat, masks, cnts, flow_fw_r, flow_bw_r):
        B, T, C, H, W = cond_lat.shape

        fw_cond_lats = cond_lat.clone()
        bw_cond_lats = cond_lat.clone()
        fw_cnts  = cnts.clone()
        bw_cnts  = cnts.clone()
 
        fw_final_acc_flow = torch.zeros(B, T, 2, H, W, device = cond_lat.device, dtype = cond_lat.dtype)
        bw_final_acc_flow = torch.zeros(B, T, 2, H, W, device = cond_lat.device, dtype = cond_lat.dtype)

        for i in range(T):    
            for j in range(i + 1, T):
                if j == i + 1:
                    acc_flow = flow_fw_r[:, j - 1]
                else:
                    acc_flow = compose_flow(flow_fw_r[:, j - 1], acc_flow, self.backward_warp_cached)
                
                warp_lat = self.backward_warp_cached(cond_lat[:, j], acc_flow)
                warp_cnt = self.backward_warp_cached(cnts[:, j], acc_flow)
                
                if self.use_fb_check:
                    for k in range(j, i, -1):
                        if k == j:
                            acc_flow_check = flow_bw_r[:, k - 1]
                        else:
                            acc_flow_check = compose_flow(flow_bw_r[:, k - 1], acc_flow_check, self.backward_warp_cached)
                            
                    flow_valid_mask = fb_consistency_mask(acc_flow, acc_flow_check, self.backward_warp_cached)       
                
                    warp_lat = warp_lat * flow_valid_mask
                    warp_cnt = warp_cnt * flow_valid_mask
                
                fw_cond_lats[:, i] = fw_cond_lats[:, i] + (1 - fw_cnts[:, i]) * warp_lat
                fw_cnts[:, i] = fw_cnts[:, i] + (1 - fw_cnts[:, i]) * warp_cnt
            fw_final_acc_flow[:, i] = acc_flow
                            
            for j in range(i - 1, -1, -1):
                if j == i - 1:
                    acc_flow = flow_bw_r[:, j]
                else:
                    acc_flow = compose_flow(flow_bw_r[:, j], acc_flow, self.backward_warp_cached)
                    
                warp_lat = self.backward_warp_cached(cond_lat[:, j], acc_flow)
                warp_cnt = self.backward_warp_cached(cnts[:, j], acc_flow)
                
                if self.use_fb_check:
                    for k in range(j, i):
                        if k == j:
                            acc_flow_check = flow_fw_r[:, k]
                        else:
                            acc_flow_check = compose_flow(flow_fw_r[:, k], acc_flow_check, self.backward_warp_cached)
                            
                    flow_valid_mask = fb_consistency_mask(acc_flow, acc_flow_check, self.backward_warp_cached)       
                
                    warp_lat = warp_lat * flow_valid_mask
                    warp_cnt = warp_cnt * flow_valid_mask
                    
                bw_cond_lats[:, i] = bw_cond_lats[:, i] + (1 - bw_cnts[:, i]) * warp_lat
                bw_cnts[:, i] = bw_cnts[:, i] + (1 - bw_cnts[:, i]) * warp_cnt
            bw_final_acc_flow[:, i] = acc_flow

        return self._post_fuse(cond_lat, masks, fw_cond_lats, bw_cond_lats, fw_cnts, bw_cnts, fw_final_acc_flow, bw_final_acc_flow)
        
        
    def _forward_reference_pairs(self, cond_lat, masks, cnts, flow_fw_r, flow_bw_r, flow_pairs_info):
        B, T, C, H, W = cond_lat.shape
        K = flow_fw_r.shape[1]
        assert K == len(flow_pairs_info), \
            f"flow_pairs_info length {len(flow_pairs_info)} != flows length {K}"
    
        device = cond_lat.device
        dtype  = cond_lat.dtype

        parent_idx = [-1] * T
        parent_k = [-1] * T
    
        for k, (s, t) in enumerate(flow_pairs_info):
            assert 0 <= s < T and 0 <= t < T, f"pair ({s},{t}) out of range"
            if parent_idx[t] != -1:
                raise ValueError(f"Frame {t} has multiple parents in flow_pairs_info.")
            parent_idx[t] = s
            parent_k[t] = k
    
        fw_cond_lats = cond_lat.clone()
        bw_cond_lats = cond_lat.clone()
        fw_cnts = cnts.clone()
        bw_cnts = cnts.clone()
    
        fw_final_acc_flow = torch.zeros(B, T, 2, H, W, device = device, dtype = dtype)
        bw_final_acc_flow = torch.zeros(B, T, 2, H, W, device = device, dtype = dtype)
    
        for i in range(T):
            acc_flow_fwd = None 
            acc_flow_bwd = None 
    
            curr = i
            s = parent_idx[curr]
            
            while s != -1:
                k = parent_k[curr]
                flow_st = flow_fw_r[:, k]
    
                if acc_flow_fwd is None:
                    acc_flow_fwd = flow_st
                else:
                    acc_flow_fwd = compose_flow(flow_st, acc_flow_fwd, self.backward_warp_cached)
    
                warp_lat = self.backward_warp_cached(cond_lat[:, s], acc_flow_fwd)
                warp_cnt = self.backward_warp_cached(cnts[:, s], acc_flow_fwd)
    
                if self.use_fb_check:
                    flow_ts = flow_bw_r[:, k]
                    if acc_flow_bwd is None:
                        acc_flow_bwd = flow_ts
                    else:
                        acc_flow_bwd = compose_flow(flow_ts, acc_flow_bwd, self.backward_warp_cached)
    
                    flow_valid_mask = fb_consistency_mask(acc_flow_fwd, acc_flow_bwd, self.backward_warp_cached)
                    warp_lat = warp_lat * flow_valid_mask
                    warp_cnt = warp_cnt * flow_valid_mask
    
                fw_cond_lats[:, i] = fw_cond_lats[:, i] + (1.0 - fw_cnts[:, i]) * warp_lat
                fw_cnts[:, i] = fw_cnts[:, i] + (1.0 - fw_cnts[:, i]) * warp_cnt
                fw_final_acc_flow[:, i] = acc_flow_fwd
    
                curr = s
                s = parent_idx[curr]
    
        for i in range(T):
            acc_flow_fwd = None
            acc_flow_bwd = None
            
            curr = i
            p = parent_idx[curr]
            
            while p != -1:
                k = parent_k[curr]
                flow_tp = flow_bw_r[:, k]
    
                if acc_flow_fwd is None:
                    acc_flow_fwd = flow_tp
                else:
                    acc_flow_fwd = compose_flow(flow_tp, acc_flow_fwd, self.backward_warp_cached)
    
                warp_lat = self.backward_warp_cached(cond_lat[:, p], acc_flow_fwd)
                warp_cnt = self.backward_warp_cached(cnts[:, p], acc_flow_fwd)
    
                if self.use_fb_check:
                    flow_pt = flow_fw_r[:, k]
                    if acc_flow_bwd is None:
                        acc_flow_bwd = flow_pt
                    else:
                        acc_flow_bwd = compose_flow(flow_pt, acc_flow_bwd, self.backward_warp_cached)
    
                    flow_valid_mask = fb_consistency_mask(acc_flow_fwd, acc_flow_bwd, self.backward_warp_cached)
                    warp_lat = warp_lat * flow_valid_mask
                    warp_cnt = warp_cnt * flow_valid_mask
    
                bw_cond_lats[:, i] = bw_cond_lats[:, i] + (1.0 - bw_cnts[:, i]) * warp_lat
                bw_cnts[:, i] = bw_cnts[:, i] + (1.0 - bw_cnts[:, i]) * warp_cnt
                bw_final_acc_flow[:, i] = acc_flow_fwd
    
                curr = p
                p = parent_idx[curr]
    
        return self._post_fuse(cond_lat, masks, fw_cond_lats, bw_cond_lats, fw_cnts, bw_cnts, fw_final_acc_flow, bw_final_acc_flow)
        
    
    def _post_fuse(self, cond_lat, masks, fw_cond_lats, bw_cond_lats, fw_cnts, bw_cnts, fw_final_acc_flow, bw_final_acc_flow):
        B, T, C, H, W = cond_lat.shape

        cond_lats_ = rearrange(cond_lat, "b t c h w -> (b t) c h w")
        masks_ = rearrange(masks, "b t 1 h w -> (b t) 1 h w")
        warped_lats_fw_ = rearrange(fw_cond_lats, "b t c h w -> (b t) c h w")
        warped_lats_bw_ = rearrange(bw_cond_lats, "b t c h w -> (b t) c h w")
        fw_cnts_ = rearrange(fw_cnts, "b t 1 h w -> (b t) 1 h w")
        bw_cnts_ = rearrange(bw_cnts, "b t 1 h w -> (b t) 1 h w")
        fw_final_acc_flow_ = rearrange(fw_final_acc_flow, "b t c h w -> (b t) c h w")
        bw_final_acc_flow_ = rearrange(bw_final_acc_flow, "b t c h w -> (b t) c h w")

        cond_f = torch.cat([cond_lats_, warped_lats_fw_, fw_final_acc_flow_, fw_cnts_, masks_], dim = 1)
        cond_b = torch.cat([cond_lats_, warped_lats_bw_, bw_final_acc_flow_, bw_cnts_, masks_], dim = 1)

        lats_fw_aligned = self.align_forward(cond_lats_, cond_f, fw_final_acc_flow_)
        # lats_fw_aligned = self.align_forward(cond_lats_, cond_f, warped_lats_fw_)
        
        lats_bw_aligned = self.align_backward(cond_lats_, cond_b, bw_final_acc_flow_)
        # lats_bw_aligned = self.align_backward(cond_lats_, cond_b, warped_lats_bw_)

        rf = self.refine_forward(torch.cat([cond_lats_, lats_fw_aligned, masks_], dim = 1))
        rb = self.refine_backward(torch.cat([cond_lats_, lats_bw_aligned, masks_], dim = 1))

        out_f = lats_fw_aligned + rf
        out_b = lats_bw_aligned + rb
        
        # out_f = self.fw_conv_in(out_f)
        # out_b = self.bw_conv_in(out_b)
        
        mask_in = masks_.view(-1, 1, H, W)
        outputs = self.fuse(torch.cat([out_b, out_f, mask_in], dim = 1)) + cond_lats_

        out   = rearrange(outputs, "(b t) c h w -> b t c h w", b = B)
        out_f = rearrange(out_f, "(b t) c h w -> b t c h w", b = B)
        out_b = rearrange(out_b, "(b t) c h w -> b t c h w", b = B)
        
        return out_b, out_f, out