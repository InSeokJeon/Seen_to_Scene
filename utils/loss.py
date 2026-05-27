import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

 
class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def flow_warp(self, x, flow, interpolation = 'bilinear', padding_mode = 'zeros', align_corners = True):

        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        device = flow.device
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(x, grid_flow, mode = interpolation, padding_mode = padding_mode, align_corners = align_corners)
        
        return output
    
    def rgb2gray(self, image):
        gray_image = image[:, 0] * 0.299 + image[:, 1] * 0.587 + 0.110 * image[:, 2]
        gray_image = gray_image.unsqueeze(1)
        return gray_image

    def ternary_transform(self, image, max_distance=1):
        device = image.device
        patch_size = 2 * max_distance + 1
        intensities = self.rgb2gray(image) * 255
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
        weights = torch.from_numpy(w).float().to(device)
        patches = F.conv2d(intensities, weights, stride=1, padding=1)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
        
        return transf_norm

    def hamming_distance(self, t1, t2):
        dist = torch.square(t1 - t2)
        dist_norm = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
        return dist_sum

    def ternary_loss2(self, frame1, warp_frame21, confMask, masks, max_distance=1):

        t1 = self.ternary_transform(frame1)
        t21 = self.ternary_transform(warp_frame21)
        dist = self.hamming_distance(t1, t21) 
        loss = torch.mean(dist * confMask * masks) / torch.mean(masks)
        
        return loss

    def ternary_loss(self, flow_comp, flow_gt, mask, current_frame, shift_frame, scale_factor = 1):
        if scale_factor != 1:
            current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode='bilinear')
            shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode='bilinear')
        warped_sc = self.flow_warp(shift_frame, flow_gt.permute(0, 2, 3, 1))
        noc_mask = torch.exp(-50. * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
        warped_comp_sc = self.flow_warp(shift_frame, flow_comp.permute(0, 2, 3, 1))
        loss = self.ternary_loss2(current_frame, warped_comp_sc, noc_mask, mask)
        return loss

    def forward(self, pred_flows, gt_flows, masks, frames):
        loss = 0
        warp_loss = 0
        h, w = pred_flows[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:,:-1,...]
        frames1 = frames[:,1:,...]
        current_frames = [frames0, frames1]
        next_frames = [frames1, frames0]
        
        for i in range(len(pred_flows)):    
            combined_flow = pred_flows[i] * masks[i] + gt_flows[i] * (1-masks[i])
            l1_loss = self.l1_criterion(pred_flows[i] * masks[i], gt_flows[i] * masks[i]) / torch.mean(masks[i])
            l1_loss += self.l1_criterion(pred_flows[i] * (1-masks[i]), gt_flows[i] * (1-masks[i])) / torch.mean((1-masks[i]))

            warp_loss_i = self.ternary_loss(combined_flow.reshape(-1,2,h,w), gt_flows[i].reshape(-1,2,h,w), 
                            masks[i].reshape(-1,1,h,w), current_frames[i].reshape(-1,3,h,w), next_frames[i].reshape(-1,3,h,w)) 

            loss += l1_loss
            warp_loss += warp_loss_i
            
        return loss, warp_loss
    
