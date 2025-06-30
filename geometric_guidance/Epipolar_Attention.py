import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange, repeat


def quaternion_to_rotation_matrix(quaternions):
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    batch_size = quaternions.size(0)
    rot = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w], dim=1),
        torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=1),
        torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2], dim=1)
    ], dim=1)
    return rot.view(batch_size, 3, 3)


def compute_skew_symmetric(v):
    batch_size = v.size(0)
    M = torch.zeros((batch_size, 3, 3), device=v.device)
    M[:, 0, 1] = -v[:, 2]
    M[:, 0, 2] =  v[:, 1]
    M[:, 1, 0] =  v[:, 2]
    M[:, 1, 2] = -v[:, 0]
    M[:, 2, 0] = -v[:, 1]
    M[:, 2, 1] =  v[:, 0]
    return M


def compute_fundamental_matrix(K1, K2, R, t):
    t_skew = compute_skew_symmetric(t)
    E = torch.bmm(t_skew, R)
    U, S, Vt = torch.linalg.svd(E)
    S[:, 2] = 0
    E = torch.bmm(U, torch.bmm(torch.diag_embed(S), Vt))
    F = torch.bmm(torch.inverse(K2.transpose(1, 2)), torch.bmm(E, torch.inverse(K1)))
    return F


def calculate_epipolar_lines(points, fundamental_matrices):
    epipolar_lines = torch.bmm(fundamental_matrices, points)
    epipolar_lines = epipolar_lines / epipolar_lines[:, 2:3, :]
    return epipolar_lines


def visualize_attention_map(attention_map, batch_idx=0, column_idx=600, save_path='attention_map_visualization.png'):
    reshaped = attention_map[batch_idx].view(1024, 32, 32)
    grid = reshaped.view(32, 32, 32, 32).permute(0, 2, 1, 3).contiguous().view(32*32, 32*32)
    norm = (grid - grid.min()) / (grid.max() - grid.min()) * 255
    Image.fromarray(norm.cpu().numpy().astype(np.uint8)).save(save_path)
    single = attention_map[batch_idx, :, column_idx].view(32, 32)
    norm_single = (single - single.min()) / (single.max() - single.min()) * 255
    Image.fromarray(norm_single.cpu().numpy().astype(np.uint8)).save(f'single_{save_path}')


class PatchifyAttention(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.avg_pool(x)
        return x.view(B, -1, 1)


class EpipolarAttention(nn.Module):
    def __init__(self, feature_dim, img_height, img_width, patch_size=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.img_height = img_height
        self.img_width = img_width
        self.softmax = nn.Softmax(dim=-1)
        self.patchify = PatchifyAttention(patch_size)

    def forward(self, f_tar, f_src, K1, K2, R, t):
        B, C, H, W = f_src.shape
        f_src_flat = f_src.view(B, C, H*W)
        f_tar_flat = f_tar.view(B, C, H*W)

        A = torch.einsum('bik,bkj->bij', f_src_flat.permute(0,2,1), f_tar_flat)
        A = A.view(B, H*W, H*W)

        idx_x, idx_y = torch.meshgrid(
            torch.arange(H, device=f_src.device, dtype=torch.float32),
            torch.arange(W, device=f_src.device, dtype=torch.float32),
            indexing='ij'
        )
        ones = torch.ones_like(idx_x)
        idx = torch.stack([idx_x.reshape(-1), idx_y.reshape(-1), ones.reshape(-1)], dim=1)
        idx = idx.unsqueeze(0).repeat(B,1,1).permute(0,2,1)

        F_mat = compute_fundamental_matrix(K1, K2, R, t)
        epi_lines = torch.bmm(F_mat, idx) / F_mat.new_tensor([1.0]).unsqueeze(0)

        x0 = torch.zeros(B,1,H*W, device=epi_lines.device)
        y0 = -epi_lines[:,2:3,:] / epi_lines[:,1:2,:]
        p0 = torch.cat([x0, y0, torch.ones_like(x0)], dim=1)

        x1 = torch.full((B,1,H*W), W, device=epi_lines.device)
        y1 = -(epi_lines[:,2:3,:] + epi_lines[:,0:1,:]*W) / epi_lines[:,1:2,:]
        p1 = torch.cat([x1, y1, torch.ones_like(x1)], dim=1)

        d = self.compute_epipolar_distance(idx, p0, p1)
        weight = 1 - self.softmax(5*(d - 0.10))

        attention_map = F.softmax(weight, dim=1)
        f_src_attended = torch.einsum('bik,bkj->bij', attention_map, f_src_flat.permute(0,2,1))
        return f_src_attended.view(B, C, H, W)

    def compute_epipolar_distance(self, p, p0, p1):
        diff = p0 - p1
        p_exp = p.unsqueeze(3)
        p0_exp = p0.unsqueeze(3)
        cross = torch.cross(p_exp - p0_exp, diff.unsqueeze(2), dim=1)
        return torch.norm(cross, dim=1) / torch.norm(diff, dim=1, keepdim=True)
