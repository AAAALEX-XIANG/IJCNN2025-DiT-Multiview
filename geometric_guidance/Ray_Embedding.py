import os
import torch
import numpy as np
from PIL import Image
import cv2
from sklearn.manifold import TSNE

def quaternion_to_rotation_matrix(quaternions):
    """
    Converts a batch of quaternions to rotation matrices.
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    batch_size = quaternions.size(0)
    rot = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,      2*x*z + 2*y*w], dim=1),
        torch.stack([2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=1),
        torch.stack([2*x*z - 2*y*w,       2*y*z + 2*x*w,      1 - 2*x**2 - 2*y**2], dim=1)
    ], dim=1)
    return rot.view(batch_size, 3, 3)

def compute_relative_transform(src_pose, tar_pose):
    return torch.bmm(tar_pose, torch.inverse(src_pose))

def load_resize_image_cv2(image_path, device):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    image = cv2.resize(image, (w // 20, h // 20), interpolation=cv2.INTER_CUBIC)
    tensor = torch.tensor(image, dtype=torch.float32, device=device)
    return tensor.permute(2, 0, 1)

def compute_plucker_coordinates(extrinsic_matrix, intrinsic_matrix, H, W):
    device = extrinsic_matrix.device
    i, j = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    ndc_x = (j - cx) / fx
    ndc_y = (i - cy) / fy
    ndc = torch.stack([ndc_x, ndc_y, torch.ones_like(ndc_x)], dim=-1).reshape(-1, 3)
    extrinsic_inv = torch.inverse(extrinsic_matrix)
    rays_cam = torch.cat([ndc, torch.ones(ndc.shape[0], 1, device=device)], dim=-1) @ extrinsic_inv.T
    origins = rays_cam[:, :3]
    directions = ndc / ndc.norm(dim=-1, keepdim=True)
    normals = torch.cross(origins, directions, dim=-1)
    return torch.cat([directions, normals], dim=-1)

def visualize_plucker_coordinates_tsne(plucker_coordinates, H, W):
    plucker_np = plucker_coordinates.cpu().numpy()
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(plucker_np)
    tsne_min, tsne_max = tsne_result.min(0), tsne_result.max(0)
    tsne_norm = ((tsne_result - tsne_min) / (tsne_max - tsne_min) * 255).astype(np.uint8)
    img = Image.fromarray(tsne_norm.reshape(H, W, 3))
    img.save('plucker_tsne_image.png')
