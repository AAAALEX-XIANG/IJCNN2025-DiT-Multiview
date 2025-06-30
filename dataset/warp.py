import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternions
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x**2 - 2*z**2,2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def compute_relative_transform(src_pose: np.ndarray, tar_pose: np.ndarray) -> np.ndarray:
    return tar_pose @ np.linalg.inv(src_pose)

def depth_to_3d_points_with_colors(depth_map: np.ndarray, K: np.ndarray, image: np.ndarray):
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    x = (j - K[0, 2]) * depth_map / K[0, 0]
    y = (i - K[1, 2]) * depth_map / K[1, 1]
    z = depth_map
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = image.transpose(1, 2, 0).reshape(-1, image.shape[0])
    return points, colors

def project_points_with_colors(points_with_colors: np.ndarray, K: np.ndarray):
    pts = points_with_colors[:, :3]
    cols = points_with_colors[:, 3:]
    proj = pts @ K.T
    uv = proj[:, :2] / proj[:, 2:3]
    return np.concatenate((uv, cols), axis=1)

def populate_image_with_colors(p2d_color: np.ndarray, H: int, W: int) -> np.ndarray:
    img = np.zeros((H, W, p2d_color.shape[1] - 2), dtype=np.float32)
    x, y, cols = p2d_color[:, 0].astype(int), p2d_color[:, 1].astype(int), p2d_color[:, 2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    img[y[mask], x[mask]] = cols[mask]
    return img

def warp_image(src_image: np.ndarray, tgt_uv: np.ndarray) -> np.ndarray:
    C, H, W = src_image.shape
    map_x = ((tgt_uv[:, 0].reshape(H, W) + 1) * (W - 1) / 2).astype(np.float32)
    map_y = ((tgt_uv[:, 1].reshape(H, W) + 1) * (H - 1) / 2).astype(np.float32)
    src = src_image.transpose(1, 2, 0)
    warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped.transpose(2, 0, 1)

def center_crop_img_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    while min(img.shape[:2]) >= 2 * size:
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)
    scale = size / min(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    y, x = (img.shape[0] - size)//2, (img.shape[1] - size)//2
    return img[y:y+size, x:x+size]

def load_resize_image_cv2(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path).astype(np.float32)
    return img

def calculate_valid_pixel_ratio(masks: np.ndarray) -> float:
    m = masks[:, :, 0]
    return np.sum(m > 0) / (m.shape[0] * m.shape[1])

def save_depth_map(depth_map: np.ndarray, save_path: str):
    Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8)).save(save_path)

def project_and_save_tsne_image(src_features: np.ndarray, output_path='tsne_src_viz.png') -> np.ndarray:
    if src_features.ndim == 4:
        src = src_features[0]
    else:
        src = src_features
    C, H, W = src.shape
    flat = src.reshape(C, -1).T
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    res = tsne.fit_transform(flat).reshape(H, W, 3)
    norm = (res - res.min()) / (res.max() - res.min()) * 255
    img = norm.astype(np.uint8)
    cv2.imwrite(output_path, img)
    return img
