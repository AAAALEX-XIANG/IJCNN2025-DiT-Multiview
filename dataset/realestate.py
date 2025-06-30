import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import json
from timm.models.vision_transformer import PatchEmbed
from torch import nn
import math
import cv2


class RealEstateFeatureRT(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.features_dir = 'image_features'
        self.scene_paths = [
            os.path.join(root_dir, scene) for scene in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, scene))
        ]
        self.scene_features = []
        self.scene_poses = []

        for scene_path in self.scene_paths:
            pose_path = os.path.join(scene_path, "pose.json")
            with open(pose_path, 'r') as file:
                poses = json.load(file)
            poses = {entry["timestamp"]: entry for entry in poses}
            feature_folder = os.path.join(scene_path, self.features_dir)
            features = sorted(os.listdir(feature_folder))
            scene_features = []
            scene_poses = []
            for feature in features:
                frame = feature.split(".")[0]
                if frame not in poses:
                    continue
                pose = np.array(poses[frame]["pose"])
                loaded_feature = np.load(os.path.join(feature_folder, feature))
                scene_features.append(loaded_feature)
                scene_poses.append(pose)
            self.scene_features.append(scene_features)
            self.scene_poses.append(scene_poses)

    def __len__(self):
        return len(self.scene_features) * 1_000_000

    def __getitem__(self, idx):
        idx = idx % len(self.scene_features)
        features = self.scene_features[idx]
        poses = self.scene_poses[idx]

        if len(features) < 3:
            raise ValueError("Not enough frames in the scene to sample from.")

        idx1 = np.random.randint(0, len(features) - 3)
        interval = np.random.randint(3, 13)
        idx2 = min(idx1 + interval, len(features) - 1)

        feature_1 = features[idx1]
        feature_2 = features[idx2]
        pose_1 = poses[idx1]
        pose_2 = poses[idx2]

        T = pose_2 @ np.linalg.inv(pose_1)
        quaternion, translation = matrix_to_quaternion_translation(T)

        return (
            torch.from_numpy(feature_1),
            torch.from_numpy(feature_2),
            torch.from_numpy(quaternion).float(),
            torch.from_numpy(translation).float()
        )


def matrix_to_quaternion_translation(matrix):
    rotation_matrix = matrix[:3, :3]
    trace = np.trace(rotation_matrix)
    if trace >= 0:
        S = (trace + 1.0) ** 0.5 * 2
        qw = 0.25 * S
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        S = (1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) ** 0.5 * 2
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (rotation_matrix[1, 0] + rotation_matrix[0, 1]) / S
        qz = (rotation_matrix[2, 0] + rotation_matrix[0, 2]) / S
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        S = (1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) ** 0.5 * 2
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qx = (rotation_matrix[1, 0] + rotation_matrix[0, 1]) / S
        qy = 0.25 * S
        qz = (rotation_matrix[2, 1] + rotation_matrix[1, 2]) / S
    else:
        S = (1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) ** 0.5 * 2
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        qx = (rotation_matrix[2, 0] + rotation_matrix[0, 2]) / S
        qy = (rotation_matrix[2, 1] + rotation_matrix[1, 2]) / S
        qz = 0.25 * S

    quaternion = np.array([qw, qx, qy, qz])
    quaternion /= np.linalg.norm(quaternion)
    translation = matrix[:3, 3]

    return quaternion, translation


def depth_to_3d_points_with_colors(depth_map, intrinsic_matrix, image):
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    x = (j - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (i - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map
    points_3D = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = np.transpose(image, (1, 2, 0)).reshape(-1, 4)
    return points_3D, colors


def transform_points_with_colors(points, colors, rotation_matrix, translation_vector):
    points_transformed = points @ rotation_matrix.T + translation_vector
    return np.concatenate((points_transformed, colors), axis=1)


def project_points_with_colors(points_with_colors, intrinsic_matrix):
    points = points_with_colors[..., :3]
    colors = points_with_colors[..., 3:]
    projected = points @ intrinsic_matrix.T
    points_2d = projected[:, :2] / projected[:, 2:3]
    return np.concatenate((points_2d, colors), axis=1)


def populate_image_with_colors(projected_points_2D_colors, H, W):
    image = np.zeros((H, W, 4), dtype=np.float32)
    x_coords = projected_points_2D_colors[..., 0].astype(int)
    y_coords = projected_points_2D_colors[..., 1].astype(int)
    colors = projected_points_2D_colors[..., 2:]
    valid = (
        (x_coords >= 0) & (x_coords < W) &
        (y_coords >= 0) & (y_coords < H)
    )
    image[y_coords[valid], x_coords[valid]] = colors[valid]
    return image


def center_crop_img_and_resize(src_image, image_size):
    while min(src_image.shape[:2]) >= 2 * image_size:
        new_size = (src_image.shape[1] // 2, src_image.shape[0] // 2)
        src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_AREA)
    scale = image_size / min(src_image.shape[:2])
    new_size = (round(src_image.shape[1] * scale), round(src_image.shape[0] * scale))
    src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_CUBIC)
    crop_y = (src_image.shape[0] - image_size) // 2
    crop_x = (src_image.shape[1] - image_size) // 2
    return src_image[crop_y:crop_y + image_size, crop_x:crop_x + image_size]
