from __future__ import print_function
import os
import argparse
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, float32, int64

@jit(float32[:](float32[:,:], float32[:,:]), nogil=True, nopython=True)
def compute_criteria(loc_translation, map_translation):
    num_velo = loc_translation.shape[0]
    num_map_poses = map_translation.shape[0]
    criteria = np.zeros(num_velo, dtype=np.float32)
    for i in range(num_velo):
        loc_translation_i = loc_translation[i]
        for j in range(num_map_poses):
            map_translation_j = map_translation[j]
            dist_ij = (loc_translation_i[0] - map_translation_j[0]) ** 2 + \
                      (loc_translation_i[1] - map_translation_j[1]) ** 2 + \
                      (loc_translation_i[2] - map_translation_j[2]) ** 2
            if dist_ij <= 5 ** 2:
                criteria[i] = 1
                break
    return criteria

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def transform_point_cloud_numpy(point_cloud, rotation, translation):
    rot_mat = rotation
    return rot_mat @ (point_cloud.T) + translation

axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

def draw_point_cloud(ax, title, velo_frame, color, label='', point_size=0.05, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=color, label=label)
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)
    if zlim3d != None:
        ax.set_zlim3d(zlim3d)

def compare_velodyne_scan(src_velo, target_velo, transformed_src_velo, transformed_src_gt_velo, save_dir,
                          save_fname_postfix, transformed_src_kp, target_kp, points=0.2):

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    src_velo_range = range(0, src_velo.shape[0], points_step)  # Nx(3+C)
    src_velo_frame = src_velo[src_velo_range, :]
    target_velo_range = range(0, target_velo.shape[0], points_step)  # Nx(3+C)
    target_velo_frame = target_velo[target_velo_range, :]
    transformed_src_range = range(0, transformed_src_velo.shape[0], points_step)  # Nx(3+C)
    transformed_src_frame = transformed_src_velo[transformed_src_range, :]
    transformed_src_gt_range = range(0, transformed_src_gt_velo.shape[0], points_step)  # Nx(3+C)
    transformed_src_gt_frame = transformed_src_gt_velo[transformed_src_gt_range, :]

    # print(transformed_src_frame)

    # Draw point cloud data as 3D plot
    f1 = plt.figure(figsize=(15, 8))
    ax2 = f1.add_subplot(111, projection='3d')
    # draw_point_cloud(ax2, 'Source Velodyne scan', src_velo_frame, 'r', xlim3d=(-10, 30), point_size=0.2)
    draw_point_cloud(ax2, 'Velodyne scan', target_velo_frame, 'g', label='Target', xlim3d=(-10, 30), point_size=0.2)
    draw_point_cloud(ax2, 'Velodyne scan', transformed_src_frame, 'b', label='Transformed src', xlim3d=(-10, 30), point_size=0.2)
    # draw_point_cloud(ax2, 'Velodyne scan', transformed_src_gt_frame, 'k', xlim3d=(-10, 30), point_size=0.2)
    if transformed_src_kp is not None:
        ax2.scatter(*np.transpose(transformed_src_kp[:, [0,1,2]]), s=10, c='k', label='transformed src kp')
    if target_kp is not None:
        ax2.scatter(*np.transpose(target_kp[:, [0,1,2]]), s=10, c='r', label='target kp')
    save_fname = "{}/{}".format(save_dir, save_fname_postfix)
    f1.savefig(save_fname)
    # plt.show()

    # Draw point cloud data as plane projections
    f2, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        ax3[0],
        'Source vs Target',
        src_velo_frame,
        'r',
        axes=[0, 1]  # X and Z axes
    )
    draw_point_cloud(
        ax3[0],
        'Source vs Target',
        target_velo_frame,
        'g',
        axes=[0, 1]  # X and Z axes
    )
    draw_point_cloud(
        ax3[1],
        'Target vs Predicted',
        target_velo_frame,
        'g',
        axes=[0, 1]  # X and Y axes
    )
    draw_point_cloud(
        ax3[1],
        'Target vs Predicted',
        transformed_src_frame,
        'b',
        axes=[0, 1]  # X and Y axes
    )
    draw_point_cloud(
        ax3[2],
        'Target vs Ground Truth Transformed',
        target_velo_frame,
        'g',
        axes=[0, 1]  # Y and Z axes
    )
    draw_point_cloud(
        ax3[2],
        'Target vs Ground Truth Transformed',
        transformed_src_gt_frame,
        'k',
        axes=[0, 1]  # Y and Z axes
    )
    # plt.show()
    plane_save_fname = "{}/plane_{}".format(save_dir, save_fname_postfix)
    f2.savefig(plane_save_fname)

    return f1, f2