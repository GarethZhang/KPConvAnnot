''' This Python script validates Applanix localization performance both qualitatively and quantitatively by using GPS
data and localization poses captured by Applanix pipeline. Localization poses can be divided into poses with/without
learned labelling process

Assumptions are made regarding the directory structure and file formats. Please see following:

DATA_DIR/
|-- LOG_DIR/
|   |
|   |-- applanix_localization/
|       |-- map_poses.txt
|       |-- transformStampedData.csv (poses from Applanix with learned labelling process to remove dynamic points)
|   |-- applanix/
|       |-- sbet.out
|...

map_poses.npy records poses in $T_map_velo$ with 13 numbers (1 timestamp + 12 (3x4) pose matrix).

'''

import os
import cv2
import argparse
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
# from evtk.hl import pointsToVTK
from utils.ply import read_ply
from tqdm import tqdm

class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # current working directory
    scan_dir = ''

    movie_dir = ''

    pose_fname = ''

    keywords = ''

    channel = 'pre'

    save_fname = ''

    every_n = 10

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
axes_limits = [
    [-50, 50], # X axis range
    [-50, 50], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

def map_color(color):
    color_list = []
    for i in range(color.shape[0]):
        if color[i] == 0:
            color_list.append('purple')
        elif color[i] == 1:
            color_list.append('blue')
        elif color[i] == 2:
            color_list.append('green')
        elif color[i] == 3:
            color_list.append('yellow')
        elif color[i] == 4:
            color_list.append('red')
        else:
            assert False, 'color not found'
    return color_list

def draw_point_cloud(velo_frame, ax, title, points=0.2, axes=[0, 1, 2], xlim2d=None, ylim2d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    point_size = 2.0
    velo_color = map_color(velo_frame[::points_step, 3])
    ax.scatter(*np.transpose(velo_frame[::points_step, axes]), s=point_size, c=velo_color)
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))

    ax.set_xlim(xlim2d)
    ax.set_ylim(ylim2d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', type=str, default='', metavar='N',
                        help='Directory that stores all valid scans with annotations')
    parser.add_argument('--movie_dir', type=str, default='', metavar='N',
                        help='Directory to store images and generated movie. Default: scan_dir/../movie')
    parser.add_argument('--pose', type=str, default='', metavar='N',
                        help='Absolute path to pose file that stores absolute pose from scan to map. '
                             'If empty, means scan is zero-centered.')
    parser.add_argument('--keywords', type=str, default='', metavar='N',
                        help='Only files including keywords would be used to generate video')
    parser.add_argument('--channel', type=str, default='', metavar='N',
                        help='Channel name. Default: pre')
    parser.add_argument('--save_fname', type=str, default='', metavar='N',
                        help='Relative save filename.')
    args = parser.parse_args()

    config = Config()

    if args.scan_dir == '':
        assert False, 'Scan directory cannot be empty'
    else:
        print('Scan directory found {:s}'.format(args.scan_dir))
        config.scan_dir = args.scan_dir

    if args.movie_dir == '':
        movie_dir = '/'.join(config.scan_dir.split('/')[:-1])
        print('Set default movie directory to {:s}'.format(movie_dir))
        config.movie_dir = movie_dir
    else:
        print('Movie directory found {:s}'.format(args.movie_dir))
        config.movie_dir = args.movie_dir

    if args.pose == '':
        print('No poses required to generate self-centered images')
    else:
        print('Pose file found {:s}'.format(args.pose))
        config.pose_fname = args.pose

    if args.keywords == '':
        print('No keywords required')
    else:
        print('Use keywords: {:s}'.format(args.keywords))
        config.keywords = args.keywords

    if args.channel != '':
        print('Channel name: {:s}'.format(args.channel))
        config.channel = args.channel

    if args.save_fname != '':
        print('Save filename: {:s}'.format(args.save_fname))
        config.save_fname = args.save_fname
    else:
        config.save_fname = 'video'


    movie_dir = config.movie_dir
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    # loop through each day to generate movie
    time_statistics = {
        'read_ply': 0.0,
        'draw_ply': 0.0,
        'save_fig': 0.0,
    }

    scan_fnames = [scan_fname for scan_fname in sorted(os.listdir(config.scan_dir)) if config.keywords in scan_fname]
    day_movie_dir = config.movie_dir
    day_poses_fname = config.pose_fname

    if day_poses_fname != '':
        day_poses_raw = np.loadtxt(day_poses_fname)
        day_poses = []
        pose = np.identity(4)
        for i in range(day_poses_raw.shape[0]):
            pose[:3,:3] = day_poses_raw[i][1:].reshape((4,3))[:3,:3]
            pose[:3, 3] = day_poses_raw[i][1:].reshape((4,3))[ 3,:3]
            day_poses.append(pose.copy())
        assert len(day_poses) == len(scan_fnames), 'Poses wrong!'

    img_dir = os.path.join(day_movie_dir, '{:s}_images'.format(config.keywords))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # for i, scan_fname in tqdm(enumerate(scan_fnames)):
    #     if not i % config.every_n:
    #         fig = plt.figure(figsize=(12, 12))
    #         ax = fig.add_subplot(111)
    #         current_time = time.time()
    #         velo_data = read_ply(os.path.join(config.scan_dir, scan_fname))
    #         velo = np.vstack((velo_data['x'], velo_data['y'], velo_data['z'], velo_data['pre'])).T
    #         time_statistics['read_ply'] += time.time() - current_time
    #
    #         # depends on whether a pose is required, set limit for x and y axis plotting
    #         if day_poses_fname == '':
    #             xlim = [-50, 50]
    #             ylim = [-50, 50]
    #         else:
    #             xlim = [day_poses[i][0,3] - 50, day_poses[i][0,3] + 50]
    #             ylim = [day_poses[i][1,3] - 50, day_poses[i][1,3] + 50]
    #
    #         current_time = time.time()
    #         draw_point_cloud(velo,
    #                          ax,
    #                          'Scan: {:s}'.format(scan_fname),
    #                          axes=[0, 1], # X and Y axes
    #                          xlim2d=xlim,
    #                          ylim2d=ylim
    #                          )
    #         time_statistics['draw_ply'] += time.time() - current_time
    #
    #         current_time = time.time()
    #         filename = '{:s}/frame_{:04}.png'.format(img_dir, i)
    #         plt.savefig(filename)
    #         plt.close(fig)
    #         time_statistics['save_fig'] += time.time() - current_time
    #         print(time_statistics)

    # generate video from images
    img_array = []
    print(img_dir)
    for filename in tqdm(sorted(os.listdir(img_dir))):
        print(filename)
        if filename.split('.')[-1] == 'png':
            img = cv2.imread(os.path.join(img_dir, filename))
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

    out = cv2.VideoWriter('{:s}/{:s}.avi'.format(day_movie_dir, config.save_fname),cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()