''' This Python script does post-processing to scans recorded from ROS node semantic_labeller_node. This applies to all
DAY_DIR under DATA_DIR

Assumptions are made regarding the directory structure and file formats. Please see following:

DATA_DIR/
|-- LOG_DIR/
|   |
|   |-- applanix_localization/
|       |-- scans/
|           |-- lidar frames (in .ply format)
|           |
|           |-- map_poses.txt
|...

Note that lidar frames are named under the convention velo_SEQ_TIMESTAMP.ply.

Procedures to involved in post-processing process
1. Fix the sequence number problem, e.g velo_1_TIMESTAMP ==> velo_00001_TIMESTAMP, by default 5-digits
2. Based on number of entries available in map_poses.npy, remove redundant scans based on timestamp.

map_poses.npy records poses in $T_map_velo$ with 13 numbers (1 timestamp + 12 (3x4) pose matrix).

'''
# sys imports
import os
import sys
import math
import argparse
import itertools

# third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# external package imports
opj = os.path.join

class Config:
    """
    Class containing the parameters you want to modify
    """

    ##################
    # Input parameters
    ##################

    # top-level data directory and associate working directories
    data_dir = '/home/haowei/Documents/data/boreas'

    raw_dir = 'raw'

    seq_dir = 'sequences'

    log_dirs = ['boreas-2021-01-31-local-00']

    scan_dir = 'velodyne'

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--slurm_dir', type=str, default='', metavar='N',
                        help='SLURM directory')

    args = parser.parse_args()

    config = Config()

    config.data_dir = args.slurm_dir

    # loop through all log directory
    for log_dir in sorted(config.log_dirs):
        print('===================================================')
        print('===========LOG: {:s}============'.format(log_dir))
        print('===================================================')

        scan_dir = os.path.join(config.data_dir, config.raw_dir, config.seq_dir, log_dir, config.scan_dir)

        # start renaming files assuming scans are only filetypes under scan_dir
        # also keep notice of timestamps
        num_renamed = 0
        scan_timestamps = []
        scan_fnames = sorted(os.listdir(scan_dir))
        for fname in scan_fnames:
            fname_split = fname.split('_')
            updated_fname = fname_split[0] + '_' + '0' * (5 - len(fname_split[1])) + fname_split[1] + '_' + fname_split[2]
            scan_timestamps.append(int(fname_split[2].split('.')[0]))
            if updated_fname == fname:
                continue
            else:
                os.rename(opj(scan_dir, fname), opj(scan_dir, updated_fname))
                num_renamed += 1
        print("Renamed: {:d}/{:d}".format(num_renamed, len(scan_fnames)))


    print('===================================================')
    print('===================JOB COMPLETE====================')
    print('===================================================')