''' This Python script separates a Boreas run data into both training, validation and testing sequences using symlinks

Assumptions are made regarding the directory structure and file formats. Please see following:

DATA_DIR (e.g. /raid/gzh/Data/Boreas)/
|-- raw/
|   |-- sequences/
|       |-- DAY_DIR (e.g. boreas-2020-12-01-13-26)/
|           |-- velodyne (in .ply format)
|           |-- map_poses.txt
|-- training/
|   |-- sequences/
|       |-- ...
|-- validation/
|   |-- sequences/
|       |-- ...
|-- test/
|   |-- sequences/
|       |-- ...
|...

For training and validation sequences, it would rely on pre-defined poses provided to determine whether a frame belongs
to either training, or validation, or None. This pre-defined poses is manually selected from first localization run.

For test sequence, it would rely on a test_day variable pre-defined. Test sequence is nowhere from either training or
validation.

'''

import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils.ply import read_ply
from tqdm import tqdm
from utils.pc_utils import compute_criteria

def get_nearby_poses(T_src, T_tgt):
    translation_src = T_src[:,:3,3]
    translation_tgt = T_tgt[:,:3,3]
    inds = compute_criteria(translation_tgt, translation_src)
    return inds

class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # current data directory
    data_dir = '/raid/gzh/Data/Boreas'

    # stores un-split data
    raw_dir = 'raw'

    # training data
    train_dir = 'training'

    # validation data
    val_dir = 'validation'

    # test data
    test_dir = 'test'

    # test_days = ['boreas-2021-01-31-dufferin-00', 'boreas-2021-01-31-highway-00-00', 'boreas-2021-01-31-local-00']

    test_days = ['boreas-2021-01-31-local-00']

    seq_dir = 'sequences'

    scan_dir = 'velodyne'

    pose_fname = 'map_poses.txt'

    train_seq_si = [1150, 2470, 7630]

    train_seq_ei = [1920, 6800, 8600]

    val_seq_si = [1930, 6810]

    val_seq_ei = [2450, 7615]


    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/raid/gzh/Data/Boreas', metavar='N',
                        help='Data directory')
    args = parser.parse_args()

    config = Config()
    config.data_dir = args.data_dir

    raw_dir = join(config.data_dir, config.raw_dir)

    seq_dir = join(raw_dir, config.seq_dir)

    train_dir = join(config.data_dir, config.train_dir)

    val_dir = join(config.data_dir, config.val_dir)

    test_dir = join(config.data_dir, config.test_dir)

    train_seq_dir = join(train_dir, config.seq_dir)

    val_seq_dir = join(val_dir, config.seq_dir)

    test_seq_dir = join(test_dir, config.seq_dir)

    T_map_velo_ref = None

    T_map_velo_train = None

    T_map_velo_val = None

    for i, day in enumerate(sorted(os.listdir(seq_dir))):
        if day not in config.test_days:
            print('======================================')
            print('=============DAY-{:s}============='.format(day))
            print('======================================')
            # read directories from raw
            day_dir = join(seq_dir, day)
            day_scan_dir = join(day_dir, config.scan_dir)
            day_scan_fnames = sorted(os.listdir(day_scan_dir))
            day_pose_fname = join(day_dir, config.pose_fname)

            # create new directories for training if not exists
            train_day_dir = join(train_seq_dir, day)
            train_day_scan_dir = join(train_day_dir, config.scan_dir)
            if not os.path.exists(train_day_scan_dir):
                print("Create train day scan directory: {:s}".format(day))
                os.makedirs(train_day_scan_dir)

            val_day_dir = join(val_seq_dir, day)
            val_day_scan_dir = join(val_day_dir, config.scan_dir)
            if not os.path.exists(val_day_scan_dir):
                print("Create validation day scan directory: {:s}".format(day))
                os.makedirs(val_day_scan_dir)

            # For first localization run, use sequence indices to determine train and validation
            poses_all = np.loadtxt(day_pose_fname).astype(np.float32)
            num_poses_all = poses_all.shape[0]
            T_map_velo_loc = np.expand_dims(np.identity(4, dtype=np.float32), axis=0).repeat(num_poses_all, axis=0)
            T_map_velo_loc[:,:3,:3] = poses_all[:,1:].reshape((-1, 4, 3))[:,:3,:3]
            T_map_velo_loc[:,:3, 3] = (poses_all[:,1:].reshape((-1, 4, 3))[:, 3,:3])

            if i == 0:
                # save training and testing poses
                train_inds = [list(range(config.train_seq_si[ind], config.train_seq_ei[ind])) for ind in range(len(config.train_seq_si))]
                train_inds = np.array([item for sublist in train_inds for item in sublist]).astype(np.int32)

                val_inds = [list(range(config.val_seq_si[ind], config.val_seq_ei[ind])) for ind in range(len(config.val_seq_si))]
                val_inds = np.array([item for sublist in val_inds for item in sublist]).astype(np.int32)

                T_map_velo_train = T_map_velo_loc[train_inds]
                T_map_velo_val = T_map_velo_loc[val_inds]
            else:
                train_inds = get_nearby_poses(T_map_velo_train, T_map_velo_loc)
                train_inds = np.where(train_inds > 0.5)[0]
                val_inds = get_nearby_poses(T_map_velo_val, T_map_velo_loc)
                val_inds = np.where(val_inds > 0.5)[0]
            print("day: {:s} ==> train: {:d} test: {:d}".format(day, train_inds.shape[0], val_inds.shape[0]))

            poses_all = np.loadtxt(day_pose_fname).astype(np.float32)
            train_day_pose_fname = join(train_day_dir, config.pose_fname)
            np.savetxt(train_day_pose_fname, poses_all[train_inds])

            val_day_pose_fname = join(val_day_dir, config.pose_fname)
            np.savetxt(val_day_pose_fname, poses_all[val_inds])

            # save training and validation scans
            train_day_scan_fnames = [day_scan_fnames[ind] for ind in range(len(day_scan_fnames)) if ind in train_inds]
            val_day_scan_fnames = [day_scan_fnames[ind] for ind in range(len(day_scan_fnames)) if ind in val_inds]
            for train_day_scan_fname in train_day_scan_fnames:
                if not os.path.exists(join(train_day_scan_dir, train_day_scan_fname)):
                    os.symlink(join(day_scan_dir, train_day_scan_fname),
                               join(train_day_scan_dir, train_day_scan_fname))
            for val_day_scan_fname in val_day_scan_fnames:
                if not os.path.exists(join(val_day_scan_dir, val_day_scan_fname)):
                    os.symlink(join(day_scan_dir, val_day_scan_fname),
                               join(val_day_scan_dir, val_day_scan_fname))
        else:
            print('======================================')
            print('==========TEST DAY-{:s}=========='.format(day))
            print('======================================')
            # read directories from raw
            day_dir = join(seq_dir, day)

            test_day_dir = join(test_seq_dir, day)

            if not os.path.exists(test_seq_dir):
                print("Create test seq directory")
                os.makedirs(test_seq_dir)

            if not os.path.exists(test_day_dir):
                print("Create test day scan directory: {:s}".format(day))
                os.symlink(day_dir, test_day_dir)
            else:
                print("Test day scan directory already exists")


    print('============================')
    print('============DONE============')
    print('============================')