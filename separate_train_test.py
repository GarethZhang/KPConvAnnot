''' This Python script separates a Boreas run data into both training and testing sequences using softlinks

Assumptions are made regarding the directory structure and file formats. Please see following:

DATA_DIR (e.g. /raid/gzh/Data/Boreas)/
|-- raw/
|   |-- sequences/
|       |-- DAY_DIR (e.g. boreas-2020-12-01-13-26)/
|           |-- annotated (in .ply format)
|           |-- velodyne (in .ply format)
|           |-- redundant_scans (in .ply format)
|           |-- map_poses.txt
|-- train/
|   |-- sequences/
|       |-- ...
|-- test/
|   |-- sequences/
|       |-- ...
|...

For both training and testing sequences, it would rely on pre-defined poses provided to determine whether a frame belongs
to either training, testing, or None. This pre-defined poses is manually selected from first localization run.

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

    # testing data
    test_dir = 'validation'

    seq_dir = 'sequences'

    scan_dir = 'velodyne'

    annot_dir = 'annotated'

    pose_fname = 'map_poses.txt'

    train_seq_si = [1150, 2470, 7630]

    train_seq_ei = [1920, 6800, 8600]

    test_seq_si = [1930, 6810]

    test_seq_ei = [2450, 7615]


    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

if __name__ == '__main__':

    config = Config()

    raw_dir = join(config.data_dir, config.raw_dir)

    seq_dir = join(raw_dir, config.seq_dir)

    train_dir = join(config.data_dir, config.train_dir)

    test_dir = join(config.data_dir, config.test_dir)

    train_seq_dir = join(train_dir, config.seq_dir)

    test_seq_dir = join(test_dir, config.seq_dir)

    T_map_velo_ref = None

    T_map_velo_train = None

    T_map_velo_test = None

    for i, day in enumerate(sorted(os.listdir(seq_dir))):
        # read directories from raw
        day_dir = join(seq_dir, day)
        day_scan_dir = join(day_dir, config.scan_dir)
        day_annot_dir = join(day_dir, config.annot_dir)
        day_scan_fnames = sorted(os.listdir(day_scan_dir))
        day_annot_fnames = sorted(os.listdir(day_annot_dir))
        day_pose_fname = join(day_dir, config.pose_fname)
        assert len(day_scan_fnames) == len(day_annot_fnames), "Scan and annotations must have the same number of frames!"

        # create new directories for training if not exists
        train_day_dir = join(train_seq_dir, day)
        train_day_scan_dir = join(train_day_dir, config.scan_dir)
        if not os.path.exists(train_day_scan_dir):
            print("Create train day scan directory: {:s}".format(day))
            os.makedirs(train_day_scan_dir)
        train_day_annot_dir = join(train_day_dir, config.annot_dir)
        if not os.path.exists(train_day_annot_dir):
            print("Create train day annot directory: {:s}".format(day))
            os.makedirs(train_day_annot_dir)

        test_day_dir = join(test_seq_dir, day)
        test_day_scan_dir = join(test_day_dir, config.scan_dir)
        if not os.path.exists(test_day_scan_dir):
            print("Create test day scan directory: {:s}".format(day))
            os.makedirs(test_day_scan_dir)
        test_day_annot_dir = join(test_day_dir, config.annot_dir)
        if not os.path.exists(test_day_annot_dir):
            print("Create test day annot directory: {:s}".format(day))
            os.makedirs(test_day_annot_dir)

        # For first localization run, use sequence indices to determine train and test
        poses_all = np.loadtxt(day_pose_fname).astype(np.float32)
        num_poses_all = poses_all.shape[0]
        T_map_velo_loc = np.expand_dims(np.identity(4, dtype=np.float32), axis=0).repeat(num_poses_all, axis=0)
        T_map_velo_loc[:,:3,:3] = poses_all[:,1:].reshape((-1, 4, 3))[:,:3,:3]
        T_map_velo_loc[:,:3, 3] = (poses_all[:,1:].reshape((-1, 4, 3))[:, 3,:3])

        if i == 0:
            # save training and testing poses
            train_inds = [list(range(config.train_seq_si[ind], config.train_seq_ei[ind])) for ind in range(len(config.train_seq_si))]
            train_inds = np.array([item for sublist in train_inds for item in sublist]).astype(np.int32)

            test_inds = [list(range(config.test_seq_si[ind], config.test_seq_ei[ind])) for ind in range(len(config.test_seq_si))]
            test_inds = np.array([item for sublist in test_inds for item in sublist]).astype(np.int32)

            T_map_velo_train = T_map_velo_loc[train_inds]
            T_map_velo_test = T_map_velo_loc[test_inds]
        else:
            train_inds = get_nearby_poses(T_map_velo_train, T_map_velo_loc)
            train_inds = np.where(train_inds > 0.5)[0]
            test_inds = get_nearby_poses(T_map_velo_test, T_map_velo_loc)
            test_inds = np.where(test_inds > 0.5)[0]
        print("day: {:s} ==> train: {:d} test: {:d}".format(day, train_inds.shape[0], test_inds.shape[0]))

        poses_all = np.loadtxt(day_pose_fname).astype(np.float32)
        train_day_pose_fname = join(train_day_dir, config.pose_fname)
        np.savetxt(train_day_pose_fname, poses_all[train_inds])

        test_day_pose_fname = join(test_day_dir, config.pose_fname)
        np.savetxt(test_day_pose_fname, poses_all[test_inds])

        # save training and testing scans
        train_day_scan_fnames = [day_scan_fnames[ind] for ind in range(len(day_scan_fnames)) if ind in train_inds]
        test_day_scan_fnames = [day_scan_fnames[ind] for ind in range(len(day_scan_fnames)) if ind in test_inds]
        for train_day_scan_fname in train_day_scan_fnames:
            os.symlink(join(day_scan_dir, train_day_scan_fname),
                       join(train_day_scan_dir, train_day_scan_fname))
        for test_day_scan_fname in test_day_scan_fnames:
            os.symlink(join(day_scan_dir, test_day_scan_fname),
                       join(test_day_scan_dir, test_day_scan_fname))

        # save training and testing annot
        train_day_annot_fnames = [day_annot_fnames[ind] for ind in range(len(day_annot_fnames)) if ind in train_inds]
        test_day_annot_fnames = [day_annot_fnames[ind] for ind in range(len(day_annot_fnames)) if ind in test_inds]
        for train_day_annot_fname in train_day_annot_fnames:
            train_src = join(day_annot_dir, train_day_annot_fname)
            train_dst = join(train_day_annot_dir, train_day_annot_fname)
            if not os.path.exists(train_dst):
                os.symlink(train_src, train_dst)
        for test_day_annot_fname in test_day_annot_fnames:
            test_src = join(day_annot_dir, test_day_annot_fname)
            test_dst = join(test_day_annot_dir, test_day_annot_fname)
            if not os.path.exists(test_dst):
                os.symlink(test_src, test_dst)

    print('============================')
    print('============DONE============')
    print('============================')