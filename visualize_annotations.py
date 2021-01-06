import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # current working directory
    data_dir = '/mnt/hdd2/Boreas/boreas-2020-11-26-13-58/annotation_old/annotated_frames'

    # log data
    sample_dir = 'samples'

    every_n = 1000

    # visualization configs
    training_log = 'training.txt'
    validation_log = 'validation.txt'
    training_visualization_options = ['out_loss', 'train_accuracy']
    validation_visualization_options = ['out_loss', 'validation_accuracy']

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22



if __name__ == '__main__':

    config = Config()

    day_dirs = sorted(os.listdir(config.data_dir))

    # create top-level sample dir
    sample_dir = os.path.join(config.data_dir, config.sample_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # loop through each day and sample
    for day_dir in day_dirs:
        # create day-based sample dir
        sample_day_dir = os.path.join(sample_dir, day_dir)
        if not os.path.exists(sample_day_dir):
            os.makedirs(sample_day_dir)

        day_fnames = sorted(os.listdir(os.path.join(config.data_dir, day_dir)))
        sample_day_fnames = [day_fname for i, day_fname in enumerate(day_fnames) if not i % config.every_n]

        # create softlink
        for sample_day_fname in sample_day_fnames:
            src = os.path.join(config.data_dir, day_dir, sample_day_fname)
            dst = os.path.join(config.data_dir, sample_day_dir, sample_day_fname)
            os.symlink(src, dst)
            # print(src, dst)