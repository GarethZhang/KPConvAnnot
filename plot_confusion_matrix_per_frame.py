import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils.ply import read_ply

class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # current working directory
    work_dir = '/home/haowei/MEGA/Research/src/KPConvAnnot'

    # result directory
    result_dir = 'test'

    # log data
    log_dir = 'Log_2020-12-08_V0'

    # default report directory
    fname = 'boreas-2020-12-01-13-26_0000043.ply'
    # report_dir = 'reports'
    row_si = 7
    num_rows = 5

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_dir', type=str, default='', metavar='N',
                        help='Job directory')
    parser.add_argument('--fname', type=str, default='', metavar='N',
                        help='Frame name')

    args = parser.parse_args()

    config = Config()

    # update config
    config.log_dir = args.job_dir if args.job_dir != '' else config.log_dir
    config.fname = args.fname if args.fname != '' else config.fname

    plt.rc('font', size=config.BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=config.MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=config.MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=config.BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=config.BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=config.SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=config.BIGGER_SIZE)  # fontsize of the figure title

    pred = np.load('/home/haowei/MEGA/Research/src/KPConvAnnot/test/Log_2021-02-16_V0.1/val_probs/boreas-2020-12-01-13-26_0000043.npy')

    #########################
    # Plot confusion matrices
    #########################
    log_dir = join(config.work_dir,
                   config.log_dir)
    fname = join(log_dir,
                 'val_predictions',
                 config.fname)
    # report_dir = join(log_dir,
    #                   config.report_dir)
    # report_fnames = [join(report_dir, report) for report in sorted(os.listdir(report_dir))]

    column_labels = ['Uncertain', 'Ground', 'Still', 'LongT', 'ShortT']
    row_labels = ['Uncertain', 'Ground', 'Still', 'LongT', 'ShortT']

    frame_data = read_ply(fname)
    frame_gt = frame_data['gt']
    frame_pre = frame_data['pre']

    # build confusion matrix from per-frame prediction
    confusion_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            confusion_matrix[i, j] = np.count_nonzero((frame_gt == i) * (frame_pre == j))

    # Plot merged confusion matrix
    confusion_matrix_in_percent = confusion_matrix / (np.sum(confusion_matrix, axis=1, keepdims=True) + 1e-6)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    heatmap = ax.pcolor(confusion_matrix_in_percent, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(confusion_matrix_in_percent.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(confusion_matrix_in_percent.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    # loop through data for adding text
    for i in range(confusion_matrix_in_percent.shape[0]):
        for j in range(confusion_matrix_in_percent.shape[1]):
            ax.text(i + 0.2, j + 0.5, "{0:.2%}".format(confusion_matrix_in_percent[j][i]), style='italic')
    fig.savefig('{:s}/confusion_matrix_{:s}.png'.format(log_dir, config.fname.split('.')[0]))