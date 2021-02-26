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
    work_dir = '/home/haowei/MEGA/Research/src/KPConvAnnot'

    # result directory
    result_dir = 'test'

    # log data
    log_dir = 'Log_2020-12-08_V0'

    # default report directory
    report_dir = 'reports'
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

    args = parser.parse_args()

    config = Config()

    # update config
    config.log_dir = args.job_dir if args.job_dir != '' else config.log_dir

    plt.rc('font', size=config.BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=config.MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=config.MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=config.BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=config.BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=config.SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=config.BIGGER_SIZE)  # fontsize of the figure title

    #########################
    # Plot confusion matrices
    #########################
    log_dir = join(config.work_dir,
                   config.log_dir)
    report_dir = join(log_dir,
                      config.report_dir)
    report_fnames = [join(report_dir, report) for report in sorted(os.listdir(report_dir))]

    confusion_matrices = []
    column_labels = ['Uncertain', 'Ground', 'Still', 'LongT', 'ShortT']
    row_labels = ['Uncertain', 'Ground', 'Still', 'LongT', 'ShortT']

    merged_confusion_matrix = None
    for i, report_fname in enumerate(report_fnames):
        confusion_matrix = np.loadtxt(report_fname, skiprows=config.row_si, max_rows=config.num_rows)
        merged_confusion_matrix = merged_confusion_matrix + confusion_matrix if merged_confusion_matrix is not None else confusion_matrix
        confusion_matrix_in_percent = confusion_matrix / (np.sum(confusion_matrix, axis=1, keepdims=True) + 1e-6)
        confusion_matrices.append(confusion_matrix_in_percent)

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
        fig.savefig('{:s}/confusion_matrix_{:s}.png'.format(log_dir, report_fname.split('/')[-1][:-4]))


    # Plot merged confusion matrix
    merged_confusion_matrix_in_percent = merged_confusion_matrix / (np.sum(merged_confusion_matrix, axis=1, keepdims=True) + 1e-6)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    heatmap = ax.pcolor(merged_confusion_matrix_in_percent, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(merged_confusion_matrix_in_percent.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(merged_confusion_matrix_in_percent.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    # loop through data for adding text
    for i in range(merged_confusion_matrix_in_percent.shape[0]):
        for j in range(merged_confusion_matrix_in_percent.shape[1]):
            ax.text(i + 0.2, j + 0.5, "{0:.2%}".format(merged_confusion_matrix_in_percent[j][i]), style='italic')
    fig.savefig('{:s}/merged_confusion_matrix.png'.format(log_dir))