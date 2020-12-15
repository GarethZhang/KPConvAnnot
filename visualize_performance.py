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

    # log data
    log_dir = 'Log_2020-12-08_V0'

    # visualization configs
    training_log = 'training.txt'
    visualization_options = ['out_loss', 'train_accuracy']

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_dir', type=str, default='', metavar='N',
                        help='Job directory')
    parser.add_argument('--keywords', type=str, default='', metavar='N',
                        help='Keywords to describe this log')

    args = parser.parse_args()

    config = Config()

    # update config
    config.log_dir = args.job_dir if args.job_dir != '' else config.log_dir

    plt.rc('font', size=config.SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=config.MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=config.MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=config.SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=config.SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=config.SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=config.BIGGER_SIZE)  # fontsize of the figure title

    ##################
    # Plot convergence
    ##################
    log_dir = join(config.work_dir,
                   config.log_dir)
    training_log_fname = join(log_dir,
                              config.training_log)
    training_titles = ['epochs', 'steps', 'out_loss', 'offset_loss', 'train_accuracy', 'time']
    training_stats = np.loadtxt(training_log_fname, skiprows=1)
    y_labels = ['Epoch', 'Step', 'Cross Entropy Loss', '', 'Accuracy', 'Time']

    # configure visualization options
    visualization_options = config.visualization_options

    # compile epoch stats
    epochs = np.unique(training_stats[:,0])
    num_epochs = epochs.shape[0]
    num_titles = len(training_titles)
    num_vo = len(visualization_options)

    epoch_stats = np.zeros((num_epochs, num_titles))
    for i in range(num_epochs):
        epoch_stats[i] = np.mean(training_stats[training_stats[:, 0] == i], axis=0)

    ncols = 2
    nrows = num_vo // ncols + 1
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('{:s}:{:s}'.format(args.keywords, config.log_dir))
    for i, vo in enumerate(visualization_options):
        index = training_titles.index(vo)
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.plot(np.arange(num_epochs), epoch_stats[:, index])
        ax.set_title(vo)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(y_labels[index])
    fig.savefig('{:s}/convergence.png'.format(log_dir))