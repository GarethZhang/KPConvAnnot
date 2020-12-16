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
    validation_log = 'validation.txt'
    training_visualization_options = ['out_loss', 'train_accuracy']
    validation_visualization_options = ['out_loss', 'validation_accuracy']

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
    validation_log_fname = join(log_dir,
                                config.validation_log)

    validation_exists = False
    if os.path.exists(validation_log_fname):
        validation_exists = True

    training_titles = ['epochs', 'steps', 'out_loss', 'offset_loss', 'train_accuracy', 'time']
    training_stats = np.loadtxt(training_log_fname, skiprows=1)
    training_y_labels = ['Epoch', 'Step', 'Cross Entropy Loss', '', 'Accuracy', 'Time']

    # configure visualization options
    training_visualization_options = config.training_visualization_options

    # compile epoch stats for training
    training_epochs = np.unique(training_stats[:,0])
    num_training_epochs = training_epochs.shape[0]
    num_training_titles = len(training_titles)
    num_training_vo = len(training_visualization_options)

    epoch_training_stats = np.zeros((num_training_epochs, num_training_titles))
    for i in range(num_training_epochs):
        epoch_training_stats[i] = np.mean(training_stats[training_stats[:, 0] == i], axis=0)

    ncols = 2
    nrows = num_training_vo // ncols + 1 if num_training_vo % ncols else num_training_vo // ncols
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('{:s}:{:s}'.format(args.keywords, config.log_dir))
    for i, vo in enumerate(training_visualization_options):
        index = training_titles.index(vo)
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.plot(np.arange(num_training_epochs), epoch_training_stats[:, index])
        ax.set_title(vo)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(training_y_labels[index])
    fig.savefig('{:s}/training_convergence.png'.format(log_dir))

    if validation_exists:
        validation_titles = ['epochs', 'index', 'out_loss', 'offset_loss', 'validation_accuracy', 'time']
        validation_stats = np.loadtxt(validation_log_fname, skiprows=1)
        validation_y_labels = ['Epoch', 'Index', 'Cross Entropy Loss', '', 'Accuracy', 'Time']

        # configure visualization options
        validation_visualization_options = config.validation_visualization_options

        # compile epoch stats for validation
        validation_epochs = np.unique(validation_stats[:,0])
        num_validation_epochs = validation_epochs.shape[0]
        num_validation_titles = len(validation_titles)
        num_validation_vo = len(validation_visualization_options)

        epoch_validation_stats = np.zeros((num_validation_epochs, num_validation_titles))
        for i in range(1, num_validation_epochs):
            epoch_validation_stats[i] = np.mean(validation_stats[validation_stats[:, 0] == i], axis=0)

        ncols = 2
        nrows = num_validation_vo // ncols + 1 if num_validation_vo % ncols else num_validation_vo // ncols
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('{:s}:{:s}'.format(args.keywords, config.log_dir))
        for i, vo in enumerate(validation_visualization_options):
            index = validation_titles.index(vo)
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.plot(np.arange(1, num_validation_epochs), epoch_validation_stats[1:, index])
            ax.set_title(vo)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(validation_y_labels[index])
        fig.savefig('{:s}/validation_convergence.png'.format(log_dir))

        # plot training vs validation
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('{:s}:{:s}'.format(args.keywords, config.log_dir))
        for i, vo in enumerate(validation_visualization_options):
            training_index = training_titles.index(training_visualization_options[i])
            validation_index = validation_titles.index(vo)
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.plot(np.arange(num_training_epochs), epoch_training_stats[:, training_index], label='train')
            ax.plot(np.arange(1, num_validation_epochs), epoch_validation_stats[1:, validation_index], label='validation')
            ax.set_title(vo)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(validation_y_labels[validation_index])
            ax.legend()
        fig.savefig('{:s}/convergence.png'.format(log_dir))