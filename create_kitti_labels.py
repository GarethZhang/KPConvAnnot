import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils.ply import read_ply
from tqdm import tqdm

class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # # current working directory
    # data_dir = '/home/gzh/Documents/research/src/KPConvAnnot/test/Log_2021-01-15_V0.5/val_predictions'
    #
    # save_dir = '/home/gzh/Documents/research/src/KPConvAnnot/test/Log_2021-01-15_V0.5'

    pred_fname = 'pred_labels'

    log_dirs = ['00']

    keywords = ['00_']

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_dir', type=str, default='results/Log_2020-04-22_18-29-55', metavar='N',
                        help='Job directory')
    parser.add_argument('--slurm_dir', type=str, default='results/Log_2020-04-22_18-29-55', metavar='N',
                        help='Job directory')

    args = parser.parse_args()

    data_dir = args.job_dir + '/predictions'
    save_dir = args.job_dir

    config = Config()

    fnames = sorted(os.listdir(data_dir))

    for j, log_dir in enumerate(config.log_dirs):

        log_fnames = [fname for fname in fnames if config.keywords[j] in fname]

        print(log_fnames)

        time_fname = os.path.join(args.slurm_dir, 'sequences', log_dir, 'times.txt')

        time_file = open(time_fname, 'r')

        times = time_file.readlines()

        print(len(log_fnames), len(times))

        if len(log_fnames) != len(times):
            print('Skip {:s} due to non-matching'.format(log_dir))
            continue

        pred_labels_fname = os.path.join(save_dir, '{:s}_{:s}.txt'.format(config.pred_fname, log_dir))

        if os.path.exists(pred_labels_fname):
            print('Label already exists: {:s}'.format(log_dir))
            continue

        with open(pred_labels_fname, "a") as pred_labels_file:
            for i, fname in tqdm(enumerate(log_fnames)):
                data = read_ply(os.path.join(data_dir, fname))
                pred = np.array(data['pre']).astype(np.int32)
                pred_str = [str(pred_i) for pred_i in pred]
                pred_str = [str(float(times[i]))] + pred_str
                pred_labels_file.writelines(' '.join(pred_str) + '\n')
                # # pred_labels_file.write(b'\n')
                # np.savetxt(pred_labels_fname, pred, fmt='%d')
                # pred_labels_file.write(b'hi\n')
        pred_labels_file.close()

        print('Done {:s}'.format(log_dir))

    print('============================')
    print('============DONE============')
    print('============================')