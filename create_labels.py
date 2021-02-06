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

    # current working directory
    data_dir = '/home/gzh/Documents/research/src/KPConvAnnot/test/Log_2021-01-15_V0.5/val_predictions'

    save_dir = '/home/gzh/Documents/research/src/KPConvAnnot/test/Log_2021-01-15_V0.5'

    pred_fname = 'pred_labels.txt'

    # font config
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

if __name__ == '__main__':

    config = Config()

    fnames = sorted(os.listdir(config.data_dir))

    pred_labels_fname = os.path.join(config.save_dir, config.pred_fname)

    with open(pred_labels_fname, "a") as pred_labels_file:
        for fname in tqdm(fnames):
            data = read_ply(os.path.join(config.data_dir, fname))
            pred = np.array(data['pre']).astype(np.int32)
            pred_str = [str(pred_i) for pred_i in pred]
            pred_labels_file.writelines(' '.join(pred_str) + '\n')
            # # pred_labels_file.write(b'\n')
            # np.savetxt(pred_labels_fname, pred, fmt='%d')
            # pred_labels_file.write(b'hi\n')
    pred_labels_file.close()

    print('============================')
    print('============DONE============')
    print('============================')