import os
import copy
import numpy as np
import scipy.io as sio
from scipy.io import savemat

if __name__ == "__main__":

    labeled_datasets = True

    # these are the only datasets with abstentions in the train and validation
    # predictiosn
    datasets = ['basketball', 'imdb', 'sms', 'yelp', 'youtube']

    # load all the data from the saved files
    dataset_folder_path = './datasets/'

    for d_ind, dataset in enumerate(datasets):
        # load data
        dataset_path = os.path.join(dataset_folder_path, dataset + '.mat')

        mdic = sio.loadmat(dataset_path)

        tp = mdic['train_pred']
        vp = mdic['val_pred']

        num_lf = tp.shape[1]
        num_tpts = tp.shape[0]
        num_vpts = vp.shape[0]

        for i in range(num_lf):
            tp_sels = tp[:, i] == -1
            vp_sels = vp[:, i] == -1

            tp[tp_sels, i] = np.random.binomial(1, 0.5, size=tp_sels.sum())
            vp[vp_sels, i] = np.random.binomial(1, 0.5, size=vp_sels.sum())

        mdic['train_pred'] = tp
        mdic['val_pred'] = vp

        fn = dataset + "_filled.mat"
        savemat(os.path.join(dataset_folder_path, fn), mdic)
