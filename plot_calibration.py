import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.calibration import calibration_curve

# Don't use the computed calibrations for random methods because the way they're
# saved is
# too idiosyncractic and we want the calibration for the average prediction
# of random methods anyways.  we'll find the average prediction and compute
# the calibration for that.

def swap_eles(lst, start, end):
    nlst = copy.deepcopy(lst)
    for s_ind, e_ind in zip(start, end):
        nlst[e_ind] = lst[s_ind]

    return nlst

if __name__ == "__main__":

    labeled_datasets = True
    # labeled_datasets = False

    if labeled_datasets:
        datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', \
                'imdb', 'obs', 'sms', 'yelp', 'youtube']
        rand_methods = ['Snorkel', 'EBCC', 'AMCL_CC', 'WMRC', 'WMRC+MV']
    else:
        # crowdsourcing datasets
        datasets = ['bird', 'rte']
        rand_methods = ['Snorkel', 'EBCC']

    det_methods = ['MV', 'HyperLM', 'WMRC unsup']
    wmrc_folder = ['WMRC/accuracy/'] + ['WMRC/accuracy+MV/']
    wmrc_fn_ends = ['_accuracy_semisup_100validlabels_ineqconst_binomial.mat',
                        '_accuracy_semisup_100validlabels_ineqconst_binomial_add_mv.mat']

    # load all the data from the saved files
    results_folder_path = './results'

    # hard code plot settings
    # settings = ['-c^', 'rD-', 'y*-', 'mX-', 'b-', 'go-', 'go:', 'go--']
    if labeled_datasets:
        settings = ['-', '-', '-', '-', '-', '-', ':', '--']
        colors = ['c', 'r', 'y', 'm', 'b', 'g', 'chartreuse', 'olive']
    else:
        settings = ['-', '-', '-', '-', '--']
        colors = ['c', 'r', 'y', 'm', 'b', 'olive']

    for dataset in datasets:
        dataset_result_path = os.path.join(results_folder_path, dataset)

        x_calib = []
        y_calib = []
        name = []
        for i, method in enumerate(det_methods):
            if i < 2:
                method_result_path = os.path.join(dataset_result_path, method)
                method_fn = method + '_' + dataset + '.mat'
            else:
                method_result_path = os.path.join(dataset_result_path, 'WMRC/accuracy')
                method_fn = 'WMRC_' + dataset + '_accuracy_unsup_trainlabels_ineqconst.mat'

            method_fn_full = os.path.join(method_result_path, method_fn)

            # load data
            mdic = sio.loadmat(method_fn_full)
            x_calib.append(mdic['x_calibration_train'].squeeze())
            y_calib.append(mdic['y_calibration_train'].squeeze())
            name.append(method)

        for i, method in enumerate(rand_methods):
            wmrc_first_ind = 3
            if i < 3:
                method_result_path = os.path.join(dataset_result_path, method)
                method_fn = method + '_' + dataset + '.mat'
            else:
                wmrc_ind = i - wmrc_first_ind
                method_result_path = os.path.join(dataset_result_path, wmrc_folder[wmrc_ind])
                method_fn = 'WMRC_' + dataset + wmrc_fn_ends[wmrc_ind]
            method_fn_full = os.path.join(method_result_path, method_fn)

            # load data
            mdic = sio.loadmat(method_fn_full)
            all_preds = mdic['pred_train']
            mean_preds = 0
            for i in range(10):
                mean_preds += all_preds[i] / 10

            # compute calibration
            prob_true_train, prob_pred_train = calibration_curve(
                    np.squeeze(mdic["true_labels_train"]),
                    np.clip(mean_preds[:, 1], 0, 1),
                    n_bins=10)
            x_calib.append(prob_pred_train)
            y_calib.append(prob_true_train)
            name.append(method)

        # swap elements so WMRC is all together
        if labeled_datasets:
            x_calib = swap_eles(x_calib, [3, 1, 4, 2, 7, 6, 5], [1, 3, 2, 7, 6, 5, 4])
            name = swap_eles(name, [3, 1, 4, 2, 7, 6, 5], [1, 3, 2, 7, 6, 5, 4])
            y_calib = swap_eles(y_calib, [3, 1, 4, 2, 7, 6, 5], [1, 3, 2, 7, 6, 5, 4])
        else:
            x_calib = swap_eles(x_calib, [3, 1, 2, 4], [1, 3, 4, 2])
            name = swap_eles(name, [3, 1, 2, 4], [1, 3, 4, 2])
            y_calib = swap_eles(y_calib, [3, 1, 2, 4], [1, 3, 4, 2])

        # plot calibration
        plot_fn = os.path.join(dataset_result_path,\
                dataset + '_' + 'calibration.pdf')

        fig, ax = plt.subplots()
        # fig.dpi = 1200

        ax.plot([0, 1], [0, 1], 'k-', label='Perfectly Calibrated')
        for i in range(len(name)):
            if x_calib[i].size == 1:
                ax.plot(x_calib[i], y_calib[i], settings[i] + 'o', label=name[i], lw=2, color=colors[i])
            else:
                ax.plot(x_calib[i], y_calib[i], settings[i], label=name[i], lw=2, color=colors[i])

        ax.legend()
        ax.set_ylabel('Actual Fraction of Positives')
        ax.set_xlabel('Mean Predicted Fraction of Positives')

        fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
        plt.close(fig)
