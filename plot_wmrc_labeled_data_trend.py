import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from wrench.labelmodel import WMRC

if __name__ == "__main__":
    datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
            'imdb', 'obs', 'sms', 'yelp', 'youtube']

    wmrc_names = ['WMRC']
    wmrc_fn_ends = ['_accuracy_semisup_validlabels_ineqconst_binomial.mat']

    # load all the data from the saved files
    results_folder_path = './results'

    for dataset in datasets:

        dataset_result_path = os.path.join(results_folder_path, dataset)

        if dataset == 'aa2':
            n_labeled_cts = [100, 113, 125, 137, 150, 161, 172]
        else:
            n_labeled_cts = [100, 150, 200, 250, 300]

        num_pts = len(n_labeled_cts)

        acc_mean = np.zeros(num_pts)
        acc_97_5 = np.zeros(num_pts)
        acc_2_5 = np.zeros(num_pts)
        logloss_mean = np.zeros(num_pts)
        logloss_97_5 = np.zeros(num_pts)
        logloss_2_5 = np.zeros(num_pts)
        brier_mean = np.zeros(num_pts)
        brier_97_5 = np.zeros(num_pts)
        brier_2_5 = np.zeros(num_pts)

        # load oracle losses
        oracle_result_fn = 'WMRC_' + dataset + '_accuracy_semisup_trainlabels'+\
                '_eqconst.mat'
        oracle_wmrc_full = os.path.join(dataset_result_path, 'WMRC/accuracy/' +\
                    oracle_result_fn)

        oracle_mdic = sio.loadmat(oracle_wmrc_full)
        oracle_acc = oracle_mdic['acc_train']
        oracle_logloss = oracle_mdic['log_loss_train']
        oracle_brier = oracle_mdic['brier_score_train']

        fig, ax = plt.subplots()
        # fig.dpi = 1200
        plot_fn = os.path.join(dataset_result_path,\
                'WMRC_' + dataset + '_labeled_trend.pdf')

        wmrc_names = ['WMRC/accuracy/'] + ['WMRC/accuracy+MV/']
        wmrc_fn_ends = ['.mat', '_add_mv.mat']

        for wmrc_name, wmrc_fn_end in zip(wmrc_names, wmrc_fn_ends):
            for i, n_labeled in enumerate(n_labeled_cts):
                result_fn = 'WMRC_' + dataset + '_accuracy_semisup_' +\
                        str(n_labeled) + 'validlabels_ineqconst_binomial' + wmrc_fn_end
                wmrc_fn_full = os.path.join(dataset_result_path, wmrc_name +\
                        result_fn)

                # load data
                mdic = sio.loadmat(wmrc_fn_full)

                accs = mdic['acc_train']
                loglosses = mdic['log_loss_train']
                briers = mdic['brier_score_train']

                accs = accs.squeeze() if accs.ndim > 1 else accs
                loglosses = loglosses.squeeze() if loglosses.ndim > 1 else loglosses
                briers = briers.squeeze() if briers.ndim > 1 else briers

                if accs.shape != ():
                    acc_mean[i] = mdic['acc_train_mean']
                    logloss_mean[i] = mdic['log_loss_train_mean']
                    brier_mean[i] = mdic['brier_score_train_mean']

                    # get empirical quantiles
                    coverage = 0.95
                    lb = (1 - coverage) / 2
                    ub = (1 + coverage) / 2
                    acc_cis = np.quantile(accs, [lb, ub])
                    logloss_cis = np.quantile(loglosses, [lb, ub])
                    brier_cis = np.quantile(briers, [lb, ub])

                    acc_97_5[i] = acc_cis[1]
                    logloss_97_5[i] = logloss_cis[1]
                    brier_97_5[i] = brier_cis[1]
                    acc_2_5[i] = acc_cis[0]
                    logloss_2_5[i] = logloss_cis[0]
                    brier_2_5[i] = brier_cis[0]

                else:
                    acc_mean[i] = accs
                    logloss_mean[i] = loglosses
                    brier_mean[i] = briers

                    acc_97_5[i] = accs
                    logloss_97_5[i] = loglosses
                    brier_97_5[i] = briers
                    acc_2_5[i] = accs
                    logloss_2_5[i] = loglosses
                    brier_2_5[i] = briers

            acc_mean = (1 - acc_mean) * 100
            acc_97_5 = (1 - acc_97_5) * 100
            acc_2_5 = (1 - acc_2_5) * 100

            if wmrc_fn_end == '.mat':
                mthd = 'WMRC'
                c = 'g'
                sttng = 'o-'
            else:
                mthd = 'WMRC + MV'
                c = 'b'
                sttng = '^--'
            ax.plot(n_labeled_cts, acc_mean, c + sttng, label=mthd + ' 0-1 Loss')
            ax.fill_between(n_labeled_cts, acc_2_5, acc_97_5, color=c, alpha=0.3,
                label=mthd + ' 95% Confidence')
        # ax.plot(n_labeled_cts, logloss_mean, 'ro-', label='Log Loss')
        # ax.fill_between(n_labeled_cts, logloss_2_5, logloss_97_5, color='r', alpha=0.3)
                # label='95% Confidence')
        # ax.plot(n_labeled_cts, brier_mean, 'bo-', label='Brier Score')
        # ax.fill_between(n_labeled_cts, brier_2_5, brier_97_5, color='b', alpha=0.3)
                # label='95% Confidence')

        # x_min = min(n_labeled_cts)
        # x_max = max(n_labeled_cts)
        # ax.hlines(1 - oracle_acc, x_min, x_max, 'g', label='Oracle 0-1')
        # ax.hlines(oracle_logloss, x_min, x_max, 'r', label='Oracle Log Loss')
        # ax.hlines(oracle_brier, x_min, x_max, 'b', label='Oracle Brier Score')

        ax.legend()
        ax.set_ylabel('0-1 Loss (%)')
        ax.set_xlabel('# Labeled Points')

        fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
        plt.close(fig)
