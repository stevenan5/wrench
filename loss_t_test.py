import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from scipy.stats.mstats import ttest_rel

def swap_eles(lst, start, end):
    nlst = copy.deepcopy(lst)
    for s_ind, e_ind in zip(start, end):
        nlst[e_ind] = lst[s_ind]

    return nlst

if __name__ == "__main__":

    labeled_datasets = True
    # labeled_datasets = False

    if labeled_datasets:
        datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
                'imdb', 'obs', 'sms', 'yelp', 'youtube']
        rand_methods = ['EBCC', 'Snorkel', 'AMCL_CC', 'WMRC', 'WMRC+MV']
    else:
        # crowdsourcing datasets
        datasets = ['bird', 'rte', 'dog', 'web']
        rand_methods = ['EBCC', 'Snorkel']

    det_methods = ['MV', 'HyperLM', 'WMRC unsup']

    wmrc_folder = ['WMRC/accuracy/'] + ['WMRC/accuracy+MV/']
    wmrc_fn_ends = ['_accuracy_semisup_100validlabels_ineqconst_binomial.mat',
                        '_accuracy_semisup_100validlabels_ineqconst_binomial_add_mv.mat']

    # load all the data from the saved files
    results_folder_path = './results'
    losses = ['brier_score_train', 'log_loss_train', 'err_train']

    n_methods = len(rand_methods) + len(det_methods)
    res_mdic = {}
    for loss in losses:
        dist_results = np.zeros((n_methods, len(datasets)))
        agg_loss = np.zeros(dist_results.shape)

        wmrc_oracle_loss = np.zeros(len(datasets))
        wmrc_avg_labeled_pts_per_classifier = np.zeros(len(datasets))
        wmrc_avg_num_classifier_available = np.zeros(len(datasets))
        wmrc_avg_fit_time = np.zeros(len(datasets))
        amcl_cc_avg_fit_time = np.zeros(len(datasets))

        num_classifier_per_dataset = np.zeros(len(datasets))


        for d_ind, dataset in enumerate(datasets):
            print(f'==========Dataset: {dataset}==========\n')
            dataset_result_path = os.path.join(results_folder_path, dataset)


            name = []
            mdl_losses = []
            print(f'------Loss: {loss}')
            ### load all the results
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
                # in this case, only 1 loss since the model is deterministic
                mdl_loss = np.repeat(mdic[loss], 10)
                mdl_losses.append(mdl_loss)
                name.append(method)

            for i, method in enumerate(rand_methods):
                wmrc_first_ind = 3
                if i < 3:
                    wmrc_ind = None
                    method_result_path = os.path.join(dataset_result_path, method)
                    method_fn = method + '_' + dataset + '.mat'
                else:
                    wmrc_ind = i - wmrc_first_ind
                    method_result_path = os.path.join(dataset_result_path, wmrc_folder[wmrc_ind])
                    method_fn = 'WMRC_' + dataset + wmrc_fn_ends[wmrc_ind]
                method_fn_full = os.path.join(method_result_path, method_fn)

                # load data
                mdic = sio.loadmat(method_fn_full)
                mdl_losses.append(mdic[loss].squeeze())

                if labeled_datasets and i == 2:
                    amcl_cc_avg_fit_time[d_ind] = mdic['fit_elapsed_time_mean']

                # retrieve avg classifiers, avg labeled data per classifier
                # when we're using 100 labeled datapoints, regular wmrc
                if wmrc_ind == 0:
                    wmrc_avg_labeled_pts_per_classifier[d_ind] = mdic["avg_num_labeled_per_rule_mean"]
                    wmrc_avg_num_classifier_available[d_ind] = mdic["n_rules_used_mean"]
                    wmrc_avg_fit_time[d_ind] = mdic['fit_elapsed_time_mean']
                    num_classifier_per_dataset[d_ind] = len(mdic["rule_weights"][0])

                name.append(method)

            # load WMRC oracle to aggregate its results
            method_result_path = os.path.join(dataset_result_path, wmrc_folder[0])
            method_fn = 'WMRC_' + dataset + '_accuracy_semisup_trainlabels_eqconst.mat'
            wmrc_oracle_fn_full = os.path.join(method_result_path, method_fn)
            mdic = sio.loadmat(wmrc_oracle_fn_full)
            wmrc_oracle_loss[d_ind] = mdic[loss]

            # permute the methods so they're in the order we want
            if labeled_datasets:
                # wrench dataset permutation
                name = swap_eles(name, [2, 6, 7, 4, 3, 1, 5], [7, 5, 6, 1, 2, 3, 4])
                mdl_losses = swap_eles(mdl_losses, [2, 6, 7, 4, 3, 1, 5], [7, 5, 6, 1, 2, 3, 4])

            else:
                # crowdsourced dataset permutation
                name = swap_eles(name, [4, 1, 3, 2], [1, 3, 2, 4])
                mdl_losses = swap_eles(mdl_losses, [], [])

            mean_mdl_losses = []
            for mdl_loss in mdl_losses:
                mean_mdl_losses.append(np.round(np.mean(mdl_loss),4))

            print(mean_mdl_losses)

            agg_loss[:, d_ind] = mean_mdl_losses

            sorted_inds = np.argsort(mean_mdl_losses)
            sorted_loss = np.sort(mean_mdl_losses)

            best_mthd_ind = sorted_inds[0]
            indist_inds = [best_mthd_ind]
            dist_results[best_mthd_ind, d_ind] = 1

            print(f'Best Method: {name[best_mthd_ind]}'
                    f' {mean_mdl_losses[best_mthd_ind]}')
            # print(f'--Methods indistinguishable by 2 sided t-test p=0.05--')
            indist = []
            for i, mthd in enumerate(name):
                if i == best_mthd_ind:
                    continue

                # if identical arrays, we don't want to run a ttest
                res_norm = np.linalg.norm(mdl_losses[best_mthd_ind] - mdl_losses[i])
                if np.isclose(res_norm, 0):
                    p_val = 100
                else:
                    _, p_val = ttest_rel(mdl_losses[best_mthd_ind], mdl_losses[i])

                # print(name[i], mean_mdl_losses[i], p_val)
                # print(mdl_losses[i])
                if p_val > 0.05:
                    indist.append(name[i] + f'({mean_mdl_losses[i]})')
                    dist_results[i, d_ind] = 1
                    indist_inds.append(i)

            if len(indist) > 0:
                print()
                print('Indistinguishable (p=0.05): ', end='')
                print(*indist, sep=', ')

            print()

            res_mdic[loss + "_ttest"] = dist_results
            res_mdic[loss] = agg_loss
            res_mdic['wmrc_oracle_' + loss] = wmrc_oracle_loss
            if labeled_datasets:
                res_mdic['wmrc_avg_labeled_pts_per_classifier'] = wmrc_avg_labeled_pts_per_classifier
                res_mdic['wmrc_avg_num_classifier_available'] = wmrc_avg_num_classifier_available
                res_mdic['num_classifier_per_dataset'] = num_classifier_per_dataset
                res_mdic['wmrc_avg_fit_time'] = wmrc_avg_fit_time
                res_mdic['amcl_cc_avg_fit_time'] = amcl_cc_avg_fit_time

        res_mdic["method_names"] = name

    if labeled_datasets:
        fn = 'loss_labeled_ttest.mat'
    else:
        fn = 'loss_unlabeled_ttest.mat'
    savemat(os.path.join(results_folder_path, fn), res_mdic)
