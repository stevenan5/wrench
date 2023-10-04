import os
import json
import logging

import numpy as np
import scipy as sp
import numpy.random
from numpy.matlib import repmat
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

from wrench.dataset import load_image_dataset
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import EBCC
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_ebcc(
        dataset_prefix,
        dataset_name=None,
        use_test=False,
        save_path=None,
        n_runs=10,
        num_samples=500,
        replot=False,
        get_confidences=True
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    num_class = np.max(data['train_labels']) + 1
    result_filename = get_result_filename(dataset)

    if use_test:
        test_data = [data['test_pred'], data['test_labels']]

    #### Run label model: EBCC

    mdic = None
    for run_no in range(n_runs):
        if not replot:
            if n_runs > 1:
                logger.info('--------Run Number %d--------', run_no + 1)

            # remake the model to force the seed to change
            label_model = EBCC(
                    num_groups=10,
                    repeat=100,
                    inference_iter=100,
                    empirical_prior=True)
            label_model.fit(train_data)

            ### make predictions
            Y_p_train = label_model.predict_proba(train_data)
            # pred_train = np.argmax(Y_p_train, axis=1)
            true_labels_train = np.squeeze(train_data[1])
            if use_test:
                Y_p_test = label_model.predict_proba(test_data)
                # pred_test = np.argmax(Y_p_test, axis=1)
                true_labels_test = np.squeeze(test_data[1])

            ### compute losses
            brier_score_train = multi_brier(true_labels_train, Y_p_train)
            logloss_train = log_loss(true_labels_train,Y_p_train)
            acc_train = label_model.test(train_data, 'acc')

            # only run calibration code if num_class is 2 and you're on the 1st run
            if num_class == 2 and run_no == 0:
                prob_true_train, prob_pred_train = calibration_curve(
                        np.squeeze(train_data[1]),
                        np.clip(Y_p_train[:, 1], 0, 1),
                        n_bins=10)
                if use_test:
                    prob_true_test, prob_pred_test = calibration_curve(
                            np.squeeze(test_data[1]),
                            np.clip(Y_p_test[:, 1], 0, 1),
                            n_bins=10)

            if run_no == 0:
                mdic = {
                        "pred_train": [],
                        "log_loss_train": [],
                        "true_labels_train": true_labels_train,
                        "brier_score_train": [],
                        "acc_train": [],
                        "err_train": [],
                        # now do EBCC params
                        "seed": [],
                        "eta_km": [],
                        "nu_k": [],
                        "mu_jkml": [],
                        "rho_ikm": [],
                        }
                if num_class == 2:
                    mdic["x_calibration_train"] = []
                    mdic["y_calibration_train"] = []

            mdic["pred_train"].append(Y_p_train)
            mdic["log_loss_train"].append(logloss_train)
            mdic["brier_score_train"].append(brier_score_train)
            mdic["acc_train"].append(acc_train)
            mdic["err_train"].append(1 - acc_train)
            mdic["num_rule"] = train_data[0].shape[1]
            if num_class == 2:
                mdic["x_calibration_train"].append(prob_pred_train)
                mdic["y_calibration_train"].append(prob_true_train)

            # save computed parameters from ebcc
            params = label_model.params
            mdic["seed"].append(params["seed"])
            mdic["eta_km"].append(params["eta_km"])
            mdic["nu_k"].append(params["nu_k"])
            mdic["mu_jkml"].append(params["mu_jkml"])
            mdic["rho_ikm"].append(params["rho_ikm"])

            if use_test:
                brier_score_test = multi_brier(true_labels_test, Y_p_test)
                logloss_test = log_loss(true_labels_test, Y_p_test)
                acc_test = label_model.test(test_data, 'acc')
                mdic_test = {
                            "pred_test": [],
                            "true_labels_test": true_labels_test,
                            "log_loss_test": [],
                            "brier_score_test": [],
                            "acc_test": [],
                            "err_test": [],
                            }
                if num_class == 2:
                    mdic_test["x_calibration_test"] = []
                    mdic_test["y_calibration_test"] = []

                    mdic.update(mdic_test)

                mdic["pred_test"].append(Y_p_test)
                mdic["log_loss_test"].append(logloss_test)
                mdic["brier_score_test"].append(brier_score_test)
                mdic["acc_test"].append(acc_test)
                mdic["err_test"].append(1 - acc_test)
                if num_class == 2:
                    mdic_test["x_calibration_test"].append(prob_pred_test)
                    mdic_test["y_calibration_test"].append(prob_true_test)

            ### report results
            logger.info('================Results================')
            logger.info('train acc (train err): %.4f (%.4f)',
                    acc_train, 1 - acc_train)
            logger.info('train log loss: %.4f', logloss_train)
            logger.info('train brier score: %.4f', brier_score_train)
            if use_test:
                logger.info('test acc (test err): %.4f (%.4f)',
                        acc_test, 1 - acc_test)
                logger.info('test log loss: %.4f', logloss_test)
                logger.info('test brier score: %.4f', brier_score_test)

        # compute confidence intervals if it's the first run
        if (run_no == 0 and get_confidences) or replot:
            logger.info('------Computing Confidence Intervals------')
            if replot:
                mdic = sio.loadmat(os.path.join(save_path, result_filename))

            L = train_data[0]
            y = train_data[1]

            # for key, _ in mdic.items():
            #     print(key)

            num_pts, _ , num_factor = mdic["rho_ikm"][0].shape
            num_rule = mdic["num_rule"]

            # make y one hot encoded
            y_aug = np.squeeze(
                    (np.arange(num_class) == y.T[..., None]).astype(int))

            def joint_simp(tau, G, pi, V, L):
                # expecting tau to be a 1d vector of length num_class
                # expecting pi to be a 2d array of size (num_class, num_factor)
                # expecting V to be 4d array of size (num_rule, num_class,
                # num_factor, num_class)
                res = np.zeros((num_pts, num_class))
                res += np.log(tau)
                for i in range(num_pts):
                    res[i, :] += np.log(pi[:, G[i]])
                    for j in range(num_rule):
                        if L[i, j] != -1:
                            res[i, :] += np.log(V[j, :, G[i], L[i, j]])

                return sp.special.softmax(res, axis=1)

            def sample_q_no_z(nu, q_g, eta, mu, n_samples = 5000):
                # sample tau
                nu = nu.squeeze()
                tau = np.random.dirichlet(nu, size=n_samples)
                G = np.zeros((num_pts, n_samples))
                for i in range(num_pts):
                    G[i, :] = np.argmax(
                            np.random.multinomial(1, q_g[i, :], size=n_samples),
                                axis=1)
                pi = np.zeros((n_samples, num_class, num_factor))
                for k in range(num_class):
                    pi[:, k, ...] = np.random.dirichlet(eta[k, :],
                            size=n_samples)
                V = np.zeros((n_samples, num_rule, num_class, num_factor,
                    num_class))
                for j in range(num_rule):
                    for k in range(num_class):
                        for m in range(num_factor):
                            V[:, j, k, m, :] = \
                                    np.random.dirichlet(mu[j, k, m, :],
                                    size=n_samples)

                return tau, G, pi, V

            if not replot:
                q_g = np.clip(mdic["rho_ikm"][0].sum(axis=1), 0, 1)
                tau, G, pi, V = sample_q_no_z(mdic["nu_k"][0], q_g,
                        mdic["eta_km"][0], mdic["mu_jkml"][0],
                        n_samples=num_samples)

                G = G.astype(int)
                L = L.astype(int)

                results = np.zeros((num_samples, num_pts, num_class))
                for s in range(num_samples):
                    s = int(s)
                    results[s, ...] = joint_simp(
                            tau[s, :],
                            G[:, s],
                            pi[s, ...],
                            V[s, ...],
                            L)

                # get empirical quantiles
                coverage = 0.95
                lb = (1 - coverage) / 2
                ub = (1 + coverage) / 2
                # first index is lower bound or upper bound
                # second index is datapoint
                # third index is class
                cis = np.quantile(results, [lb, ub], axis=0)
                # round to four places after the decimal to get rid of
                # tiny numbers
                cis = np.round(cis, decimals=4)

                # compute how many confidence intervals are actually valid
                num_invalid_cis = np.sum(y_aug < cis[0, ...])
                num_invalid_cis += np.sum(cis[1, ...] < y_aug)
                mdic["cis_ebcc"] = cis
                mdic["ci_num_sample"] = num_samples
                mdic["num_invalid_cis"] = num_invalid_cis

                # print number of invalid intervals
                total_cis = num_pts * num_class
                pct_invalid = num_invalid_cis / total_cis * 100
                logger.info(f"number invalid intervals %d (%2.2f %%)", num_invalid_cis, pct_invalid)

                # print out min/max/mean confidence interval width
                diff = cis[1, ...] - cis[0, ...]
                logger.info("minimum CI width: %.4f", np.min(diff))
                logger.info("maximum CI width: %.4f", np.max(diff))
                logger.info("mean CI width: %.4f", np.mean(diff))

            if replot:
                cis = mdic["cis_ebcc"]
                diff = cis[1, ...] - cis[0, ...]
                num_samples = mdic["ci_num_sample"].squeeze()
                num_invalid_cis = mdic["num_invalid_cis"]

            name = get_result_filename(dataset_name)[:-4]
            ci_file_name = name + '_' + str(num_samples) + "samples" + str(num_invalid_cis.squeeze()) + "invalid.eps"
            full_ci_fn = os.path.join(save_path, ci_file_name)
            plot_confidence_intervals(full_ci_fn, dataset_name, diff,
                    num_samples, num_invalid_cis)

            if replot:
                break

    # if number of runs is >1, report and store mean results and standard
    # deviations
    if n_runs > 1 and not replot:
        mdic["log_loss_train_mean"]     = np.mean(mdic["log_loss_train"])
        mdic["brier_score_train_mean"]  = np.mean(mdic["brier_score_train"])
        mdic["acc_train_mean"]          = np.mean(mdic["acc_train"])
        mdic["err_train_mean"]          = np.mean(mdic["err_train"])

        mdic["log_loss_train_std"]     = np.std(mdic["log_loss_train"])
        mdic["brier_score_train_std"]  = np.std(mdic["brier_score_train"])
        mdic["acc_train_std"]          = np.std(mdic["acc_train"])
        mdic["err_train_std"]          = np.std(mdic["err_train"])

        if use_test:
            mdic["log_loss_test_mean"]    = np.mean(mdic["log_loss_test"])
            mdic["brier_score_test_mean"] = np.mean(mdic["brier_score_test"])
            mdic["acc_test_mean"]         = np.mean(mdic["acc_test"])
            mdic["err_test_mean"]         = np.mean(mdic["err_test"])

            mdic["log_loss_test_std"]    = np.std(mdic["log_loss_test"])
            mdic["brier_score_test_std"] = np.std(mdic["brier_score_test"])
            mdic["acc_test_std"]         = np.std(mdic["acc_test"])
            mdic["err_test_std"]         = np.std(mdic["err_test"])

        logger.info('================Aggregated Results================')
        logger.info('Total number of runs: %d', n_runs)
        logger.info('train mean acc +- std (mean train err):'
                ' %.4f +- %.4f (%.4f)', mdic['acc_train_mean'],
                mdic['acc_train_std'], mdic['err_train_mean'])
        logger.info('train mean log loss +- std: %.4f +- %.4f',
                mdic['log_loss_train_mean'], mdic['log_loss_train_std'])
        logger.info('train mean brier score +- std: %.4f +- %.4f',
                mdic['brier_score_train_mean'], mdic['brier_score_train_std'])

        if use_test:
            logger.info('test mean acc +- std (mean test err):'
                    ' %.4f +- %.4f (%.4f)', mdic['acc_test_mean'],
                    mdic['acc_test_std'], mdic['err_test_mean'])
            logger.info('test mean log loss +- std: %.4f +- %.4f',
                    mdic['log_loss_test_mean'], mdic['log_loss_test_std'])
            logger.info('test mean brier score +- std: %.4f +- %.4f',
                    mdic['brier_score_test_mean'], mdic['brier_score_test_std'])

    if not replot:
        savemat(os.path.join(save_path, result_filename), mdic)

    return mdic

def get_result_filename(dataset_name):
    filename = "EBCC_"\
            + dataset_name + ".mat"

    return filename

def multi_brier(labels, pred_probs):
    """
    multiclass brier score
    assumes labels are a 1D vector with values in {0, 1, n_class - 1}
    """
    n_class = int(np.max(labels) + 1)
    labels = (np.arange(n_class) == labels[..., None]).astype(int)
    sq_diff = np.square(labels - pred_probs)
    datapoint_loss =  np.sum(sq_diff, axis=1)

    return np.mean(datapoint_loss)

def plot_confidence_intervals(file_name, dataset_name, ci_diffs, num_samples,
        num_invalid):

    num_class = ci_diffs.shape[1]
    num_pts = ci_diffs.shape[0]
    ci_diffs = ci_diffs.flatten()

    fig = plt.figure()
    counts, edges, _ = plt.hist(ci_diffs, bins='auto')
    # don't print counts because that's too much clutter
    # for ind, count in enumerate(counts):
    #     if count > 0:
    #         bin_loc = (edges[ind] + edges[ind + 1])/2
    #         plt.text(bin_loc, count, str(int(count)), ha='center')

    plt.xlabel('Confidence Interval Width')
    plt.ylabel(f'Count (k={num_class} * n={num_pts})')

    # plt.title(f'{dataset_name} EBCC CI widths, {num_samples} samples, '
    #         f'{num_invalid} invalid')

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()

# pylint: disable=C0103
if __name__ == '__main__':
    # create results folder if it doesn't exist
    results_folder_path = './results'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # path for config jsons
    dataset_prefix = './datasets/'

    # wrench datasets
    datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
            'imdb', 'obs', 'sms', 'yelp', 'youtube']
    # crowdsourcing datasets
    # datasets += ['bird', 'rte', 'dog', 'web']

    replot_figs = False
    # replot_figs = True

    for dataset in datasets:
        # make result folder if it doesn't exist
        dataset_result_path = os.path.join(results_folder_path, dataset)
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        # make folder for EBCC specifically
        method_result_path = os.path.join(dataset_result_path, 'EBCC')
        if not os.path.exists(method_result_path):
            os.makedirs(method_result_path)

        if not replot_figs:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            formatter = logging.Formatter('%(asctime)s - %(message)s',
                    '%Y-%m-%d %H:%M:%S')

            # do some formatting for log name
            log_filename = get_result_filename(dataset)[:-4] + '.log'
            log_filename_full = os.path.join(method_result_path,
                    log_filename)
            file_handler = logging.FileHandler(log_filename_full, 'w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # log all the run parameters
        logger.info('------------Running EBCC------------')
        logger.info('dataset: %s', dataset)

        run_ebcc(
                dataset_prefix,
                dataset_name = dataset,
                save_path=method_result_path,
                replot=replot_figs,
                )
