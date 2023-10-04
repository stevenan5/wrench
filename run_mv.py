import os
import logging

import numpy as np
from scipy.io import savemat
import scipy.io as sio

from wrench._logging import LoggingHandler
from wrench.labelmodel import MajorityVoting
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_mv(
        dataset_prefix,
        dataset_name=None,
        use_test=False,
        save_path=None,
        n_runs=1,
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    n_classes = np.max(data['train_labels']) + 1

    if use_test:
        test_data = [data['test_pred'], data['test_labels']]

    #### Run label model: MV
    label_model = MajorityVoting()

    for run_no in range(n_runs):
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

        # only run calibration code if n_classes is 2 and you're on the 1st run
        if n_classes == 2 and run_no == 0:
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
                    }
            if n_classes == 2:
                mdic["x_calibration_train"] = [prob_pred_train]
                mdic["y_calibration_train"] = [prob_true_train]

        mdic["pred_train"].append(Y_p_train)
        mdic["log_loss_train"].append(logloss_train)
        mdic["brier_score_train"].append(brier_score_train)
        mdic["acc_train"].append(acc_train)
        mdic["err_train"].append(1 - acc_train)
        mdic["num_rule"] = train_data[0].shape[1]

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
            if n_classes == 2:
                mdic_test["x_calibration_test"] = [prob_pred_test]
                mdic_test["y_calibration_test"] = [prob_true_test]

                mdic.update(mdic_test)

            mdic["pred_test"].append(Y_p_test)
            mdic["log_loss_test"].append(logloss_test)
            mdic["brier_score_test"].append(brier_score_test)
            mdic["acc_test"].append(acc_test)
            mdic["err_test"].append(1 - acc_test)

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

    result_filename = get_result_filename(dataset)

    savemat(os.path.join(save_path, result_filename), mdic)

    return mdic

def get_result_filename(dataset_name):
    filename = "MV_"\
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
    datapoint_loss = np.sum(sq_diff, axis=1)

    return np.mean(datapoint_loss)

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

    for dataset in datasets:
        # make result folder if it doesn't exist
        dataset_result_path = os.path.join(results_folder_path, dataset)
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)

        # make folder for MV specifically
        method_result_path = os.path.join(dataset_result_path, 'MV')
        if not os.path.exists(method_result_path):
            os.makedirs(method_result_path)

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
        logger.info('------------Running MV------------')
        logger.info('dataset: %s', dataset)

        run_mv(
                dataset_prefix,
                dataset_name = dataset,
                save_path=method_result_path,
                )
