import os
import json
import logging
from time import perf_counter

import numpy as np
from numpy.matlib import repmat
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

from wrench._logging import LoggingHandler
from wrench.labelmodel import WMRC
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_wmrc(
        dataset_prefix,
        dataset_name=None,
        n_classes=0,
        constraint_form='accuracy',
        add_mv_const=False,
        labeled_set='valid',
        bound_method='binomial',
        use_inequality_consts=True,
        get_confidences=True,
        pattern_neighborhood_size = 20,
        n_max_labeled = -1,
        pred_prob_incr = 5,
        reject_threshold = 40,
        n_runs = 1,
        use_test=True,
        verbose=False,
        # verbose=True,
        save_path=None,
        solver='MOSEK',
        conf_solver='GUROBI',
        n_fit_tries=10,
        bound_scale=1,
        conf_objs=['preds', 'xent'],
        # conf_objs=['preds'],
        replot=False,
        logger=logger,
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    n_train_points = train_data[0].shape[0]
    n_train_rules = train_data[0].shape[1]
    if add_mv_const:
        n_train_rules += 1

    # result path
    # only add datapoint count if we're not oracle
    if n_max_labeled > 0:
        n_labeled = n_max_labeled
    else:
        n_labeled = None
    result_filename = get_result_filename(dataset, constraint_form,
            labeled_set, bound_method, use_inequality_consts, add_mv_const,
            n_labeled=n_labeled)

    use_all_valid = False
    try:
        valid_data = [data['val_pred'], data['validation_labels']]
        use_all_valid = labeled_set == 'valid' and\
                n_max_labeled == valid_data[0].shape[0]
    except KeyError:
        print('No validation set found!')

    if use_test:
        test_data = [data['test_pred'], data['test_labels']]

    if labeled_set == 'valid':
        labeled_data = valid_data
    else:
        labeled_data = train_data

    ### if in oracle setting, always force n_runs to be 1
    # or if we're using all validation points or replotting
    is_oracle = labeled_set == 'train' and not use_inequality_consts
    if is_oracle or use_all_valid or bound_method == 'unsupervised' or replot:
        n_runs = 1

    #### Run label model: WMRC
    label_model = WMRC(solver=solver, conf_solver=conf_solver, verbose=verbose)

    for run_no in range(n_runs):
        if not replot:
            if n_runs > 1:
                logger.info('------------Run Number %d------------', run_no + 1)
            n_fit_runs = 0
            # reset problem status so we can refit
            label_model.prob_status = None
            while label_model.prob_status != 'optimal' and\
                    n_fit_runs < n_fit_tries:
                n_fit_runs += 1
                start_time = perf_counter()
                label_model.fit(
                        train_data,
                        labeled_data,
                        constraint_type=constraint_form,
                        bound_method=bound_method,
                        use_inequality_consts=use_inequality_consts,
                        majority_vote=add_mv_const,
                        n_max_labeled=n_max_labeled,
                        bound_scale=bound_scale,
                        logger=logger,
                        )
                end_time = perf_counter()

                # increase size of bounds if uncertainty set is infeasible
                if bound_method == 'unsupervised'\
                        and label_model.prob_status != 'optimal':
                    bound_scale += 0.1

            elapsed_time = end_time - start_time

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

            # only run calibration code if n_classes is 2
            if n_classes == 2:
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
                        "n_rules_used": [],
                        "rule_weights": [],
                        "class_freq_weights": [],
                        "avg_num_labeled_per_rule": [],
                        "fit_elapsed_time": [],
                        "wmrc_xent_ub": [],
                        }
                if n_classes == 2:
                    mdic["x_calibration_train"] = []
                    mdic["y_calibration_train"] = []

            mdic["pred_train"].append(Y_p_train)
            mdic["log_loss_train"].append(logloss_train)
            mdic["brier_score_train"].append(brier_score_train)
            mdic["acc_train"].append(acc_train)
            mdic["err_train"].append(1 - acc_train)
            mdic["n_rules_used"].append(label_model.n_rules_used)
            mdic["rule_weights"].append(label_model.param_wts)
            mdic["class_freq_weights"].append(label_model.class_freq_wts)
            mdic["avg_num_labeled_per_rule"].append(\
                    label_model.avg_labeled_per_rule)
            mdic["fit_elapsed_time"].append(elapsed_time)
            # optimization problem finds the negative entropy and is not
            # averaged over the total number of datapoints
            mdic["wmrc_xent_ub"].append(-1 * label_model.prob.value\
                    / n_train_points)
            if n_classes == 2:
                mdic["x_calibration_train"].append(prob_pred_train)
                mdic["y_calibration_train"].append(prob_true_train)

            # if we're in the unsupervised setting and the bound scale changed,
            # record that. note the only time bound_scale changes is when
            # unsupervised bounds are used.
            if bound_scale != 1:
                mdic["bound_scale"] = bound_scale

            if use_test:
                brier_score_test = multi_brier(true_labels_test, Y_p_test)
                logloss_test = log_loss(true_labels_test, Y_p_test)
                acc_test = label_model.test(test_data, 'acc')
                if run_no == 0:
                    mdic_test = {
                                "pred_test": [],
                                "true_labels_test": true_labels_test,
                                "log_loss_test": [],
                                "brier_score_test": [],
                                "acc_test": [],
                                "err_test": [],
                                }
                    if n_classes == 2:
                        mdic_test["x_calibration_test"] = []
                        mdic_test["y_calibration_test"] = []

                    mdic.update(mdic_test)

                mdic["pred_test"].append(Y_p_test)
                mdic["log_loss_test"].append(logloss_test)
                mdic["brier_score_test"].append(brier_score_test)
                mdic["acc_test"].append(acc_test)
                mdic["err_test"].append(1 - acc_test)
                if n_classes == 2:
                    mdic_test["x_calibration_test"].append(prob_pred_test)
                    mdic_test["y_calibration_test"].append(prob_true_test)

            ### report results
            logger.info('================Results================')
            logger.info('time to fit: %.1f seconds', elapsed_time)
            logger.info('train acc (train err): %.4f (%.4f)',
                    acc_train, 1 - acc_train)
            logger.info('wmrc train log loss upper bound %.4f',
                    mdic['wmrc_xent_ub'][-1])
            logger.info('train log loss: %.4f', logloss_train)
            logger.info('train brier score: %.4f', brier_score_train)
            if use_test:
                logger.info('test acc (test err): %.4f (%.4f)',
                        acc_test, 1 - acc_test)
                logger.info('test log loss: %.4f', logloss_test)
                logger.info('test brier score: %.4f', brier_score_test)
        else:
            mdic = sio.loadmat(os.path.join(save_path, result_filename))

        # get confidences if it's oracle, or when 100 labeled points from
        # the validation set are used.  also must not add MV constraint
        valid_100 = labeled_set == 'valid' and n_max_labeled == 100\
                and not add_mv_const
        compute_confidences = get_confidences and (is_oracle or valid_100)
        replot_confidences = replot and (is_oracle or valid_100)
        if (compute_confidences and run_no == 0) or replot_confidences:
            if not is_oracle and n_max_labeled > 0:
                n_labeled = n_max_labeled
            else:
                n_labeled = None
            logger.info('------Computing Confidence Intervals------')
            # first return value are the confidence intervals for the groups
            # second return value are the mean predictions for each group
            # third return value is the mean ground truth labels for each group

            # try to load the oracle results
            oracle_filename = get_result_filename(dataset,
                    constraint_form, 'train', 'binomial', False, add_mv_const)
            oracle_filename_full = os.path.join(save_path, oracle_filename)

            oracle_res_exist = os.path.isfile(oracle_filename_full)
            combine_intervals = oracle_res_exist and not is_oracle

            if combine_intervals:
                oracle_mdic = sio.loadmat(oracle_filename_full)
            else:
                oracle_mdic = None
                logger.info('oracle results not found, won\'t combine graphs')

            # try to load eta if it's in the dataset
            if 'eta_train' in data:
                eta_train = data['eta_train'].reshape((-1, n_classes))
            else:
                eta_train = None

            file_name_cis = get_result_filename(dataset_name,
                    constraint_form, labeled_set, bound_method,
                    use_inequality_consts, add_mv_const,
                    n_labeled=n_labeled)[:-4]

            for conf_obj in conf_objs:
                if conf_obj == 'preds':
                    conf_res_folder = 'Probability_Confidences'
                elif conf_obj == 'xent':
                    conf_res_folder = 'Cross_Entropy_Confidences'
                save_path_conf = os.path.join(save_path, conf_res_folder)

                if not os.path.exists(save_path_conf):
                    os.makedirs(save_path_conf)

                # don't combine intervals if we're doing cross entropy
                # confidences.  IMPORTANT THAT YOU RUN conf_obj == 'preds'
                # FIRST IF WANTED.
                if conf_obj == 'xent':
                    combine_intervals = False

                logger.info('-----Computing %s-----', conf_res_folder)
                ### pattern neighborhoods
                file_name_pn = file_name_cis + '_' + conf_obj\
                        + '_pattern_neigh_'\
                        + str(pattern_neighborhood_size)\
                        + '.eps'
                file_name_pn_full = os.path.join(save_path_conf, file_name_pn)

                mdic_conf_pattern_neigh = compute_plot_save_confidences(
                        file_name_pn_full, dataset_name, constraint_form,
                        label_model, 'pattern_neigh', train_data, logger,
                        n_classes, is_oracle, mdic, oracle_mdic, objective=conf_obj,
                        pattern_neigh_size=pattern_neighborhood_size,
                        eta_train=eta_train,
                        combine_intervals=combine_intervals, replot=replot
                        )

                mdic.update(mdic_conf_pattern_neigh)

                ### prediction percentiles
                thresholds = np.arange(100 - pred_prob_incr,\
                        100/n_classes - pred_prob_incr, -pred_prob_incr)
                thresholds[-1] = int(100/n_classes)

                file_name_pp = file_name_cis  + '_' + conf_obj\
                        + '_pred_prob' + '_incr' + str(pred_prob_incr) + '.eps'
                file_name_pp_full = os.path.join(save_path_conf, file_name_pp)

                mdic_conf_pred_prob = compute_plot_save_confidences(
                        file_name_pp_full, dataset_name, constraint_form,
                        label_model, 'pred_prob', train_data, logger,
                        n_classes, is_oracle, mdic, oracle_mdic, objective=conf_obj,
                        eta_train=eta_train, thresholds=thresholds,
                        combine_intervals=combine_intervals, replot=replot
                        )

                mdic.update(mdic_conf_pred_prob)

                ### prediction percentiles lower bound only
                incr = 10
                thresholds_lb_only = np.arange(100 - incr,\
                        100/n_classes - incr, - incr)
                # just in case n_classes is 3 or something
                thresholds_lb_only[-1] = int(100/n_classes)

                file_name_pp_lb = file_name_cis + '_' + conf_obj \
                         + '_pred_prob_lb_only' + '_incr' + str(incr)\
                         + '.eps'
                file_name_pp_lb_full = os.path.join(save_path_conf, file_name_pp_lb)

                mdic_conf_pred_prob_lb = compute_plot_save_confidences(
                        file_name_pp_lb_full, dataset_name, constraint_form,
                        label_model, 'pred_prob_lb', train_data, logger,
                        n_classes, is_oracle, mdic, oracle_mdic, objective=conf_obj,
                        eta_train=eta_train, thresholds=thresholds_lb_only,
                        combine_intervals=False, replot=replot
                        )

                mdic.update(mdic_conf_pred_prob_lb)

                ### prediction percentiles lower bound only, combined classes
                incr_comb_class = 5
                thresholds_lb_only_comb_class = np.arange(100 - incr_comb_class,\
                        100/n_classes - incr_comb_class, - incr_comb_class)
                # just in case n_classes is 3 or something
                thresholds_lb_only_comb_class[-1] = int(100/n_classes)

                file_name_pp_lb_cc = file_name_cis + '_' + conf_obj \
                         + '_pred_prob_lb_only_comb_class' + '_incr' + str(incr_comb_class)\
                         + '.eps'
                file_name_pp_lb_cc_full = os.path.join(save_path_conf, file_name_pp_lb_cc)

                mdic_conf_pred_prob_lb_cc = compute_plot_save_confidences(
                        file_name_pp_lb_cc_full, dataset_name, constraint_form,
                        label_model, 'pred_prob_lb_comb_class', train_data, logger,
                        n_classes, is_oracle, mdic, oracle_mdic, objective=conf_obj,
                        eta_train=eta_train, thresholds=thresholds_lb_only_comb_class,
                        combine_intervals=False, replot=replot, skip_plot=True
                        )

                mdic.update(mdic_conf_pred_prob_lb_cc)

                ### patterns
                if mdic["n_patterns"] < reject_threshold:
                    file_name_p = file_name_cis + '_' + conf_obj +'_pattern.eps'
                    file_name_p_full = os.path.join(save_path_conf, file_name_p)

                    mdic_conf_pattern = compute_plot_save_confidences(
                            file_name_p_full, dataset_name, constraint_form,
                            label_model, 'pattern', train_data, logger,
                            n_classes, is_oracle, mdic, oracle_mdic, objective=conf_obj,
                            pattern_neigh_size=pattern_neighborhood_size,
                            eta_train=eta_train,
                            combine_intervals=combine_intervals, replot=replot
                            )

                    mdic.update(mdic_conf_pattern)
                else:
                    logger.info('too many patterns, skipping pattern confidence ints')

    # if number of runs is >1, report and store mean results and standard
    # deviations
    if n_runs > 1 and not replot:
        mdic["log_loss_train_mean"]     = np.mean(mdic["log_loss_train"])
        mdic["brier_score_train_mean"]  = np.mean(mdic["brier_score_train"])
        mdic["acc_train_mean"]          = np.mean(mdic["acc_train"])
        mdic["err_train_mean"]          = np.mean(mdic["err_train"])
        mdic["n_rules_used_mean"]       = np.mean(mdic["n_rules_used"])
        mdic["avg_num_labeled_per_rule_mean"] =\
                np.mean(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_mean"]   = np.mean(mdic["fit_elapsed_time"])
        mdic["wmrc_xent_ub_mean"]       = np.mean(mdic["wmrc_xent_ub"])

        mdic["log_loss_train_std"]     = np.std(mdic["log_loss_train"])
        mdic["brier_score_train_std"]  = np.std(mdic["brier_score_train"])
        mdic["acc_train_std"]          = np.std(mdic["acc_train"])
        mdic["err_train_std"]          = np.std(mdic["err_train"])
        mdic["avg_num_labeled_per_rule_std"] =\
                np.std(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_std"]   = np.std(mdic["fit_elapsed_time"])
        mdic["wmrc_xent_ub_std"]       = np.std(mdic["wmrc_xent_ub"])

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
        logger.info('Average time to fit: %.1f seconds (std: %.1f)',
                mdic['fit_elapsed_time_mean'], mdic['fit_elapsed_time_std'])
        logger.info('Average of %.2f rules out of %d had labeled data',
                mdic["n_rules_used_mean"], n_train_rules)
        logger.info('Average of %d labeled datapoints out of %d per rule (std: %.4f)',
                mdic["avg_num_labeled_per_rule_mean"], n_max_labeled,\
                        mdic["avg_num_labeled_per_rule_std"])
        logger.info('train mean acc +- std (mean train err):'
                ' %.4f +- %.4f (%.4f)', mdic['acc_train_mean'],
                mdic['acc_train_std'], mdic['err_train_mean'])
        logger.info('wmrc train log loss upper bound mean +- std: %.4f +- %.4f',
                mdic['wmrc_xent_ub_mean'], mdic['wmrc_xent_ub_std'])
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

def get_result_filename(dataset_name, constraint_name,
        labeled_set, bound_method, use_inequality_consts, add_mv_const,
        n_labeled=None):

    bound_name = ''
    if bound_method == 'unsupervised':
        est_pdgm = 'unsup_'
    else:
        est_pdgm = 'semisup_'
        if labeled_set == 'valid':
            bound_name = '_' + bound_method

    if use_inequality_consts:
        use_ineq = 'ineqconst'
    else:
        use_ineq = 'eqconst'

    if labeled_set == 'valid':
        labeled_set_used = 'validlabels_'
    else:
        labeled_set_used = 'trainlabels_'

    if add_mv_const:
        use_mv = '_add_mv'
    else:
        use_mv = ''

    n_lab = ''
    if n_labeled is not None:
        if n_labeled > 0 and use_inequality_consts and labeled_set == 'valid':
            n_lab = str(n_labeled)

    filename = "WMRC_"\
            + dataset_name + '_'\
            + constraint_name + '_'\
            + est_pdgm\
            + n_lab\
            + labeled_set_used\
            + use_ineq\
            + bound_name\
            + use_mv\
            + ".mat"

    return filename

def multi_brier(labels, pred_probs):
    """
    multiclass brier score
    assumes labels is 1d vector with elements in {0, 1, ..., n_class - 1}
    position of the ture class and 0 otherwise
    """
    n_class = int(np.max(labels) + 1)
    labels = (np.arange(n_class) == labels[..., None]).astype(int)
    sq_diff = np.square(labels - pred_probs)
    datapoint_loss = np.sum(sq_diff, axis=1)

    return np.mean(datapoint_loss)

def compute_plot_save_confidences(file_name, dataset_name, constraint_form,
        label_model, grouping_method, train_data, logger, n_classes, is_oracle,
        mdic, oracle_mdic,
        objective='preds', pattern_neigh_size=None, eta_train=None,
        thresholds=None, combine_intervals=False, replot=False,
        skip_plot=False):
    """
    file_name: prefix of the file name for the plotted confidences
    dataset_name: name of the dataset
    constraint_form: the type of constraint used, e.g. accuracy
    label_model: WMRC label model
    grouping_method: string name for how the datapoints are grouped
                    'pattern_neigh', 'pattern', 'pred_prob', 'pred_prob_lb',
                    'pred_prob_comb_class', 'pred_prob_lb_comb_class'
    train_data: actual training dataset
    logger: the logger used before calling this
    n_classes: number of classes
    is_oracle: boolean on whether an oracle instance is calling this
    mdic: dictionary of results
    oracle_mdic: dictionary of oracle results, only used if combining graphs
    objective: whether the confidences are in terms of probabilities or Xent
    pattern_neigh_size: pattern neighborhood size, used to compute selections
    eta_train: the 'true' underlying distribution, used for synthetic data
    thresholds: the probability thresholds to group datapoints by predicted prob
    combine_intervals: whether or not to make graphs where oracle info is there
    replot: whether to just load in pre-computed confidences and plot them
    """
    # determine if we're combining classes
    combined_class = grouping_method[-10:] == 'comb_class'

    ### load stuff from oracle confidences
    if combine_intervals and oracle_mdic is not None:

        # load variables that we want
        oracle_preds = oracle_mdic['pred_train'].squeeze() # always load this

        # make sure that the oracle cis' pattern neighborhood size is the same
        # as the current one we're using
        # trivially make this True because it doesn't matter for the non-
        # 'pattern_neigh' grouping methods. can possibly be set to false below
        same_pat_neigh_size = True

        if grouping_method in ['pattern_neigh', 'pattern']:
            oracle_cis = oracle_mdic[objective + '_cis_' + grouping_method + '_train']
            if objective == 'preds':
                oracle_mean_preds = oracle_mdic[objective + '_ci_' + grouping_method + '_mean_pred_train']
            else:
                oracle_mean_preds = None

            # we using the same pattern neighborhood size since it doens't
            # depend on the objective
            if grouping_method == 'pattern_neigh':
            #'ci_pattern_neighborhood_size' in oracle_mdic:
                same_pat_neigh_size = oracle_mdic['ci_pattern_neighborhood_size'] \
                        == pattern_neigh_size
        else:
            oracle_mean_preds = None

    else:
        logger.info('oracle results not found, skipping combined graphs')

    ### either load or compute confidences and other needed stuff
    if replot:
        # the label_model needs to know this so we can use certain functions
        label_model.n_class = n_classes

        cis_train = mdic[objective + '_cis_' + grouping_method + '_train']
        if objective == 'preds' and grouping_method[-10:] != 'comb_class':
            ci_mean_pred_train = mdic[objective + '_ci_' + grouping_method + '_mean_pred_train']
        else:
            ci_mean_pred_train = None
        ci_mean_gt_train = mdic[objective + '_ci_' + grouping_method + '_mean_gt_train']
        selections = mdic['groupings_' + grouping_method]
        n_patterns = mdic["n_patterns"] if 'n_patterns' in mdic else None
        if grouping_method in ['pred_prob', 'pred_prob_lb', 'pred_prob_comb_class', 'pred_prob_lb_comb_class']:
            percentile_ci_train = mdic['percentile_ci' + grouping_method[9:] + '_train']
    else:
        if objective == 'xent':
            wmrc_preds = mdic['pred_train'][0]
        elif objective == 'preds' and combined_class:
            # when put into cross entropy formula, the result simplifies to
            # 0-1 loss
            wmrc_preds = np.exp(-1 * mdic['pred_train'][0])
        else:
            wmrc_preds = None

        cis_train, ci_mean_pred_train, ci_mean_gt_train = \
                label_model.get_confidences(
                        train_data,
                        grouping=grouping_method,
                        neighborhood_size=pattern_neigh_size,
                        prediction_thresholds=thresholds,
                        wmrc_preds=wmrc_preds
                        )
        if grouping_method == 'pattern_neigh':
            selections = label_model.pattern_neigh_selections
        elif grouping_method == 'pattern':
            selections = label_model.pattern_selections
        elif grouping_method == 'pred_prob':
            selections = label_model.pred_prob_selections
        elif grouping_method == 'pred_prob_lb':
            selections = label_model.pred_prob_selections_lb_only
        elif grouping_method == 'pred_prob_comb_class':
            selections = label_model.pred_prob_selections_comb_class
        elif grouping_method == 'pred_prob_lb_comb_class':
            selections = label_model.pred_prob_selections_lb_only_comb_class

        # we will need this if we want to plot pattern CIs
        n_patterns = label_model.n_patterns

        # records which prediction probabilities were used to make the groups
        # moreover, the lower bounds are recorded, e.g. if an element is
        # 0.8 and incr=5, then the range of (0.85, 0.8) was used to form the
        # group.
        if grouping_method == 'pred_prob':
            percentile_ci_train = label_model.used_thresholds
        elif grouping_method == 'pred_prob_lb':
            percentile_ci_train = label_model.used_thresholds_lb_only
        elif grouping_method == 'pred_prob_comb_class':
            percentile_ci_train = label_model.used_thresholds_comb_class
        elif grouping_method == 'pred_prob_lb_comb_class':
            percentile_ci_train = label_model.used_thresholds_lb_only_comb_class
        else:
            percentile_ci_train = None

    ### plot the confidence intervals
    # the following makes it so we plot the oracle prediction as oracle preds
    # even when we're not combining intervals
    tmp_mean_preds = ci_mean_pred_train
    tmp_oracle_mean_preds = None
    if is_oracle:
        tmp_oracle_mean_preds = tmp_mean_preds
        tmp_mean_preds = None

    # we're not going to load eta_train from the synthetic datasets
    # because it's just faster to rerun and replot
    if eta_train is not None:
        mean_eta_train=label_model._mean_group_preds(selections, eta_train)
    else:
        mean_eta_train = None

    group_dist = np.sum(selections, axis=1) / np.sum(selections)
    pos_args = [file_name, dataset_name, n_classes,
            grouping_method, ci_mean_gt_train, cis_train, constraint_form,
            group_dist, objective]
    kw_args = {'mean_preds': tmp_mean_preds,
            'oracle_mean_preds':tmp_oracle_mean_preds,
            "mean_eta": mean_eta_train}

    if not skip_plot:
        plot_confidence_intervals(*pos_args, **kw_args)

        if combine_intervals and same_pat_neigh_size:
            file_name_comb = file_name[:-4] + '_combined.eps'

            pos_args[0] = file_name_comb
            kw_args["mean_preds"] = ci_mean_pred_train
            if grouping_method in ['pattern', 'pattern_neigh']:
                kw_args["max_conf_ints"] = cis_train
                pos_args[5] = oracle_cis
            elif grouping_method in ['pred_prob']:
                # recompute oracle mean preds since the groupings can change
                # between oracle and non-oracle runs
                oracle_mean_preds = label_model._mean_group_preds(
                        selections, oracle_preds)

            kw_args["oracle_mean_preds"] = oracle_mean_preds
            plot_confidence_intervals(*pos_args, **kw_args)

    elif combine_intervals and not same_pat_neigh_size:
        logger.info('Oracle had pattern neighborhood size %d, we have'
                'pattern neighborhood size %d',
                oracle_mdic['ci_pattern_neighborhood_size'],
                pattern_neigh_size)

    ### save the confidence intervals
    res_mdic = {}

    res_mdic[objective + '_cis_' + grouping_method + '_train'] = cis_train
    if ci_mean_pred_train is not None:
        res_mdic[objective + '_ci_' + grouping_method + '_mean_pred_train'] = ci_mean_pred_train
    res_mdic[objective + '_ci_' + grouping_method + '_mean_gt_train'] = ci_mean_gt_train
    res_mdic['groupings_' + grouping_method] = selections
    res_mdic["n_patterns"] = n_patterns

    if grouping_method in ['pred_prob', 'pred_prob_lb', 'pred_prob_comb_class', 'pred_prob_lb_comb_class']:
        res_mdic['percentile_ci' + grouping_method[9:] + '_train'] = percentile_ci_train
        logger.info(f'Used thresholds {percentile_ci_train} out of {thresholds}')

    if grouping_method == 'pattern_neigh':
        res_mdic['ci_pattern_neighborhood_size'] = pattern_neigh_size

    logger.info('finished computing %s confidences', grouping_method)

    return res_mdic

def plot_confidence_intervals(file_name, set_name, n_classes,
        grouping, pattern_gt, conf_ints, constr, group_dist, conf_objective,
        mean_preds=None, max_conf_ints=None, oracle_mean_preds=None,
        mean_eta=None):

    binary_data = n_classes == 2
    # going to require that at least 1 of mean_pred and oracle_mean_preds is
    # not None
    if conf_objective == 'preds' and mean_preds is None and oracle_mean_preds is None:
        raise ValueError("Both mean_pred and oracle_mean_preds can't be None"
                "when using probability confidences.")

    pattern_gt = pattern_gt.flatten()
    if mean_preds is not None:
        mean_preds = mean_preds.flatten()
    if oracle_mean_preds is not None:
        oracle_mean_preds = oracle_mean_preds.flatten()
    if mean_eta is not None:
        mean_eta = mean_eta.flatten()

    # figure out if we want condensed graph
    condensed_graph = binary_data and conf_objective == 'preds'

    # break up the confidence interval data so we can process it easily
    conf_int_ub = conf_ints[1, :]
    conf_int_lb = conf_ints[0, :]
    max_conf_int_ub = max_conf_ints[1, :] if max_conf_ints is not None else None
    max_conf_int_lb = max_conf_ints[0, :] if max_conf_ints is not None else None

    if condensed_graph:
        def reshape_when_not_none(arr):
            arr = arr.reshape((-1,2)) if arr is not None else None
            return arr

        var_list = [pattern_gt, mean_preds, oracle_mean_preds, mean_eta,
                conf_int_ub, conf_int_lb, max_conf_int_ub, max_conf_int_lb]
        pattern_gt, mean_preds, oracle_mean_preds, mean_eta,\
                conf_int_ub, conf_int_lb, max_conf_int_ub, max_conf_int_lb = \
                        list(map(reshape_when_not_none, var_list))

        if mean_preds is not None:
            ref_pred_dist = mean_preds
        else:
            ref_pred_dist = oracle_mean_preds

        # if grouping == pattern or pattern_neigh, then we will just look at
        # the probability of class 1.  otherwise, we'll take the class with
        # highest predicted probability
        if grouping in ['pattern', 'pattern_neigh']:
            label_sels = np.ones(ref_pred_dist.shape[0]).astype(int)
        elif grouping in ['pred_prob', 'pred_prob_lb']:
            label_sels = np.argmax(ref_pred_dist, axis=1).astype(int)

        def select_when_not_none(label_sels):
            def go(arr):
                if arr is not None:
                    n_pts = arr.shape[0]
                    arr = arr[np.arange(n_pts), label_sels]
                return arr
            return go

        # CBA to figure out why I need to remake this list to get the
        # reshaped variables.  I thought python lists were shallow copies?
        var_list = [pattern_gt, mean_preds, oracle_mean_preds, mean_eta,
                conf_int_ub, conf_int_lb, max_conf_int_ub, max_conf_int_lb]
        pattern_gt, mean_preds, oracle_mean_preds, mean_eta,\
                conf_int_ub, conf_int_lb, max_conf_int_ub, max_conf_int_lb = \
                list(map(select_when_not_none(label_sels), var_list))

    x = range(len(pattern_gt))

    # compute the heights of the data
    ref_dist = pattern_gt
    above = conf_int_ub - ref_dist
    below = conf_int_lb - ref_dist # made negative to show underestimation

    if max_conf_ints is not None:
        max_above = max_conf_int_ub - conf_int_ub
        max_below = max_conf_int_lb - conf_int_lb

    fig, ax = plt.subplots()
    fig.dpi = 1200
    # ax.set_rasterized(True)

    if max_conf_ints is not None:
        ub_label = 'Min Overestimate'
        lb_label = 'Min Underestimate'
    else:
        ub_label = 'Overestimate'
        lb_label = 'Underestimate'
    ax.bar(x, above, color='chartreuse', bottom=ref_dist, label=ub_label)
    ax.bar(x, -1 * below, color='orangered', bottom=conf_int_lb,
            label=lb_label)

    if max_conf_ints is not None:
        ax.bar(x, max_above, color='cornflowerblue',  bottom=conf_int_ub,
                # hatch='x', edgecolor='w', alpha=0.99,
                label='Extra Overestimate')
        ax.bar(x, -1 * max_below, color='gold', bottom=max_conf_int_lb,
                # hatch='x', edgecolor='w', alpha=0.99,
                label='Extra Understimate')

    # plot mean predictions
    mew = 1.5
    if conf_objective == 'preds':
        pattern_gt_label = 'Mean GT'
    elif conf_objective == 'xent':
        pattern_gt_label = 'Mean Actual Xent'
    ax.plot(x, pattern_gt, 'k+', markeredgewidth=mew, label=pattern_gt_label)
    if mean_preds is not None:
        ax.plot(x, mean_preds, 'kx', markeredgewidth=mew, label='Mean WMRC')
    if oracle_mean_preds is not None:
        ax.plot(x, oracle_mean_preds, 'ko', markeredgewidth=mew,
                markerfacecolor='none', label='Mean Oracle WMRC')
    if mean_eta is not None:
        ax.plot(x, mean_eta, 'kD', markeredgewidth=mew,
                markerfacecolor='none', label='Mean eta')

    # define n_classes variable for graph related computations
    n_classes_graph = 1 if condensed_graph else n_classes
    n_groups = int(pattern_gt.shape[0] / n_classes_graph)

    # plot vertical lines to separate the groups
    vert_lines_x = (n_classes_graph * np.arange(1, n_groups) - 0.5)
    ax.vlines(vert_lines_x, 0, 1, transform=ax.get_xaxis_transform(),
            colors='k', ls='-', lw=0.8, label='Group Sep.')

    if not binary_data:
        # plot horizontal line to show where random noise is
        ax.axhline(1 / n_classes, color='k', ls='--', lw=0.8,
                label='Random Noise')

    # add ticks to denote labels
    xticks = np.arange(pattern_gt.shape[0])
    # use our chosen label if binary classes
    if binary_data and conf_objective == 'preds':
        xticklabels = label_sels
    else:
        xticklabels = repmat(np.arange(n_classes), 1, n_groups)

    xticklabels = xticklabels.squeeze().astype(str).tolist()
    # in the case that xticklabels before squeezing/casting is a one element
    # array, then tolist() will not make it a list and the result will remain
    # a string.  this breaks the code that appends the fraction of points
    # of a group below
    if isinstance(xticklabels, str):
        xticklabels = [xticklabels]

    # compute the percentage of points in a certain group and label
    if binary_data:
        # if we have 2 classes, we just want to show the total number
        # of pts in the group
        group_label_dist = group_dist * 100
    else:
        # otherwise, we want to show the label distribution for each group
        group_dist_repeat = np.repeat(group_dist, n_classes)
        group_label_dist = np.multiply(pattern_gt, group_dist_repeat) * 100

    for ind, pct in enumerate(group_label_dist):
        pct_int = int(pct)
        val = str(pct_int) if pct_int > 0 else ''
        xticklabels[ind] += '\n' + val
    ax.set_xticks(xticks)

    # shrink font size if the number of classes is "big"
    if n_classes > 3:
        ax.set_xticklabels(xticklabels, fontsize=8)
    else:
        ax.set_xticklabels(xticklabels)

    # use minor ticks to separate label ticks for each group
    ax.set_xticks(vert_lines_x, minor=True)
    ax.tick_params('x', length=30, which='minor')

    # set range of y axis
    if max_conf_ints is not None:
        ymin = np.min(max_conf_int_lb)
        ymax = np.max(max_conf_int_ub)
    else:
        ymin = np.min(conf_int_lb)
        ymax = np.max(conf_int_ub)

    max_height_scaled = 0.1 * (ymax - ymin)
    ymin = ymin - max_height_scaled
    ymax = ymax + max_height_scaled

    # for prediction probabilities, we want show relative probabilities if
    # we're using 1 bar per group (first statement), otherwise we want full
    # range.  If we have cross entropy confidences, then we don't care.
    if conf_objective == 'preds':
        if binary_data:
            ymin = max(0, ymin)
            ymax = min(1, ymax)
        else:
            ymin = 0
            ymax = 1

    ax.set_ylim(ymin, ymax)

    bottom_text = 'Points in Group' if binary_data else 'All Points'
    ax.set_xlabel('Group Labels\n Approx % of ' + bottom_text)
    obj_term = 'Probabilities' if conf_objective == 'preds'\
            else 'Cross Entropies'
    ax.set_ylabel('Average Group Label ' + obj_term)

    # ax.set_title(f'{set_name} {constr} {grouping} '
            # 'confidence intervals')

    ax.legend(bbox_to_anchor=(1, 1))

    # fig.savefig(file_name, bbox_inches='tight')
    fig.savefig(file_name, bbox_inches='tight', format='eps')
    plt.close(fig)

# pylint: disable=C0103
if __name__ == '__main__':
    # we only want to test certain combinations
    # semi-supervised, validation set, use bounds
    # unsupervised, training set, use bounds (akin to crowdsourcing)
    # oracle, training set, no bounds (equality constraints)

    # create results folder if it doesn't exist
    results_folder_path = './results'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # path for config jsons
    dataset_prefix = './datasets/'

    use_synthetic = False
    # use_synthetic = True
    # replot_figs = True
    replot_figs = False

    datasets = []
    if use_synthetic:
        # change dataset path if using synthetic datasets
        dataset_prefix = os.path.join(dataset_prefix, 'synthetic')
        # create dataset names for synthetic datasets
        syth_filename_part = 'synth_10p_1000n_100nval__'
        n_synth = 10
        for i in range(n_synth):
            datasets.append(syth_filename_part + str(i))
    else:
        # wrench datasets
        datasets += ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
               'imdb', 'obs', 'sms', 'yelp', 'youtube']
        # crowdsourcing datasets
        # datasets += ['bird', 'rte', 'dog', 'web']

    # constraint_types = ['accuracy', 'class_accuracy', 'confusion_matrix']
    constraint_types = ['accuracy']

    for dataset in datasets:
        # read the config file
        config_filename = os.path.join(dataset_prefix, dataset\
                + '_configs.json')
        with open(config_filename, 'r') as read_file:
            cfgs = json.load(read_file)

        # make result folder if it doesn't exist
        dataset_result_path = os.path.join(results_folder_path, dataset)
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        # make folder for WMRC specifically
        method_result_path = os.path.join(dataset_result_path, 'WMRC')
        if not os.path.exists(method_result_path):
            os.makedirs(method_result_path)

        for cfg in cfgs:
            n_class = cfg['n_classes']
            # get list of labeled datapoint counts to run and deleted it
            n_max_labeled_list = cfg['n_max_labeled']
            del cfg['n_max_labeled']

            for constraint_type in constraint_types:
                if cfg['add_mv_const']:
                    full_constraint_type = constraint_type + '+MV'
                else:
                    full_constraint_type = constraint_type

                cons_result_path = os.path.join(
                        method_result_path, full_constraint_type)

                if not os.path.exists(cons_result_path):
                    os.makedirs(cons_result_path)

                for n_labeled in n_max_labeled_list:
                    if not replot_figs:
                        # change loggers every time we change settings
                        # remove old handlers
                        for handler in logger.handlers[:]:
                            logger.removeHandler(handler)
                        formatter=logging.Formatter('%(asctime)s - %(message)s',
                                '%Y-%m-%d %H:%M:%S')
                        log_filename = get_result_filename(
                                dataset,
                                constraint_type,
                                cfg['labeled_set'],
                                cfg['bound_method'],
                                cfg['use_inequality_consts'],
                                cfg['add_mv_const'],
                                n_labeled=n_labeled)[:-4] + '.log'
                        log_filename_full = os.path.join(cons_result_path,
                                log_filename)
                        file_handler=logging.FileHandler(log_filename_full, 'w')
                        file_handler.setFormatter(formatter)
                        logger.addHandler(file_handler)
                        # log all the run parameters
                        logger.info('----------Running New Instance----------')
                        logger.info('dataset: %s, n_class: %d', dataset,n_class)
                        logger.info('constraint type: %s, add MV const: %s',
                                constraint_type, cfg['add_mv_const'])
                        logger.info('labeled set: %s, use inequalities: %s',
                                cfg['labeled_set'],
                                cfg['use_inequality_consts'])
                        logger.info('bound method: %s', cfg['bound_method'])
                        if cfg['labeled_set'] == 'valid':
                            logger.info('bound method: %s',
                                    cfg['bound_method'])

                    run_wmrc(
                            dataset_prefix,
                            constraint_form=constraint_type,
                            save_path=cons_result_path,
                            replot=replot_figs,
                            logger=logger,
                            n_max_labeled=n_labeled,
                            **cfg
                            )
