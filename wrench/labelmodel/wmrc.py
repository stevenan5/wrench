# pylint: disable C0114
import logging
from typing import Any, Optional, Union

import cvxpy as cp
import numpy as np
import scipy as sp
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
# from ..dataset.utils import check_weak_labels
from .majority_voting import MajorityVoting

generic_logger = logging.getLogger(__name__)

ABSTAIN = -1
# pylint: disable=C0103
# pylint: disable=C0115
# pylint: disable=W0612
# pylint: disable=R0914
# pylint: disable=R0902
class WMRC(BaseLabelModel):
    def __init__(self,
                 solver: Optional[str] = 'MOSEK',
                 conf_solver: Optional[str] = 'GUROBI',
                 verbose: Optional[bool] = False,
                 **kwargs: Any):
        super().__init__()
        self.solver = solver
        self.conf_solver = conf_solver
        self.verbose = verbose

        ### create all the attributes and set to None to be 'pythonic'

        # logger
        self.logger = None
        # whether or not to add the majority vote classifier as a constraint
        self.majority_vote = None
        # on what quantities should be constrained along with class frequencies,
        # accuracy of each rule, accuracy of each rule depending on class,
        # or the confusion matrix of each rule
        self.constraint_type = None
        # whether or not to use equality constraints.  if equality constraints
        # are used, then the raw estimates of class frequencies, accuracies/
        # class accuracies/confusion matrices are used.  specifically, the
        # values in self.class_freq_probs[1, :] and self.param_probs[1, :]
        self.use_inequality_consts = None
        # what binomial confidence interval to use, e.g. wilson score, etc.
        self.ci_name = None
        # probability of failure for the above binomial confidence interval.
        # eg 0.05 for a 95% confidence interval
        self.binom_ci_fail_pr = None
        # number of classes
        self.n_class = None
        # number of rules
        self.p = None
        # number of training (unlabeled) datapoints
        self.n_pts = None
        # number of predictions each rule makes (on the training dataset)
        self.n_preds_per_rule = None
        # number of datapoints to be used to estimate classifier parameters
        self.n_max_labeled = None
        # number of rules that have at least 1 labeled point
        self.n_rules_used = None
        # how many labeled points each rule got
        self.avg_labeled_per_rule = None
        # 3 row array for class frequency bounds [[lower], [raw est], [upper]]
        self.class_freq_probs = None
        # 3 element array of estimates related to rules, eg accuracies.
        # first index is lower bounds, second index the raw estimate, third
        # index are upper bounds.
        self.param_probs = None
        # scaled versions of the above where one multiplies by the number of
        # predictions made on the training set
        self.class_freq_cts = None
        self.param_cts = None
        # the max width between the lower (resp. upper) bounds of the class
        # frequency or parameter probabilities and the raw estimate. used to
        # create the convex program
        self.class_freq_eps = None
        self.param_eps = None
        # the convex optimzation problem for WMRC
        self.prob = None
        # the status of the above problem
        self.prob_status = None
        # the optimization variables for the above convex problem
        self.param_wts = None
        self.class_freq_wts = None
        # a pair of convex optimization problems for find confidence intervals
        self.lb_prob = None
        self.ub_prob = None
        # the cvxpy parameter vector to select datapoints and classes
        self.sel = None
        # a 2D array where each column is a unique 'pattern', or instantiation
        # of each rule's prediction on a datapoint
        self.col_unique = None
        # indices of the first instance where each unique pattern appears
        self.col_uniq_inds = None
        # an array recording which pattern is at which position
        self.col_inv = None
        # the total number of patterns
        self.n_patterns = None
        # groupins of points based on patterns
        self.pattern_selections = None
        # groupings of points based on balls around patterns
        self.pattern_neigh_selections = None
        # groupings of points based on predicted probability (predicted
        # probabilities are uppper and lower bounded here)
        self.pred_prob_selections = None
        # same as above, but we only group points which predict probability
        # above some lower bound
        self.pred_prob_selections_lb_only = None
        # an array recording which thresholds were used when grouping datapoints
        # based on the predicted probabilities (for each class)
        self.used_thresholds = None
        # same as above, but when only the lower bounds are used
        self.used_thresholds_lb_only = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Union[BaseDataset, np.ndarray],
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            constraint_type = 'accuracy',
            bound_method: Optional[str] = 'binomial',
            majority_vote: Optional[bool] = False,
            use_inequality_consts: Optional[bool] = True,
            ci_name: Optional[str] = 'wilson',
            binom_ci_fail_pr: Optional[float] = 0.05,
            n_max_labeled: Optional[int] = -1,
            bound_scale: Optional[float] = 1,
            verbose: Optional[bool] = False,
            logger = generic_logger,
            **kwargs: Any):
            # balance: Optional[np.ndarray] = None,
            # weak: Optional[int] = None,
            # n_weaks: Optional[int] = None,
            # seed: Optional[int] = None,
            # random_guess: Optional[int] = None,

        self.majority_vote = majority_vote
        self.constraint_type = constraint_type
        self.binom_ci_fail_pr = binom_ci_fail_pr
        self.use_inequality_consts = use_inequality_consts
        self.ci_name = ci_name
        self.n_max_labeled = n_max_labeled
        self.logger=logger

        if constraint_type not in ['accuracy', 'class_accuracy',\
                'confusion_matrix']:
            raise ValueError(f"constraint_type argument ({constraint_type})"
                    f" must be in ['accuracy', 'class_accuracy',"
                    f" 'confusion_matrix']")

        # self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class

        # get ground truths for each of the datasets
        if y_valid is None:
            y_valid = np.squeeze(dataset_valid[1])
            # y_valid = np.array(dataset_valid.labels)

        # y_train = np.squeeze(dataset_train[1])

        # add majority vote classifier to ensemble if asked for

        if self.majority_vote:
            dataset_train_mod = self._add_majority_vote_const(dataset_train)
            dataset_valid_mod = self._add_majority_vote_const(dataset_valid)
            L = dataset_train_mod[0]
            L_val = dataset_valid_mod[0]
        else:
            dataset_train_mod = dataset_train
            dataset_valid_mod = dataset_valid
            L = dataset_train[0]
            L_val = dataset_valid[0]
            # L = check_weak_labels(dataset_train, n_weaks=n_weaks,\
            #            random_guess=random_guess)
            # L_val = check_weak_labels(dataset_valid, n_weaks=n_weaks,\
            #            random_guess=random_guess)

        # number of classes
        n_class = n_class or round(L.max()) + 1
        self.n_class = n_class

        # get one hot encodings of the predictions
        L_aug = self._initialize_L_aug(L)

        # classifier count, number of datapoints
        self.p, self.n_pts, _ = L_aug.shape

        # count number of predictions for every classifier
        n_preds_per_rule = np.sum(L_aug, axis=(-2, -1))
        self.n_preds_per_rule = n_preds_per_rule

        # estimate class frequencies and classifier parameters (e.g. accuracies)
        param_probs, class_freq_probs = self._get_prob_bounds(dataset_valid_mod,
                method=bound_method, bound_scale=bound_scale)

        self.class_freq_probs = class_freq_probs
        self.param_probs = param_probs

        # scale the param_probs and class_freq_probs for convex program
        param_cts, class_freq_cts = self._scale_probs_to_cts(param_probs,
                class_freq_probs, self.n_pts, n_preds_per_rule)

        self.class_freq_cts = class_freq_cts
        self.param_cts = param_cts

        # make convex program
        self.prob, sigma, gamma = self._make_dual_cp(L_aug,
                self.param_cts, self.class_freq_cts,
                use_error_bars=use_inequality_consts)

        # solve convex program
        self.prob.solve(solver=self.solver, verbose=self.verbose)
        self.prob_status = self.prob.status

        if self.constraint_type == 'accuracy':
            self.param_wts = sigma.value
        else:
            self.param_wts = []
            for j in range(self.p):
                self.param_wts.append(sigma[j].value)

        self.class_freq_wts = gamma.value

    def _add_majority_vote_const(self, dataset):
        label_model = MajorityVoting()
        label_model.fit(dataset)# , weak=weak, n_weaks=n_weaks,\
                #random_guess=random_guess, seed=seed)
        y_pred = label_model.predict_proba(dataset)# , weak=weak,\
                #n_weaks=n_weaks, random_guess=random_guess, seed=seed)
        pred_majority_vote = np.argmax(y_pred, axis=1)
        pred_majority_vote = np.expand_dims(pred_majority_vote, 1)
        L = dataset[0]
        L = np.hstack((L, pred_majority_vote))

        return [L, dataset[1]]

    def _get_prob_bounds(self, dataset, method='binomial', bound_scale=1):
                    # est_class_freqs=True):
        # get the epsilons, or the intervals which the estimate accuracy of
        # each classifier/labeling frequency falls in

        if method not in ["binomial", "unsupervised"]:
            raise ValueError("chosen method must be in"
                              " ['binomial', 'unsupervised'].")

        L = dataset[0]

        # get MV prediction as long as we're not using binomial interval
        if method != 'binomial':
            label_model = MajorityVoting()
            label_model.fit(dataset, n_class=self.n_class)
            mv_pred = label_model.predict_proba(dataset)
            mv_pred_hard = np.argmax(mv_pred, axis=1)
            mv_pred_hard_aug = self._initialize_one_hot_labels(mv_pred_hard)

        y = np.squeeze(dataset[1])

        n_all_labeled = len(y)

        # sample the datapoints if need be
        if self.n_max_labeled > 0 and len(y) > self.n_max_labeled:
            choices = np.random.choice(np.arange(len(y)), self.n_max_labeled,
                    replace=False)
            y = y[choices]
            L = L[choices, :]
            dataset = [L, y]

        L_aug = self._initialize_L_aug(L)
        y_aug = self._initialize_one_hot_labels(y)
        p, n_labeled, _ = L_aug.shape

        # compute class distribution
        class_freq_probs = np.zeros((3, self.n_class))
        # set upper bound to 1
        class_freq_probs[2, :] = 1
        if method != 'unsupervised':
            class_freq_probs[1, :] = np.bincount(y) / n_labeled
        else:
            class_freq_probs[1, :] = np.bincount(mv_pred_hard) / n_labeled

        # count the number of predictions made per rule
        n_rule_preds = np.sum(L_aug, axis=(-2, -1))

        # compute and report the number of classifiers with labeled data,
        # and average number of labeled data per classifier (ignoring
        # classifiers that got no labeled data)
        labeled_rules = n_rule_preds > 0
        n_usable_rules = int(np.sum(labeled_rules))
        self.n_rules_used = n_usable_rules
        self.avg_labeled_per_rule = np.mean(n_rule_preds[labeled_rules])
        self.logger.info('----Estimating Probabilities----')
        self.logger.info('%d out of %d labeled datapoints on average per rule ',
                self.avg_labeled_per_rule, n_all_labeled)
        self.logger.info('%d rules with labeled data out of %d total',\
                n_usable_rules, p)
        self.logger.info('Number of labeled predictions per rule:'
                f' {n_rule_preds[labeled_rules]}')
        if self.n_max_labeled > 0:
            self.logger.info('Using at maximum %d many labeled points',
                    self.n_max_labeled)
        if bound_scale != 1:
            self.logger.info('Bound probabilities scaled by: %f', bound_scale)

        n_classes_with_labels = int(np.sum(class_freq_probs[1, :] > 0))
        if n_classes_with_labels < self.n_class:
            self.logger.info('%d classes without labels out of %d total',\
                    n_classes_with_labels, self.n_class)

        if self.constraint_type == 'accuracy':
            param_probs = np.zeros((3, p))
        elif self.constraint_type == 'class_accuracy':
            param_probs = np.zeros((3, p, self.n_class))
        else:
            param_probs = np.zeros((3, p, self.n_class, self.n_class))
        # set upper bound to 1
        param_probs[2, ...] = 1

        # figure out the positions of each class
        class_pos = np.full((n_labeled, self.n_class), False, dtype=bool)
        for l in range(self.n_class):
            class_pos[:, l] = y == l

        # get bounds for class frequencies
        for j in range(self.n_class):
            if class_freq_probs[1, j] == 0:
                continue

            if not self.use_inequality_consts:
                class_freq_probs[[0, 2], j] = class_freq_probs[1, j]
            elif method == 'binomial':
                class_freq_probs[[0, 2], j] = proportion_confint(
                        n_labeled * class_freq_probs[1, j],
                        n_labeled,
                        alpha=self.binom_ci_fail_pr, method=self.ci_name)
            elif method == 'unsupervised':
                # do the same thing as the binomial case, just use the majority
                # vote as the label
                class_freq_probs[[0, 2], j] = proportion_confint(
                        n_labeled * class_freq_probs[1, j],
                        n_labeled,
                        alpha=self.binom_ci_fail_pr, method=self.ci_name)

        # scale the bounds
        class_freq_probs[0, :] /= bound_scale
        class_freq_probs[2, :] *= bound_scale
        class_freq_probs[[0, 2], :] = np.clip(class_freq_probs[[0, 2], :], 0, 1)

        for j in range(p):
            if method != 'unsupervised' and n_rule_preds[j] == 0:
                continue

            # deal with abstentions by only looking at vector of labels
            # where a prediction was made on that point
            pred_on = L[:, j] != ABSTAIN

            conf_mat_y = confusion_matrix(y[pred_on], L[pred_on, j],
                    labels=np.arange(self.n_class), normalize='all')
            if method == 'unsupervised':
                # modify the prediction matrix so the uniform distribution
                # is predicted when the rule abstains
                L_aug[j, np.logical_not(pred_on), :] = 1 / self.n_class
                conf_mat_unsup = mv_pred_hard_aug.T @ L_aug / n_all_labeled

            # compute the number of predictions the rule made given that the
            # true class was ell
            preds_per_label = n_rule_preds[j]\
                    * np.sum(conf_mat_y, axis=1)

            if self.constraint_type == 'accuracy':
                if not self.use_inequality_consts:
                    param_probs[1, j] = np.trace(conf_mat_y)
                    param_probs[[0, 2], j] = param_probs[1, j]
                elif method in 'binomial':
                    param_probs[1, j] = np.trace(conf_mat_y)
                    param_probs[[0, 2], j] = proportion_confint(
                            n_rule_preds[j] * param_probs[1, j],n_rule_preds[j],
                            alpha=self.binom_ci_fail_pr, method=self.ci_name)

                elif method == 'unsupervised':
                    # do the same thing as the binomial method, but we've
                    # articially added predictions for all the points that
                    # abstain.  Ie we brought the number of predictions up to
                    # n_all_labeled
                    param_probs[1, j] = np.trace(conf_mat_y)
                    param_probs[[0, 2], j] = proportion_confint(
                            n_rule_preds[j] * param_probs[1, j],n_rule_preds[j],
                            alpha=self.binom_ci_fail_pr, method=self.ci_name)
                    # param_probs[[0, 2], j] = proportion_confint(
                    #         n_all_labeled * param_probs[1, j], n_all_labeled,
                    #         alpha=self.binom_ci_fail_pr, method=self.ci_name)

            elif self.constraint_type == 'class_accuracy':
                if not self.use_inequality_consts:
                    param_probs[1, j, :] = np.diag(conf_mat_y)
                    param_probs[[0, 2], j, :] = param_probs[1, j, :]
                elif method == 'binomial':
                    param_probs[1, j, :] = np.diag(conf_mat_y)
                    # don't do the confidence interval on classes where
                    # there were no predictions
                    sel = preds_per_label > 0
                    lb, ub = proportion_confint(
                            preds_per_label[sel] * param_probs[1, j, sel],
                            preds_per_label[sel], alpha=self.binom_ci_fail_pr,
                            method=self.ci_name)
                    param_probs[0, j, sel] = lb
                    param_probs[2, j, sel] = ub

                elif method == 'unsupervised':
                    raise ValueError("Unsupervised estimation not implemented"
                            " for class conditional accuracy constraints")
            else:
                if not self.use_inequality_consts:
                    param_probs[1, j, :, :] = conf_mat_y
                    param_probs[[0, 2], j, :, :] = param_probs[1, j, :, :]
                elif method == 'binomial':
                    param_probs[1, j, :, :] = conf_mat_y
                    for l in range(self.n_class):
                        # don't do the confidence interval on classes where
                        # there were no predictions
                        if preds_per_label[l] == 0:
                            continue
                        # even if there were predictions on a class,
                        # some predictions might not have obtained
                        sel = param_probs[1, j, l, :] > 0
                        lb, ub = proportion_confint(
                                preds_per_label[l] * param_probs[1, j, l, sel],
                                preds_per_label[l], alpha=self.binom_ci_fail_pr,
                                method=self.ci_name)
                        param_probs[0, j, l, sel] = lb
                        param_probs[2, j, l, sel] = ub
                elif method == 'unsupervised':
                    raise ValueError("Unsupervised estimation not implemented"
                            " for confusion matrix constraints")

        # scale the bounds
        param_probs[0, ...] /= bound_scale
        param_probs[2, ...] *= bound_scale
        param_probs[[0, 2], ...] = np.clip(param_probs[[0, 2], ...], 0, 1)

        return param_probs, class_freq_probs

    def _scale_probs_to_cts(self, param_probs, class_freq_probs, n_datapoints,
            n_preds_per_rule):

        class_freq_cts = np.zeros(class_freq_probs.shape)
        class_freq_cts = n_datapoints * class_freq_probs

        param_cts = np.zeros(param_probs.shape)
        for j in range(self.p):
            if self.constraint_type == 'accuracy':
                param_cts[:, j] = n_preds_per_rule[j] * param_probs[:, j]
            elif self.constraint_type == 'class_accuracy':
                param_cts[:, j, :] = n_preds_per_rule[j]\
                        * param_probs[:, j, :]
            else:
                param_cts[:, j, :, :] = n_preds_per_rule[j]\
                        * param_probs[:, j, :, :]

        return param_cts, class_freq_cts

    def _make_dual_cp(self, L_aug, param_cts, class_freq_cts,
            use_error_bars=True):
        # make the dual convex program, which is solved in fit
        # create variables
        if self.constraint_type == 'accuracy':
            sigma = cp.Variable(self.p)
        elif self.constraint_type == 'class_accuracy':
            sigma = [cp.Variable(self.n_class) for j in range(self.p)]
        else:
            sigma = [cp.Variable((self.n_class, self.n_class))
                    for j in range(self.p)]

        gamma = cp.Variable(self.n_class)

        # create constants for the problem
        class_freq_eps = np.maximum(
                class_freq_cts[1, :] - class_freq_cts[0, :],
                class_freq_cts[2, :] - class_freq_cts[1, :])

        if self.constraint_type == 'accuracy':
            param_eps = np.zeros(self.p)
            param_eps = np.maximum(param_cts[1, :] - param_cts[0, :],
                    param_cts[2, :] - param_cts[1, :])
        elif self.constraint_type == 'class_accuracy':
            param_eps = np.zeros((self.p, self.n_class))
            for j in range(self.p):
                param_eps[j, :] = np.maximum(
                        param_cts[1, j, :] - param_cts[0, j, :],
                        param_cts[2, j, :] - param_cts[1, j, :])
        else:
            param_eps = np.zeros((self.p, self.n_class, self.n_class))
            for j in range(self.p):
                param_eps[j, :, :] = np.maximum(
                        param_cts[1, j, :, :] - param_cts[0, j, :, :],
                        param_cts[2, j, :, :] - param_cts[1, j, :, :])

        self.class_freq_eps = class_freq_eps
        self.param_eps = param_eps

        # create objective
        sigma_term = 0
        sigma_abs_term = 0
        gamma_term = gamma @ class_freq_cts[1, :]
        gamma_abs_term = cp.abs(gamma) @ class_freq_eps if use_error_bars else 0

        aggregated_weights = self._aggregate_weights(L_aug, sigma, gamma)
        if self.constraint_type == 'accuracy':
            sigma_term = sigma @ param_cts[1, :]
            if use_error_bars:
                sigma_abs_term = cp.abs(sigma) @ param_eps
        elif self.constraint_type == 'class_accuracy':
            for j in range(self.p):
                sigma_term += sigma[j] @ param_cts[1, j, :]
                if use_error_bars:
                    sigma_abs_term += cp.abs(sigma[j]) @ param_eps[j, :]
        else:
            for j in range(self.p):
                sigma_term += cp.sum(cp.multiply(sigma[j],\
                        param_cts[1, j, :, :]))
                if use_error_bars:
                    sigma_abs_term += cp.sum(
                            cp.multiply(cp.abs(sigma[j]), param_eps[j, :, :]))

        obj = cp.Maximize(sigma_term
                + gamma_term
                - sigma_abs_term
                - gamma_abs_term
                - cp.sum(cp.log_sum_exp(aggregated_weights, axis = 1)))

        # create constraints
        constrs = []

        return cp.Problem(obj, constrs), sigma, gamma

    def _initialize_L_aug(self, L):
        # convert L into a stack of matrices, one-hot encodings of each
        # classifier's predictions. index 0 is which classifier, index 1 is
        # the datapoint index, index 2 is the class index
        L = L.T
        L_aug = (np.arange(self.n_class) == L[..., None]).astype(int)
        return L_aug

    def _initialize_one_hot_labels(self, y):
        # used to convert ground truth labels into one hot encoded labels.
        # rows are datapoints, columns are classes
        return np.squeeze(self._initialize_L_aug(y))

    def _aggregate_weights(self, L_aug, param_wts, class_freq_wts, mod=cp):
        # essentially create the weighted majority vote with provided weights

        # assuming param_wts is a k by k matrix where element ij is the weight
        # associated with the classifier predicting j when true label is i.
        p, n, _ = L_aug.shape
        n_class = self.n_class

        y_pred = mod.multiply(np.ones([n, n_class]), class_freq_wts[None])

        for j in range(p):
            # pick out column of confusion matrix (since we see observed pred)
            # for every datapoint.  Resulting matrix is n by k
            if self.constraint_type == 'accuracy':
                y_pred += mod.multiply(L_aug[j], param_wts[j])

            elif self.constraint_type == 'class_accuracy':
                y_pred += L_aug[j] @ mod.diag(param_wts[j])
            else:
            # for confusion matrices where param_wts is shape (p, k, k)
                y_pred += L_aug[j] @ param_wts[j].T
        return y_pred

    def _make_wmrc_preds(self, L_aug, param_wts, class_freq_wts):
        y_pred = self._aggregate_weights(L_aug, param_wts, class_freq_wts, mod=np)
        return sp.special.softmax(y_pred, axis=1)

    def predict_proba(self,
            dataset: Union[BaseDataset, np.ndarray],
            **kwargs: Any) -> np.ndarray:
            # weak: Optional[int] = None,
            # n_weaks: Optional[int] = None,
            # random_guess: Optional[int] = None,
            # seed: Optional[int] = None,
        # L = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        if self.majority_vote:
            dataset_mod = self._add_majority_vote_const(dataset)
        else:
            dataset_mod = dataset

        L = dataset_mod[0]

        L_aug = self._initialize_L_aug(L)
        y_pred = self._make_wmrc_preds(L_aug, self.param_wts, self.class_freq_wts)
        return y_pred

    def _get_primal_constraints(self, L, z):
        # for confidences, we are solving the primal problem

        L_aug = self._initialize_L_aug(L)
        if self.majority_vote:
            dataset_tmp = self._add_majority_vote_const([L, None])
            L_aug = self._initialize_L_aug(dataset_tmp[0])
        constrs = []
        constrs = [z >= 0, cp.sum(z, axis=1) == 1]
        if self.use_inequality_consts:
            constrs += [cp.sum(z, axis=0) >= self.class_freq_cts[0, :],
                    cp.sum(z, axis=0) <= self.class_freq_cts[2, :]]
        else:
            constrs += [cp.sum(z, axis=0) == self.class_freq_cts[1, :]]

        for j in range(self.p):
            conf_mat = z.T @ L_aug[j]
            if self.constraint_type == 'accuracy':
                conf_mat_tr = cp.trace(conf_mat)
                if self.use_inequality_consts:
                    constrs += [conf_mat_tr >= self.param_cts[0, j],
                            conf_mat_tr <= self.param_cts[2, j]]
                else:
                    constrs += [conf_mat_tr == self.param_cts[1, j]]
            elif self.constraint_type == 'class_accuracy':
                conf_mat_diag = cp.diag(conf_mat)
                if self.use_inequality_consts:
                    constrs += [conf_mat_diag >= self.param_cts[0, j, :],
                            conf_mat_diag <= self.param_cts[2, j, :]]
                else:
                    constrs += [conf_mat_diag == self.param_cts[1, j, :]]

            else:
                conf_mat_vec = cp.reshape(conf_mat,
                        shape=(self.n_class ** 2, 1), order='C')
                param_cts_lb_vec = np.reshape(self.param_cts[0, j, :, :],
                                                                        (-1, 1))
                param_cts_est_vec = np.reshape(self.param_cts[1, j, :, :],
                                                                        (-1, 1))
                param_cts_ub_vec = np.reshape(self.param_cts[2, j, :, :],
                                                                        (-1, 1))
                if self.use_inequality_consts:
                    constrs += [conf_mat_vec >= param_cts_lb_vec,
                            conf_mat_vec <= param_cts_ub_vec]
                else:
                    constrs += [conf_mat_vec == param_cts_est_vec]

        return constrs

    def _make_confidence_progs(self, z, constrs, n_points, wmrc_preds=None):
        # make the linear programs used to compute the confidences
        # the constraints stay the same, and only the objective changes

        self.sel = cp.Parameter(n_points * self.n_class)

        # make the objective cross entropy -z^T log(wmrc_pred) if a prediction
        # is given
        if wmrc_preds is not None:
            obj_fn = -1 * cp.multiply(z, cp.log(wmrc_preds))
        # otherwise make it z
        else:
            obj_fn = z

        reshaped_z = cp.reshape(obj_fn, (n_points * self.n_class, 1), order='C')
        lb_obj = cp.Minimize(self.sel.T @ reshaped_z)
        ub_obj = cp.Maximize(self.sel.T @ reshaped_z)

        self.lb_prob = cp.Problem(lb_obj, constrs)
        self.ub_prob = cp.Problem(ub_obj, constrs)

    def _pattern_selections(self, L):
        # partition the predictions L via patterns. a pattern is essentially
        # the string of predictions of all classifiers on a datapoint.

        # want each prediction to be in a row
        n_pts = L.shape[0]
        L = L.T

        self.col_unique, self.col_uniq_inds, self.col_inv = \
                np.unique(L, axis=1, return_inverse=True, return_index=True)
        self.n_patterns = self.col_unique.shape[1]
        selections = np.zeros((self.col_unique.shape[1], n_pts))
        for i in range(selections.shape[0]):
            selections[i, :] = self.col_inv == i

        # cast to int so they can be used as indices
        selections = selections.astype(int)

        self.pattern_selections = selections

        return selections

    def _pattern_neighborhood_selections(self, L, neighbor_dist):
        # greedily construct groups by creating balls via hamming distance
        # get patterns, and we'll only look at patterns
        n_pts = L.shape[0]
        patt_sels = self._pattern_selections(L)
        self.n_patterns = self.col_unique.shape[1]
        remaining_patterns = np.arange(self.n_patterns)
        selections = np.zeros(patt_sels.shape)

        count = 0
        while len(remaining_patterns) > 0:
            inds_to_del = []
            curr_sels = np.zeros(n_pts)
            ref_patt_ind = remaining_patterns[0]
            curr_sels += self.col_inv == ref_patt_ind
            inds_to_del.append(0)

            for i, pat_ind in enumerate(remaining_patterns[1:]):
                if np.sum(self.col_unique[:, ref_patt_ind]\
                        != self.col_unique[:, pat_ind]) < neighbor_dist:
                    curr_sels += patt_sels[pat_ind, :]
                    # curr_sels += self.col_inv == pat_ind
                    # add 1 because the new index 0 is the old index 1
                    inds_to_del.append(i + 1)

            remaining_patterns = np.delete(remaining_patterns, inds_to_del)
            selections[count, :] = curr_sels
            count += 1

        # cast to int so they can be used as indices
        selections = selections.astype(int)
        selections = selections[:count, :]

        self.pattern_neigh_selections = selections

        return selections

    def _predicted_prob_selections(self, preds, L, prediction_thresholds,
            use_ub=True, combine_classes=False):
        # group the datapoints based on the predicted probabilities of WMRC
        # have the option of using both upper and lower bounds on predicted
        # probabilities, and whether or not one wants to group datapoints by
        # predicted probability of a certain class falling inside the interval
        # of probabilities.  e.g. if combine_classes=True, then points will be
        # grouped if at least one class has probability >= 0.9 (or another prob)

        self._pattern_selections(L)
        n_pts = preds.shape[0]
        # convert to decimal percents
        prediction_thresholds /= 100
        used_thresholds = []
        uniq_preds = preds[self.col_uniq_inds, :]
        n_patterns = self.col_unique.shape[1]
        already_selected = np.zeros(n_patterns)
        selections = np.zeros((n_patterns * self.n_class, n_pts))

        count = 0
        break_cond = False
        for l in range(self.n_class):
            if break_cond:
                break
            for i, thres in enumerate(prediction_thresholds):
                used_threshold_i = False
                if int(np.sum(already_selected)) == n_patterns:
                    break_cond = True
                    break

                # select the patterns where in class l, the prediction is above
                # thres
                # if we're combining classes, we only care about the maximum
                # predicted probability
                if combine_classes:
                    patt_inds_mask = np.max(uniq_preds, axis=1) >= thres
                else:
                    patt_inds_mask = uniq_preds[:, l] >= thres

                patt_inds_mask = patt_inds_mask.astype(int)
                # figure out the locations of the newly selected patterns
                new_patt_inds_mask = np.multiply(1 - already_selected,\
                        patt_inds_mask)
                # if we take at least 1 more pattern
                if np.sum(new_patt_inds_mask) > 0:
                    used_threshold_i = True
                    # record the new patterns we are selecting
                    already_selected += new_patt_inds_mask

                    # if we use the upper bound for predicted probabilities
                    # then we don't want the older selections
                    if use_ub:
                        used_mask = new_patt_inds_mask
                    else:
                        used_mask = patt_inds_mask

                    # figure out what indices of patterns we should take
                    used_mask = used_mask.astype(int).astype(bool)
                    patt_inds = np.arange(n_patterns)[used_mask]
                    for patt_ind in patt_inds:
                        selections[count, :] += self.col_inv == patt_ind
                    count += 1
                if used_threshold_i:
                    used_thresholds.append(i)

            if combine_classes:
                break_cond = True
                break

        used_thresholds = np.unique(used_thresholds)

        # allow ourselves to use different thesholds depending on if we use
        # the upper bound in making our groups
        final_used_thresholds = prediction_thresholds[used_thresholds]
        if combine_classes:
            if use_ub:
                self.used_thresholds_comb_class = final_used_thresholds
            else:
                self.used_thresholds_lb_only_comb_class = final_used_thresholds
        else:
            if use_ub:
                self.used_thresholds = final_used_thresholds
            else:
                self.used_thresholds_lb_only=final_used_thresholds

        # cast to int so they can be used as indices
        selections = selections.astype(int)
        selections = selections[:count, :]
        if not combine_classes:
            # permute the selections so the highest mean prediction (regardless
            # of class) will be first
            mean_preds = self._mean_group_preds(selections, preds)
            max_probs = np.amax(mean_preds, axis=1)
            max_probs_argsort = np.argsort(- max_probs)

            selections = selections[max_probs_argsort, :]

        if combine_classes:
            if use_ub:
                self.pred_prob_selections_comb_class = selections
            else:
                self.pred_prob_selections_lb_only_comb_class = selections
        else:
            if use_ub:
                self.pred_prob_selections = selections
            else:
                self.pred_prob_selections_lb_only = selections

        return selections

    def get_confidences(self, data, grouping="pattern_neigh",
            neighborhood_size = 5, prediction_thresholds=[], wmrc_preds=None):

        if grouping not in ['pattern', 'pattern_neigh', 'pred_prob',\
                'pred_prob_lb', 'pred_prob_comb_class',\
                'pred_prob_lb_comb_class']:
            raise ValueError(f"grouping argument must be in"
                    f"['pattern', 'pattern_neighbor', 'pred_prob', "
                    f"'pred_prob_lb', 'pred_prob_comb_class', "
                    f"'pred_prob_lb_comb_class'], but was {grouping}.")

        if grouping in ['pred_prob', 'pred_prob_lb',\
                'pred_prob_comb_class', 'pred_prob_lb_comb_class']:
            if len(prediction_thresholds) == 0:
                raise ValueError("prediction_thresholds must not be empty.")
            for i, val in enumerate(prediction_thresholds):
                if val < 100/self.n_class or val > 100:
                    raise ValueError(f"Value of prediction_thresholds at "
                            f"index {i} has value {val}, which is outside of "
                            f"{100/self.n_class} and 100 inclusive.")

        use_xent = wmrc_preds is not None

        L = data[0]
        y_data = data[1]
        y_data_aug = self._initialize_one_hot_labels(y_data)
        y_pred = self.predict_proba(data)
        L_aug = self._initialize_L_aug(L)
        p, n_pts, _ = L_aug.shape

        combined_class = grouping[-10:] == 'comb_class'
        if grouping == 'pattern':
            selections = self._pattern_selections(L)
        elif grouping == 'pattern_neigh':
            selections = self._pattern_neighborhood_selections(L,
                    neighborhood_size)
        elif grouping == 'pred_prob':
            selections = self._predicted_prob_selections(y_pred, L,
                    prediction_thresholds)
        elif grouping == 'pred_prob_lb':
            selections = self._predicted_prob_selections(y_pred, L,
                    prediction_thresholds, use_ub=False)
        elif grouping == 'pred_prob_comb_class':
            selections = self._predicted_prob_selections(y_pred, L,
                    prediction_thresholds, combine_classes=True)
        elif grouping == 'pred_prob_lb_comb_class':
            selections = self._predicted_prob_selections(y_pred, L,
                    prediction_thresholds, use_ub=False, combine_classes=True)

        #assume selections is number of selections by n * self.n_class
        m = selections.shape[0]

        if combined_class:
            ci = np.zeros((2, m))
        else:
            ci = np.zeros((2, m * self.n_class))

        # record the mean prediction by WMRC and from the GT for the groups
        if use_xent:
            # don't have a mean prediction since you need a ground truth and
            # prediction when dealing with cross entropies
            mean_pred = None
            # otherwise compute elementwise log loss then aggregate via
            # selections
            element_xent = -1 * np.multiply(y_data_aug, np.log(y_pred))
            gt_mean_pred = self._mean_group_preds(selections, element_xent)
        else:
            mean_pred = self._mean_group_preds(selections, y_pred)
            gt_mean_pred = self._mean_group_preds(selections, y_data_aug)

        z = cp.Variable((n_pts, self.n_class))

        # form problems
        constrs = self._get_primal_constraints(L, z)
        self._make_confidence_progs(z, constrs, n_pts, wmrc_preds=wmrc_preds)

        for i, prob in enumerate([self.lb_prob, self.ub_prob]):
            for t in range(m):
                for j in range(self.n_class):
                    if combined_class:
                        exp_sels = self._expand_sels(selections[t, :],
                                n_pts, self.n_class)
                    else:
                        # expand the selections since we need to solve for every
                        # class too
                        exp_sels = self._expand_sels(selections[t, :],
                                n_pts, self.n_class, specific_class=j)
                    self.sel.value = exp_sels

                    prob.solve(solver=self.conf_solver, verbose = self.verbose)

                    if combined_class:
                        # scale down to probabilities, based on the number of points
                        ci[i, t] = prob.value / np.sum(selections[t, :])
                        break
                    else:
                        # scale down to probabilities, based on the number of points
                        # in this case, denominator is equal to sum of selections
                        ci[i, t * self.n_class + j] = prob.value / np.sum(exp_sels)

        return ci, mean_pred, gt_mean_pred

    def _mean_group_preds(self, selections, preds):
        # average the predictions on datapoints grouped together by selections
        m = selections.shape[0]
        # cast to bool so we can mask
        selections = selections.astype(bool)
        mean_preds = np.zeros((m, self.n_class))

        for i in range(m):
            mean_preds[i, :] = np.mean(preds[selections[i, :], :], axis=0)

        return mean_preds

    def _expand_sels(self, selection, n_points, n_class, specific_class=None):
        # expands the selection of datapoints so either all classes
        # or a specific class is picked out
        if (specific_class is not None) and (specific_class < 0\
                or specific_class >= n_class):
            raise ValueError(f"specific_class must be between 0 and {n_class-1}"
                             f" when not None, but was {specific_class}")

        if specific_class is None:
            exp_sels = np.repeat(selection, n_class)
        else:
            exp_sels = np.zeros(n_points * n_class)
            for i, ele in enumerate(selection):
                exp_sels[n_class * i + specific_class ] = ele

        return exp_sels
