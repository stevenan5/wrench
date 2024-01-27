import logging
import warnings
from typing import Any, Optional, Union

import numpy as np
#from numba import njit, prange
from tqdm.auto import trange

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


#@njit(parallel=True, nogil=True)
def help_e_step(Y_p, error_rates, L_aug):
    n, n_class = Y_p.shape
    #for i in prange(n):
    for i in range(n):
        for j in range(n_class):
            Y_p[i, j] *= np.prod(np.power(error_rates[:, j, :], L_aug[i, :, :]))

#@njit(parallel=True, nogil=True)
def initialize_Y_p(Y_p, L, n_class):
    n, m = L.shape
    #for i in prange(n):
    for i in range(n):
        counts = np.zeros(n_class)
        for j in range(m):
            if L[i, j] != ABSTAIN:
                counts[L[i, j]] += 1
        if counts.sum() == 0:
            counts += 1
        Y_p[i] = counts


class DawidSkene(BaseLabelModel):
    def __init__(self,
                 n_epochs: Optional[int] = 10000,
                 tolerance: Optional[float] = 1e-5,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'n_epochs' : n_epochs,
            'tolerance': tolerance,
        }

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            model_type: Optional[str] = 'general',
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        #if isinstance(dataset_train, BaseDataset):
        #    if n_class is not None:
        #        assert n_class == dataset_train.n_class
        #    else:
        #        n_class = dataset_train.n_class

        # L = check_weak_labels(dataset_train)
        # n_class = n_class or L.max() + 1
        L = dataset_train[0].astype(int)
        self.n_class = int(L.max() + 1)
        self.model_type = model_type

        Y_p = self._initialize_Y_p(L)
        L_aug = self._initialize_L_aug(L)


        max_iter = self.hyperparas['n_epochs']
        tol = self.hyperparas['tolerance']
        old_class_marginals = None
        old_error_rates = None
        for iter in trange(max_iter):

            # M-step
            (class_marginals, error_rates) = self._m_step(L_aug, Y_p)

            # E-step
            Y_p = self._e_step(L_aug, class_marginals, error_rates)

            # # check likelihood
            # log_L = self._calc_likelihood(L_aug, class_marginals, error_rates)

            # check for convergence
            if old_class_marginals is not None:
                class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
                error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                if (class_marginals_diff < tol and error_rates_diff < tol):
                    break

            # update current values
            old_class_marginals = class_marginals
            old_error_rates = error_rates

        self.error_rates = error_rates
        self.class_marginals = class_marginals

    def _initialize_Y_p(self, L):
        n_class = self.n_class
        n, m = L.shape

        Y_p = np.zeros((n, self.n_class))
        initialize_Y_p(Y_p, L, self.n_class)

        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p

    def _initialize_L_aug(self, L):
        # do not count abstentions as another class
        L_off = L
        L_aug = (np.arange(self.n_class) == L_off[..., None]).astype(int)
        return L_aug

    def _m_step(self, L_aug, Y_p):
        n, m, _ = L_aug.shape
        class_marginals = np.sum(Y_p, 0) / n

        ## compute error rates
        # for k in range(m):
        #     for j in range(n_class):
        #         for l in range(n_class+1):
        #             error_rates[k, j, l] = np.dot(Y_p[:, j], L_aug[:, k, l])

        error_rates = np.tensordot(Y_p, L_aug, axes=[[0], [0]]).transpose(1, 0, 2)

        # # normalize by summing over all observation classes
        s = np.sum(error_rates, axis=-1, keepdims=True)

        if self.model_type == 'one_coin':
            # want overall accuracy for the one-coin model
            # we will always assume that there is at least 1 prediction
            # on the training points
            for i in range(m):
                acc = error_rates[i].trace() / s[i].sum()
                error_rates[i] = (1 - acc) / (self.n_class - 1)
                np.fill_diagonal(error_rates[i], acc)
        else:
            error_rates = np.divide(error_rates, s, where=s != 0)

            if self.model_type == 'class_conditional':
                for i in range(m):
                    for j in range(self.n_class):
                        if s[i, j] == 0:
                            # if the whole row of the confusion matrix is 0
                            # set it to the uniform disribution, which will
                            # not change the answer
                            error_rates[i, j, :] = 1 / self.n_class
                        else:
                            class_acc = error_rates[i, j, j]
                            error_rates[i, j, :] = (1 - class_acc) / (self.n_class - 1)
                            error_rates[i, j, j] = class_acc

        return (class_marginals, error_rates)

    def _e_step(self, L_aug, class_marginals, error_rates):
        n, m, _ = L_aug.shape
        n_class = self.n_class

        Y_p = np.ones([n, n_class]) * class_marginals
        help_e_step(Y_p, error_rates, L_aug)

        # normalize error rates by dividing by the sum over all observation classes
        s = np.sum(Y_p, axis=-1, keepdims=True)
        Y_p = np.divide(Y_p, s, where=s != 0)
        return Y_p

    def _calc_likelihood(self, L_aug, class_marginals, error_rates):
        n, m, _ = L_aug.shape
        n_class = self.n_class
        log_L = 0.0

        for i in range(n):
            single_likelihood = 0.0
            for j in range(n_class):
                class_prior = class_marginals[j]
                Y_p_likelihood = np.prod(np.power(error_rates[:, j, :], L_aug[i, :, :]))
                Y_p_posterior = class_prior * Y_p_likelihood
                single_likelihood += Y_p_posterior

            temp = log_L + np.log(single_likelihood)

            if np.isnan(temp) or np.isinf(temp):
                warnings.warn('!')

            log_L = temp

        return log_L

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], oracle=False, **kwargs: Any) -> np.ndarray:
        #L = check_weak_labels(dataset)
        L = dataset[0].astype(int)
        self.n_class = int(L.max() + 1)
        L_aug = self._initialize_L_aug(L)

        if oracle:
            y = dataset[1]
            # initialize_Y_p does majority vote on the unlabeled data!
            Y_p = self._initialize_L_aug(dataset[1]).squeeze().astype(float)
            # Y_p = np.round(self._initialize_Y_p(L))
            (class_marginals, error_rates) = self._m_step(L_aug, Y_p)
            Y_p = self._e_step(L_aug, class_marginals, error_rates)

        else:
            Y_p = self._e_step(L_aug, self.class_marginals, self.error_rates)

        return np.clip(Y_p, 0, 1)
