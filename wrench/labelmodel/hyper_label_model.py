# https://arxiv.org/abs/2207.13545
import logging
from typing import Any, Optional, Union

import numpy as np
from hyperlm import HyperLabelModel

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1

class HyperLM(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.n_class = None
        self.hyperlm = HyperLabelModel()

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            n_class: Optional[int] = None,
            weak: Optional[int] = None,
            n_weaks: Optional[int] = None,
            seed: Optional[int] = None,
            random_guess: Optional[int] = None,
            **kwargs: Any):
        # if isinstance(dataset_train, BaseDataset):
        #     if n_class is not None:
        #         assert n_class == dataset_train.n_class
        #     else:
                # n_class = dataset_train.n_class
        # L_train = check_weak_labels(dataset_train, n_weaks=n_weaks, random_guess=random_guess)
        L_train = dataset_train[0]
        self.n_class = n_class or int(np.max(L_train)) + 1
        # np.random.seed(seed)
        # np.random.shuffle(L_train.T)
        # L_train = L_train[:, 0:n_weaks]
        # n, m = L_train.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L_train = np.concatenate((L_train, r), axis = 1)
        self.L_train = L_train

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], weak: Optional[int] = None, n_weaks: Optional[int] = None, random_guess: Optional[int] = None, seed: Optional[int] = None,
                      **kwargs: Any) -> np.ndarray:
        # L_test = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        L_test = dataset[0]
        # np.random.seed(seed)
        # np.random.shuffle(L_test.T)
        # L_test = L_test[:, 0:n_weaks]
        # n, m = L_test.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L_test = np.concatenate((L_test, r), axis = 1)
        if hasattr(self, "L_train"):

            L_all = np.concatenate([self.L_train, L_test])
            Y_p = self.hyperlm.infer(L_all,return_probs=True)
            n_train = self.L_train.shape[0]
            Y_p_test = Y_p[n_train:,:]
        else:
            Y_p_test = self.hyperlm.infer(L_test,return_probs=True)
        return Y_p_test

