import logging
from typing import Any, Optional, Union

import numpy as np
from snorkel.labeling.model import LabelModel

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class Snorkel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 0.01,
                 l2: Optional[float] = 0.0,
                 n_epochs: Optional[int] = 100,
                 seed: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr'      : lr,
            'l2'      : l2,
            'n_epochs': n_epochs,
            'seed'    : seed or np.random.randint(1e6),
        }
        self.model = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False,
            weak: Optional[int] = None,
            n_weaks: Optional[int] = None,
            seed: Optional[int] = None,
            random_guess: Optional[int] = None,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        # L = check_weak_labels(dataset_train, n_weaks=n_weaks, random_guess=random_guess)
        # np.random.seed(seed)
        # np.random.shuffle(L.T)
        # L = L[:, 0:n_weaks]
        # n, m = L.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L = np.concatenate((L, r), axis = 1)
        L = dataset_train[0]
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        n_class = len(balance)
        self.n_class = n_class

        label_model = LabelModel(cardinality=n_class, verbose=verbose)
        label_model.fit(
            L_train=L,
            class_balance=balance,
            n_epochs=self.hyperparas['n_epochs'],
            lr=self.hyperparas['lr'],
            l2=self.hyperparas['l2'],
            seed=self.hyperparas['seed']
        )

        self.model = label_model

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], weak: Optional[int] = None, n_weaks: Optional[int] = None, random_guess: Optional[int] = None, seed: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        # L = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        L = dataset[0]
        # np.random.seed(seed)
        # np.random.shuffle(L.T)
        # L = L[:, 0:n_weaks]
        # n, m = L.shape
        # r = np.random.randint(0, 2, size=(n, random_guess))
        # L = np.concatenate((L, r), axis = 1)
        return self.model.predict_proba(L)
