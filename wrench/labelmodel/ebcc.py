from typing import Optional, Any, Union
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
from scipy.stats import entropy, dirichlet
from tqdm import trange

from wrench.basemodel import BaseLabelModel
from wrench.dataset import BaseDataset
from ..utils import create_tuples


def ebcc_vb(L,
            num_items,
            num_workers,
            num_classes,
            a_pi=0.1,
            num_groups=10,  # M
            alpha=1,  # alpha_k, it can be 1 or \sum_i gamma_ik
            a_v=4,  # beta_kk
            b_v=1,  # beta_kk', k neq k'
            eta_km=None,
            nu_k=None,
            mu_jkml=None,
            eval=False,
            seed=1234,
            inference_iter=500,
            empirical_prior=False,
            weak: Optional[int] = None, n_weaks: Optional[int] = None,
                        random_guess: Optional[int] = None):

    y_is_one_lij = []
    y_is_one_lji = []
    num_classes = int(num_classes)
    num_items = int(num_items)
    num_rules = L.shape[1]
    for k in range(int(num_classes)):
        # compute which points are selected and record which rule predicted it
        selected = [[], []]
        for i in range(num_items):
            for j in range(num_rules):
                if L[i, j] == k:
                    selected[0].append(i)
                    selected[1].append(j)
        coo_ij = ssp.coo_matrix((np.ones(len(selected[0])),
                                (selected[0], selected[1])),
                                shape=(int(num_items), num_workers),
                                dtype=np.bool_)

        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    beta_kl = np.eye(int(num_classes)) * (a_v - b_v) + b_v

    # initialize z_ik, zg_ikm, c_ik, gamma_ik, sigma_ik
    z_ik = np.zeros((int(num_items), int(num_classes)))
    for l in range(int(num_classes)):
        z_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1) + 1e-8
    z_ik /= z_ik.sum(axis=-1, keepdims=True)

    if empirical_prior:
        alpha = z_ik.sum(axis=0)

    # np.random.seed(seed)
    rng = np.random.RandomState(seed)
    zg_ikm = rng.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
    for it in range(inference_iter):
        if eval is False:
            eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)
            nu_k = alpha + z_ik.sum(axis=0)
            mu_jkml = np.zeros((num_workers, num_classes, num_groups, num_classes)) + beta_kl[None, :, None, :]
            for l in range(num_classes):
                for k in range(num_classes):
                    mu_jkml[:, k, :, l] += y_is_one_lji[l].dot(zg_ikm[:, k, :])

        Eq_log_pi_km = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
        Eq_log_tau_k = digamma(nu_k) - digamma(nu_k.sum())
        Eq_log_v_jkml = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))

        zg_ikm[:] = Eq_log_pi_km[None, :, :] + Eq_log_tau_k[None, :, None]
        for l in range(num_classes):
            for k in range(num_classes):
                zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, l])

        zg_ikm = np.exp(zg_ikm)
        zg_ikm /= zg_ikm.reshape(num_items, -1).sum(axis=-1)[:, None, None]

        last_z_ik = z_ik
        z_ik = zg_ikm.sum(axis=-1)

        if np.allclose(last_z_ik, z_ik, atol=1e-3):
            break

    ELBO = ((eta_km - 1) * Eq_log_pi_km).sum() + ((nu_k - 1) * Eq_log_tau_k).sum() + (
                (mu_jkml - 1) * Eq_log_v_jkml).sum()
    ELBO += dirichlet.entropy(nu_k)
    for k in range(num_classes):
        ELBO += dirichlet.entropy(eta_km[k])
    ELBO += (gammaln(mu_jkml) - (mu_jkml - 1) * digamma(mu_jkml)).sum()
    alpha0_jkm = mu_jkml.sum(axis=-1)
    ELBO += ((alpha0_jkm - num_classes) * digamma(alpha0_jkm) - gammaln(alpha0_jkm)).sum()
    ELBO += entropy(zg_ikm.reshape(num_items, -1).T).sum()
    return z_ik, ELBO, eta_km, nu_k, mu_jkml, zg_ikm


class EBCC(BaseLabelModel):
    """Enhanced BCC (eBCC)

    Usage:

        ebcc = EBCC(num_groups, a_pi, a_v, b_v, repeat, inference_iter, empirical_prior)
        ebcc.fit(train_data)
        ebcc.test(test_data)

    Parameters:

        num_groups: number of subtypes
        a_pi: The parameter of dirichlet distribution to generate mixture weight.
        a_v: b_kk, number of corrected labeled items under every class.
        b_v: b_kk', all kind of miss has made b_kk' times.
        repeat: ELBO update times.
        inference_iter: Iterations of variational inference.
        empirical_prior: The empirical prior of alpha.
        seed: Random seed.
    """
    def __init__(self,
                 num_groups: Optional[int] = 10,
                 a_pi: Optional[float] = 0.1,
                 alpha: Optional[float] = 1,
                 a_v: Optional[float] = 4,
                 b_v: Optional[float] = 1,
                 repeat: Optional[int] = 5,
                 inference_iter: Optional[int] = 1000,
                 seed: Optional[int] = None,
                 empirical_prior=True,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'num_groups': num_groups,
            'a_pi': a_pi,
            'alpha': alpha,
            'a_v': a_v,
            'b_v': b_v,
            'empirical_prior': empirical_prior,
            'inference_iter': inference_iter,
            **kwargs
        }
        self.params = {
            'seed': None,
            'eta_km': None,
            'nu_k': None,
            'mu_jkml': None,
            'rho_ikm': None,
        }
        self.seed = seed
        self.repeat = repeat

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        L = dataset_train[0]
        num_items = L.shape[0]
        num_classes = np.max(L) + 1
        num_workers = int(L.shape[1])
        max_elbo = float('-inf')
        # seed = np.random.randint(1e8)
        # self.seed = seed
        if self.seed is None:
            for _ in trange(0, self.repeat, unit='epoch'):
                seed = np.random.randint(1e8)
                prediction, elbo, p1, p2, p3, p4 = ebcc_vb(L,
                                                       num_items, num_workers, num_classes,
                                                       seed=seed,
                                                       **self.hyperparas)
                if elbo > max_elbo:
                    print(f'update elbo: new: {elbo}, old: {max_elbo}')
                    self.params = {
                        'seed': seed,
                        'eta_km': p1,
                        'nu_k': p2,
                        'mu_jkml': p3,
                        'rho_ikm': p4
                    }
                    max_elbo = elbo
                    pred = prediction
        else:
            pred, elbo, p1, p2, p3, p4 = ebcc_vb(L,
                                             num_items, num_workers, num_classes,
                                             seed=self.seed,
                                             **self.hyperparas)

            self.params = {
                'seed': self.seed,
                'eta_km': p1,
                'nu_k': p2,
                'mu_jkml': p3,
                'rho_ikm': p4
            }
        return pred

    def predict_proba(self,
                      dataset: Union[BaseDataset, np.ndarray],
                      weak: Optional[int] = None,
                      n_weaks: Optional[int] = None,
                      seed: Optional[int] = None,
                      random_guess: Optional[int] = None,
                      **kwargs: Any):
        evaluate = True
        L = dataset[0]
        num_items = L.shape[0]
        num_classes = np.max(L) + 1
        num_workers = int(L.shape[1])

        if self.params['nu_k'] is None or self.params['mu_jkml'] is None:
            evaluate = False

        pred, elbo, _, _, _, _ = ebcc_vb(L,
                                      num_items, num_workers, num_classes,
                                      eval=evaluate,
                                      eta_km=self.params['eta_km'],
                                      nu_k=self.params['nu_k'],
                                      mu_jkml=self.params['mu_jkml'],
                                      **self.hyperparas,
                                      )
        return pred
