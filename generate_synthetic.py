import os

import numpy as np
from scipy.io import savemat
from numpy.random import dirichlet
from numpy.random import beta
from numpy.random import multinomial

def generate_dataset(p, n, n_val):
    # p rules
    # n training datapoints
    # n_val validation points
    # assume 2 classes
    k = 2


    # dirichlet parameters for ground truth eta
    dir_param = np.ones(2)

    # fix mean, set the variance ourself
    beta_mean = 0.6
    beta_a = 2
    beta_b = (1 / beta_mean - 1) * beta_a

    # generate confusion matrices for all rules
    cm = np.zeros((p, k, k))
    for j in range(p):
        cm[j, 0, 0] = beta(beta_a, beta_b)
        cm[j, 1, 1] = beta(beta_a, beta_b)
        cm[j, 0, 1] = 1 - cm[j, 0, 0]
        cm[j, 1, 0] = 1 - cm[j, 1, 1]

    # generate eta
    eta_all = dirichlet(dir_param, size=n_val + n)

    eta_val = eta_all[:n_val, :].reshape((1, -1))
    eta_train = eta_all[n_val:, :].reshape((1, -1))

    # generate observed ground truth
    val_labels = np.zeros(n_val).astype(int)
    train_labels = np.zeros(n).astype(int)

    for i in range(n_val):
        val_labels[i] = int(multinomial_argmax(eta_all[i, :]))

    for m in range(n):
        train_labels[m] = int(multinomial_argmax(eta_all[n_val + m, :]))

    # generate predictions
    train_pred = np.zeros((n, p))
    val_pred = np.zeros((n_val, p))

    for j in range(p):
        for i in range(n_val):
            val_pred[i, j] = multinomial_argmax(cm[j, val_labels[i], :])

        for i in range(n):
            train_pred[i, j] = multinomial_argmax(cm[j, train_labels[i], :])

    mdic = {
            "train_labels": train_labels,
            "train_pred": train_pred,
            "validation_labels": val_labels,
            "val_pred": val_pred,
            "eta_train": eta_train,
            "eta_val": eta_val,
            }

    return mdic

def multinomial_argmax(dist):
    return np.argmax(multinomial(1, dist))

if __name__ == '__main__':

    # choose number of rules
    n_rules = 10

    # choose number of datapoints
    n_train = 1000

    # choose number of validation points
    n_valid = 100

    if not os.path.exists('./datasets/synthetic/'):
        os.makedirs('./datasets/synthetic/')

    path = './datasets/synthetic/synth_' + str(n_rules) + 'p_' + str(n_train) +\
            'n_' + str(n_valid) + 'nval__'
    for o in range(10):
        dataset = generate_dataset(n_rules, n_train, n_valid)
        full_path = path + str(o) + '.mat'
        savemat(full_path, dataset)
