from typing import Optional, Any, Union
import numpy as np
import cvxpy as cp
import scipy as sp
from numpy.matlib import repmat
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from wrench.basemodel import BaseLabelModel
from wrench.dataset import BaseDataset


class AMCL_CC(BaseLabelModel):
    """AMCL Convex Combination (AMCL_CC)

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
                 logger,
                 seed: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.seed = seed
        self.model_theta = None
        self.n_samples = None
        # M by C matrix of costs (M is num (train) points, C is num class)
        # meant to be parameter for LP
        self.cost_matrix = None
        self.logger = logger

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Union[BaseDataset, np.ndarray],
            y_valid: Optional[np.ndarray] = None,
            n_samples: Optional[int] = -1,
            verbose: Optional[bool] = False,
            eps = 0.3,
            T = 500,
            n_tries = 5,
            *args: Any,
            **kwargs: Any):

        L_train = dataset_train[0]
        y_train = dataset_train[1]
        L_valid = dataset_valid[0]
        y_valid = np.squeeze(dataset_valid[1])

        num_classes = np.max(y_valid) + 1
        num_items = L_train.shape[0]
        num_items_valid = L_valid.shape[0]
        num_label_fns = int(L_train.shape[1])

        # sample labeled data
        self.n_samples = n_samples
        if self.n_samples > 0 and num_items_valid > self.n_samples:
            choices = np.random.choice(np.arange(num_items_valid),
                    self.n_samples, replace=False)
            y_valid = y_valid[choices]
            L_valid = L_valid[choices, :]

        num_items_valid = L_valid.shape[0]

        # unlabeled_votes
        L_train_aug = self._initialize_L_aug(L_train, num_classes)
        # labeled_votes
        L_valid_aug = self._initialize_L_aug(L_valid, num_classes)
        # labeled_labels
        y_valid_aug = self._initialize_one_hot_labels(y_valid, num_classes)
        y_train_aug = self._initialize_one_hot_labels(y_train, num_classes)

        # SET EPS here
        # eps and T have defaults in fit function
        L = 2 * np.sqrt(num_label_fns + 1)
        squared_diam = 2
        # T = int(np.ceil(L*L*squared_diam/(eps*eps)))
        h = eps/(L*L)

        Y, constraints, used_lfs = self.compute_constraints_with_loss2(
                self.brier_loss_linear_vectorized, self.brier_score_amcl_vectorized,
                L_train_aug, L_valid_aug, y_valid_aug, y_train_aug)

        # initial weights
        # used_lfs will give a 0 in all positions where the LF is not used
        # and equally distribute the mass to all other LFs.
        init_theta = used_lfs * np.mean(used_lfs)

        model_theta = self.sub_gradient_method2(L_train_aug, Y, constraints,
                self.brier_loss_linear_vectorized, self.linear_combination_labeler_vectorized,
                self.project_to_simplex, init_theta, T, h, num_label_fns,
                num_items, num_classes)

        self.model_theta = model_theta

        # if model_theta is None, return that
        if model_theta is None:
            return None

        pred = self.predict_proba(dataset_train)

        return pred

        raise ValueError(f"Cannot generate feasible region after {n_tries} attempts")


    def predict_proba(self,
                      dataset: Union[BaseDataset, np.ndarray],
                      **kwargs: Any):
        L = dataset[0]
        num_items = L.shape[0]
        num_classes = int(np.max(L) + 1)
        num_label_fns = int(L.shape[1])

        pred = np.zeros((num_items, num_classes))

        L_aug = self._initialize_L_aug(L, num_classes)
        for i in range(num_label_fns):
            pred += L_aug[i] * self.model_theta[i]

        return pred

    def _initialize_L_aug(self, L, n_class):
        L = L.T
        L_aug = (np.arange(n_class) == L[..., None]).astype(int)
        return L_aug

    def _initialize_one_hot_labels(self, y, n_class):
        return np.squeeze(self._initialize_L_aug(y, n_class))

    # Taken from
    # https://github.com/BatsResearch/amcl/blob/main/algorithms/util.py#L92
    def project_to_simplex(self, v):
        '''
        Project a vector to the simplex
        Code - implementation taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        '''

        v = np.array(v)
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/util.py#L61
    def linear_combination_labeler(self, theta, X):
        '''
        Implementation of a linear combination (convex) of the weak classifiers.
        Note we restrict the sum of theta to be 1

        Args:
        theta - params, X - data
        '''

        num_wls, num_classes = np.shape(X)
        cum = np.zeros(num_classes)
        for i in range(num_wls):
            cum = np.add(cum, np.multiply(X[i],theta[i]))
        return cum

    def linear_combination_labeler_vectorized(self, theta, X):
        '''
        Implementation of a linear combination (convex) of the weak classifiers.
        Note we restrict the sum of theta to be 1

        Args:
        theta - params (1d vector of length N, # labeling functions)
        X - (N, M, C) (M is # unlabeled points, C is # classes)
        '''

        # weight every LF's prediction by its respective theta then sum along
        # axis 0
        return np.tensordot(theta, X, axes=1)

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/util.py#L9
    def brier_loss_linear(self, labels, preds):
        '''
        Computing brier score given labels and predictions
        '''
        # Check if y_1 and y_2 have the same length
        if len(labels) != len(preds):
            raise NameError('Computing loss on vectors of different lengths')

        sq = np.sum(np.square(preds))
        tmp1 = -np.square(preds) + sq
        tmp2 = np.square(1.0 - preds)
        x = np.dot(labels,tmp1) + np.dot(labels,tmp2)
        return x / 2

    def brier_loss_linear_vectorized(self, labels, preds):
        '''
        Computing brier score given labels and predictions

        Args:
        labels - array with shape (M, C) (M is # points, C is # class)
        preds - array with shape (M, C) (M is # points, C is # class)
        '''
        # Check if y_1 and y_2 have the same length
        if labels.shape[0] != preds.shape[0]:
            raise NameError('Computing loss on different number of datapoints')

        sq = np.sum(np.square(preds), axis=1, keepdims=True)
        tmp1 = -np.square(preds) + sq
        tmp2 = np.square(1.0 - preds)
        x = np.multiply(labels,tmp1) + np.multiply(labels,tmp2)
        return np.sum(x, axis=1) / 2

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/util.py#L23
    def brier_score_amcl(self, preds):
        '''
        Computing brier score given predictions against all possible labelings
        '''

        sq = np.sum(np.square(preds))
        tmp1 = -np.square(preds) + sq
        tmp2 = np.square(1.0 - preds)
        return (tmp1 + tmp2) / 2

    def brier_score_amcl_vectorized(self, preds):
        '''
        Computing brier score given labels and predictions

        Args:
        preds - array with shape (M, C) (M is # points, C is # class)
        '''
        sq = np.sum(np.square(preds), axis=1, keepdims=True)
        tmp1 = -np.square(preds) + sq
        tmp2 = np.square(1.0 - preds)
        return (tmp1 + tmp2) / 2

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/util.py#L160
    def compute_gradient_comb(self, theta, X, Y, h):
        '''
        Gradient computation for convex combination with Brier Loss

        Args:
        X - data, Y - labels, h - convex combination function (def. above)
        '''
        X = np.array(X)
        Y = np.array(Y)
        pred = h(theta, X)
        grad = np.array([2*np.dot(X[:,j],(pred[j]-Y[j])) for j in range(len(Y))])
        grad = np.average(grad,axis=0)
        return grad

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/subgradient_method.py#L273
    def compute_constraints_with_loss2(self, lf1, lf2, output_labelers_unlabeled, output_labelers_labeled, true_labels, train_labels, lr=False):
        '''
        Generating constraints for CVXPY implementation

        Args:
        lf1 - loss function (regular w/ two arguments)
        lf2 - loss function for adversarial labelling 
        output_labelers_unlabeled - votes of weak supervision on unlabeled data
        output_labelers_labeled - votes of weak supervision on labeled data
        true_labels - labels of labeled data
        '''

        N = len(output_labelers_unlabeled) # Number of weak classifiers
        M = len(output_labelers_unlabeled[0]) # Number of unlabeled data points
        C = len(output_labelers_unlabeled[0][0]) # Number of classes
        Ml = len(output_labelers_labeled[0]) # Number of labeled data points

        self.logger.info("Num WL: %d, Num Unlab %d, Num Classes %d, Num Lab %d" % (N,M,C,Ml))

        # Bounds: risk of a labeler must be within error+-offset
        delta = 0.1 # Between 0 and 1. Probability of the true labeling to NOT belong to the feasible set.
        B = 1  # Size of the range of the loss function
        scaling_factor = 0.4 # Direct computation of the offset could yield large values if M or Ml is small.
                             # This number can be used to scale the offset if it is too large
        # offset computation has been moved into loop below to accomodate
        # absentions on labeled data

        if Ml != len(true_labels):
            raise NameError('Labeled data points and label sizes are different')

        # Variables
        Y = cp.Variable((M,C),nonneg=True)

        # Constraint vector
        constraints = []

        # Add constraints for the sum of the label probabilities for each item to sum to 1
        constraints.append(cp.sum(Y, axis=1) == 1)

        used_constraints = np.ones(N)
        # Build constraint for weak classifier
        for i in range(N):
            # Compute the expected error over the labeled data for each weak classifier

            # compute number of labeled and unlabeled predictions
            n_lab_preds_i = np.sum(output_labelers_labeled[i])
            n_unlab_preds_i = np.sum(output_labelers_unlabeled[i])

            # Compute the coefficient of the linear constraint based on the brier score error
            # make this here becuase we'll just set it to all 0's if there are
            # no labeled points to estimate the bounds
            build_coefficients = np.zeros((M,C))

            pred_on_labeled = np.sum(output_labelers_labeled[i], axis=1).astype(bool)
            if np.sum(pred_on_labeled) == 0:
                # since we have no labeled data, we won't use the labeling
                # function at all.  I.e. we will have a trivial constraint 0=0
                cons_lb = 0
                cons_ub = 0
                offset = 0
                used_constraints[i] = 0
            else:
                error = np.mean(lf1(true_labels[pred_on_labeled, :],\
                        output_labelers_labeled[i][pred_on_labeled, :]))

                offset = B * scaling_factor * np.sqrt(
                        (n_lab_preds_i + n_unlab_preds_i)\
                                * np.log(4 * N / delta)\
                                / (2 * (n_lab_preds_i * n_unlab_preds_i)))
                # offset = 0 # Uncomment this line
                        # if you do not want to have a offset. This could be better in practice if
                        #  the number of labeled data and labeled data is very large

                pred_on_unlabeled = np.sum(output_labelers_unlabeled[i], axis=1).astype(bool)
                error2 = np.mean(lf1(train_labels[pred_on_unlabeled, :], output_labelers_unlabeled[i][pred_on_unlabeled, :]))
                valid_i_constraint = (error + offset )>= error2 and (error-offset)<= error2
                # print(valid_i_constraint)

                cons_lb = error - offset
                cons_ub = error + offset

                build_coefficients = lf2(output_labelers_unlabeled[i])/n_unlab_preds_i
                # set rows to all 0's if abstained on
                not_pred_on_unlabeled = (1 - np.sum(output_labelers_unlabeled[i], axis=1)).astype(bool)
                build_coefficients[not_pred_on_unlabeled, :] = 0

            # make matrix sparse
            build_coefficients = csr_matrix(build_coefficients)

            if(offset != 0):
                constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) <= cons_ub)
                constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) >= cons_lb)
            else:
                constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) == cons_lb)

        return Y, constraints, used_constraints

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/subgradient_method.py#L341
    def solve_lp_given_cost2(self, Y, constraints, cost, prob=None):
        if prob is None:
            self.cost_matrix = cp.Parameter(Y.shape)
            obj = cp.Minimize(cp.sum(cp.multiply(Y, self.cost_matrix)))
            prob = cp.Problem(obj, constraints)

        self.cost_matrix.value = cost
        # prob.solve(solver=cp.ECOS, eps=1e-8)
        # prob.solve(solver=cp.ECOS)
        prob.solve(solver=cp.GUROBI)
        # prob.solve(solver=cp.GUROBI, verbose=True, DualReductions=0)

        return np.reshape(Y.value, -1), prob.value, prob

    # https://github.com/BatsResearch/amcl/blob/main/algorithms/subgradient_method.py#L350
    def sub_gradient_method2(self, X_unlabeled, Y, constraints, lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):
        '''
        Running the subgradient method (via LP with cvxpy)

        Args:
        * X_unlabeled => unlabeled data (M, *) - i.e. for votes it is (M, N, C)
        * Y => cvxpy variables for LP
        * constraints => constraints of the feasible set
          as computed from the method compute_constraints_with_loss
        * lf => loss function
        * h => prediction model (the first argument is the weights of the model)
        * proj_function => projection method for the weights of the prediction model
        * initial_theta => initial weights of the prediction model
        * iteration => number of iterations
        * step_size => step size for the subgradient descent method
        * N => number of weak classifiers
        * M => number of unlabeled data points
        * C => number of classes
        * lr => flag: true if we are using multinomial logistic regression - false if we are using
          convex combination of weak classifiers
        '''
        # Evaluation for convex combination of weak classifier
        def eval_theta(th):
            cum = np.mean(lf(new_y, h(th, X_unlabeled)))

            return cum

        # Evaluation for multinomial logistic regression
        def eval_lr(th):
            cum = 0
            for j in range(M):
                cum += lf(new_y[j], h(th,X_unlabeled[j]))
            return cum/M

        # Current value of the minimax
        best_val = 10e10 # Initialized to a very high value
        theta = initial_theta # Weights of the model

        preds = h(theta, X_unlabeled)

        one_hots = np.eye(C)
        cost = np.zeros(M * C)
        for c in range(C):
            cost[c::C] = -lf(repmat(one_hots[c], M, 1), preds) / M

        # Find labeling that maximizes the error
        cost = cost.reshape((M, C))
        new_y_vec, _, lp = self.solve_lp_given_cost2(Y, constraints, cost)

        # exit and restart if the problem is infeasible
        if lp.status != 'optimal':
            return None

        new_y = new_y_vec.reshape((M,C))

        # Subgradient method core implementation
        for t in range(iteration):

            # Compute subgradient with respect to theta and the current labeling of the unlabeled data
            grad = self.compute_gradient_comb(theta, X_unlabeled, new_y, h)

            # Gradient descent step
            theta -= grad * step_size
            # Projection step
            theta = proj_function(theta)

            preds = h(theta, X_unlabeled)

            cost = np.zeros(M * C)
            for c in range(C):
                cost[c::C] = -lf(repmat(one_hots[c], M, 1), preds) / M

            # Find labeling that maximizes the error
            cost = cost.reshape((M, C))
            new_y_vec, obj, _ = self.solve_lp_given_cost2(Y, constraints, cost, prob=lp)
            new_y = new_y_vec.reshape((M, C))

            # Evaluate the current model with respect to the worst-case error
            current_eval = eval_theta(theta)
            # If the current model is better, update the best model found
            if (current_eval < best_val):
                best_theta = theta.copy()
                best_val = current_eval

            # Debug lines
            if t % 100 == 0:

                totals = h(best_theta, X_unlabeled)
                vals = lf(new_y, totals)

                self.logger.info("Bound at time %d: %f" % (t, np.mean(vals)))
                # don't print best params because that takes up a ton of space
                # if not lr:
                #     print("Best Params:", best_theta)

        # Debug lines
        self.logger.info("ENDING PARAMETERS: " + str(best_theta))

        # Return the best model found during the subgradient method execution
        return best_theta
