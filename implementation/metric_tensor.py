import numpy as np


class MetricTensor:
    def __init__(self, sep):
        """
        Skeletton of all metric tensors.
        Classes (corresponding to manifold metrics) have 2 methods:
        - value which returns G_i(\theta)
        - value_inv which returns G_i^{-1}(\theta)
        :param sep:
        """
        self.sep = sep

    def value(self, theta):
        pass

    def value_inv(self, theta):
        pass


class BayesianMetric(MetricTensor):
    def __init__(self, alpha, X):
        super(BayesianMetric, self).__init__(sep=False)
        self.X = X
        self.alpha = alpha
        self.n_examples = self.X.shape[0]
        self.dim = self.X.shape[1]

        self.XXt = np.dot(self.X, self.X.T)

    def value(self, theta):
        diag_gamma = np.dot(theta.T, self.X.T)
        logistic_term = self.logistic_fn(diag_gamma)
        diag_gamma = logistic_term * (1.0 - logistic_term)
        gamma = np.diag(diag_gamma)
        assert gamma.shape == (self.n_examples, self.n_examples)
        res = np.dot(np.dot(self.X.T, gamma), self.X)
        res = res + 1.0/self.alpha*np.eye(res.shape[0])
        return res

        # if not self.sep:
        #     return np.dot(self.X.T, self.X) + 1.0 / self.alpha*np.eye(self.dim)
        #
        # if self.sep:
        #     pass

    def value_inv(self, theta):
        """
        Using Sherman-Morrison Woodbury formula
        :param theta:
        :return:
        """
        # diag_gamma = np.dot(theta.T, self.X.T)
        # logistic_term = self.logistic_fn(diag_gamma)
        # diag_gamma = logistic_term * (1.0 - logistic_term)
        # diag_gamma_inv = 1.0 / diag_gamma
        # gamma_inv = np.diag(diag_gamma_inv)
        # inv_mat = np.linalg.pinv(gamma_inv + self.XXt)
        # return self.alpha*(np.eye(self.dim) - np.dot(np.dot(self.X.T, inv_mat), self.X))
        G = self.value(theta)  # d*d matrix inversion
        return np.linalg.pinv(G)

    def derivative(self, theta):
        """
        This method returns dg containing all partial derivatives at once
        and has shape d*d*d
        More precisely, dg[:, :, i] corresponds to \frac{\partial G}{\partial \theta_i}
        :param theta:
        :return:
        """
        diag_gamma = np.dot(theta.T, self.X.T)
        logistic_term = self.logistic_fn(diag_gamma)
        diag_gamma = logistic_term * (1.0 - logistic_term)
        gamma = np.diag(diag_gamma)

        # v computation
        diags_v = 1.0 - 2*self.logistic_fn(np.dot(theta.T, self.X.T))
        diags_v = diags_v.reshape((-1, 1))
        diags_v = diags_v*self.X
        assert diags_v.shape == self.X.shape  #N*d shape

        XtGamma = np.dot(self.X.T, gamma)  # d*N shape

        # TODO: Verifier car pas sur de mon coup ... et surtout plus long...
        # id = np.eye(self.n_examples).reshape((self.n_examples, self.n_examples, 1))
        # diags_v = diags_v.reshape((self.n_examples, 1, self.dim))
        # v = id*diags_v  # n*n*d tensor
        # left = np.tensordot(XtGamma, v, axes=(1, 0))  # shape d*N*d
        # assert left.shape == (self.dim, self.n_examples, self.dim)
        # dg = np.tensordot(left, self.X, axes=(1, 0))
        # dg = np.swapaxes(dg, axis1=-2, axis2=-1)

        dg = np.zeros((self.dim, self.dim, self.dim))
        for idx, v_i_diag in enumerate(diags_v.T):
            v_i = np.diag(v_i_diag)
            dg_di = np.dot(np.dot(XtGamma, v_i), self.X)
            dg[:, :, idx] = dg_di
        return dg

    @staticmethod
    def logistic_fn(x):
        return 1.0 / (1.0 + np.exp(-x))