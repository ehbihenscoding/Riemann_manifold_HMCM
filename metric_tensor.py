import numpy as np


class MetricTensor:
    def __init__(self, sep):
        self.sep = sep

    def value(self, theta):
        pass

    def value_inv(self, theta):
        pass


class BayesianMetric(MetricTensor):
    def __init__(self, sep, alpha, X):
        super(MetricTensor, self).__init__(sep)
        self.X = X
        self.alpha = alpha
        self.dim = self.X.shape[1]

    def value(self, theta):
        if not self.sep:
            return np.dot(self.X.T, self.X) + 1.0 / self.alpha*np.eye(self.dim)

        if self.sep:
            # TODO: Continuer
            pass



