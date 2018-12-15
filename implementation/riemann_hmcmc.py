import numpy as np

from metric_tensor import MetricTensor
from random_vector import RandomVector, PDynamic


class RiemannHMCMC:
    def __init__(self, riemann_metric: MetricTensor, grad_theta_h, p0, theta_0,
                 epsilon, n_leapfrogs=7, iter_fixed_point=7):
        self.riemann_metric = riemann_metric
        self.grad_theta_h = grad_theta_h
        self.epsilon = epsilon
        self.n_leapfrogs = n_leapfrogs
        self.iter_fixed_point = iter_fixed_point

        # Initialize P and Theta objects
        self.p = PDynamic(grad_theta_H=self.grad_theta_h,
                          p=p0,
                          iter_fixed_point=iter_fixed_point)

        self.theta = RandomVector(G=self.riemann_metric,
                                  theta=theta_0,
                                  iter_fixed_point=iter_fixed_point)

    def sample(self, n_iter, n_burn_in=1000):
        # Second step: RiemannHMCMC dynamics
        sample = []
        for id_iter in range(n_iter+n_burn_in):
            theta_eps = None
            for leap in range(self.n_leapfrogs):
                p_half_leap = self.p.p_half_leap(self.theta.theta, epsilon=self.epsilon)
                theta_eps = self.theta.update(p_half_leap=p_half_leap, epsilon=self.epsilon)
                self.p.update(theta=theta_eps, epsilon=self.epsilon)

            if id_iter > n_burn_in:
                sample.append(theta_eps)
