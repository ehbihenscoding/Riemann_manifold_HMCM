import numpy as np
from tqdm import tqdm

from metric_tensor import MetricTensor
from random_vector import RandomVector, PDynamic


class RiemannHMCMC:
    def __init__(self, riemann_metric: MetricTensor, h, grad_theta_h, p0, theta_0,
                 epsilon, n_leapfrogs=7, iter_fixed_point=7):
        """

        :param riemann_metric: MetricTensor inherited object
        :param grad_theta_h: function that returns the value of \nabla_\theta H
        :param p0: Initial value for p
        :param theta_0: Initial value for \theta
        :param epsilon: Epsilon used in the numerical scheme
        :param n_leapfrogs: Number of leapfrogs
        :param iter_fixed_point: Number of iterations used in fixed point algorithm (if needed)
        """
        self.riemann_metric = riemann_metric
        self.h = h
        self.grad_theta_h = grad_theta_h
        self.epsilon = epsilon
        self.n_leapfrogs = n_leapfrogs
        self.iter_fixed_point = iter_fixed_point

        self.dim = theta_0.shape

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
        acceptance_ratios = []
        for id_iter in tqdm(range(n_iter+n_burn_in)):
            starting_theta = self.theta.theta

            # starting_p = self.p.p
            mean = np.zeros(self.dim)
            cov = self.riemann_metric.value(starting_theta)
            starting_p = np.random.multivariate_normal(mean=mean, cov=cov)
            self.p.p = starting_p

            theta_eps = None
            n_leapfrogs = np.random.randint(low=1, high=self.n_leapfrogs+1)
            for leap in range(n_leapfrogs):
                # direction = 2*int(np.random.random() >= .5) - 1
                direction = 1
                assert np.abs(direction) == 1
                p_half_leap = self.p.half_leap_update(self.theta.theta,
                                                      epsilon=direction*self.epsilon)
                theta_eps = self.theta.update(p_half_leap=p_half_leap,
                                              epsilon=direction*self.epsilon)
                self.p.update(theta=theta_eps,
                              epsilon=direction*self.epsilon)

            candidate_theta = self.theta.theta
            candidate_p = self.p.p
            # print(candidate_p)
            # Acceptation/rejection precedure
            diff = -self.h(candidate_theta, candidate_p) + self.h(starting_theta, starting_p)
            acceptance_ratio = np.minimum(1.0, np.exp(diff))
            if np.random.random() <= acceptance_ratio:
                new_theta = candidate_theta

                # new_p = candidate_p
                # No need to change theta and p objects in this case
            else:
                new_theta = starting_theta
                self.theta.theta = new_theta

                # new_p = starting_p
                # self.p.p = new_p

            acceptance_ratios.append(acceptance_ratio)

            if id_iter > n_burn_in:
                sample.append(new_theta)
        return np.array(sample), np.array(acceptance_ratios)
