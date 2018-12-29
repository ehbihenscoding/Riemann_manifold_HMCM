import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from implementation.metric_tensor import BayesianMetric
from implementation.riemann_hmcmc import RiemannHMCMC
from implementation.vanilla_hmc import VanillaHMC


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


alpha = 100.0
epsilon = 1e-1
n_leapfrogs = 100
iter_fixed_point = 5
# n_iter = 1000
n_iter = 100
n_burn_in = 0


# df = pd.read_csv('data/diabetes.csv')
# df = df.iloc[:50]
# y = df.Outcome.values
# X = df.drop(columns='Outcome').values

X = np.random.randn(100, 2)
w = np.array([1.0, 2.0])
z = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
y = np.random.random(100) <= z
y = y.astype(float)
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(int))
# plt.show()


n_examples, dim = X.shape

metric_tensor = BayesianMetric(alpha=alpha, X=X)
p0 = 5*np.random.randn(dim)
# theta_0 = np.array([2.0, 1.0]) # 5*np.random.randn(dim)
theta_0 = np.array([10.0, 1.0]) # 5*np.random.randn(dim)



def grad_theta_h(theta, p):
    """
    Gradient of H wrt theta
    :param theta:
    :param p:
    :return:
    """
    posterior_d_deriv = -theta/alpha**2
    posterior_d_deriv += np.dot((y - sigmoid(np.dot(X, theta))).T,
                                X)
    posterior_d_deriv = - posterior_d_deriv

    # We need these 2 followings quantities to compute other derivative terms
    # This part takes the most computations: problematic ...
    dg = metric_tensor.derivative(theta)
    g_inv = metric_tensor.value_inv(theta)

    prod = np.dot(g_inv, dg)  # d*d*d matrix where the last axis corresponds to the derivative index
    trace_term = 0.5*np.trace(prod, axis1=0, axis2=1)
    left = np.dot(p.T, g_inv)
    right = left.T
    first_mul = np.tensordot(left, dg, axes=(0, 0))
    final = np.tensordot(first_mul, right, axes=(0, 0))
    prod_term = -0.5*final
    return posterior_d_deriv + trace_term + prod_term


def h(theta, p):
    """
    UP TO AN ADDITIVE CONSTANT
    :param theta:
    :param p:
    :return:
    """
    posterior_term = - 0.5/alpha*np.dot(theta.T, theta)
    prod = np.dot(X, theta)
    posterior_term += np.dot(y.T, np.log(sigmoid(prod)))
    posterior_term += np.dot((1.0 - y).T, np.log(sigmoid(-prod)))

    g_inv = metric_tensor.value_inv(theta)
    middle_term = -0.5*np.log(np.linalg.det(g_inv))  # G_inv considered here hence the neg sign
    last_term = 0.5*np.dot(np.dot(p.T, g_inv), p)

    h = -posterior_term + middle_term + last_term
    return h


if __name__ == '__main__':
    # Riemann Manifold HMC
    hmcmc = RiemannHMCMC(riemann_metric=metric_tensor, h=h, grad_theta_h=grad_theta_h, p0=p0,
                         theta_0=theta_0, epsilon=epsilon, n_leapfrogs=n_leapfrogs,
                         iter_fixed_point=iter_fixed_point)
    sample, acceptance_rates = hmcmc.sample(n_iter=n_iter, n_burn_in=n_burn_in)

    # Vanilla HMC
    import tensorflow as tf
    X_tensor = tf.convert_to_tensor(value=X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(value=y, dtype=tf.float32)
    theta_0_tensor = tf.convert_to_tensor(value=theta_0, dtype=tf.float32)

    def log_posterior(x):
        x = tf.reshape(x, [-1])
        # Prior contribution
        res = -0.5/alpha**2 * tf.reduce_sum(x**2)
        # y|theta contribution
        pred = (X_tensor * tf.reshape(x, shape=[1, -1]))
        pred = tf.reduce_sum(pred, axis=1)
        res += tf.reduce_sum(y_tensor * tf.log(tf.sigmoid(pred)))
        res += tf.reduce_sum((1.0 - y_tensor) * tf.log(tf.sigmoid(-pred)))
        return res


    vanilla_hmcmc = VanillaHMC(log_prob=log_posterior,
                               step_size=epsilon,
                               num_leapfrog_steps=n_leapfrogs,
                               theta_0=theta_0_tensor)

    vanilla_sample, vanilla_acceptance_rates = vanilla_hmcmc.sample(n_iter=n_iter,

                                                                    n_burn_in=n_burn_in)
    sample = np.insert(sample, 0, theta_0, axis=0)
    vanilla_sample = np.insert(vanilla_sample, 0, theta_0, axis=0)
    # Plots and prints
    print(acceptance_rates.mean())
    plt.plot(sample[:-10, 0], sample[:-10, 1], 'bx--', label='RHMC')
    # plt.plot(sample[-10:, 0], sample[-10:, 1], 'x', c=np.ones(10))
    # plt.show()
    print(vanilla_acceptance_rates.mean())
    plt.plot(vanilla_sample[:-10, 0], vanilla_sample[:-10, 1], 'ro--', label='Vanilla HMC')
    # plt.plot(vanilla_sample[-10:, 0], vanilla_sample[-10:, 1], c=2*np.ones(10))
    plt.legend()
    plt.show()


