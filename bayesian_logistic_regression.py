from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.special import expit as sigmo

from implementation.metric_tensor import BayesianMetric, BayesianEmpiricalMetric
from implementation.riemann_hmcmc import RiemannHMCMC
from implementation.vanilla_hmc import VanillaHMC


def sigmoid(a):
    # return sigmo(a)
    return 1.0 / (1.0 + np.exp(-a))


alpha = 100.0

# epsilon = 1e-1
# n_leapfrogs = 100

epsilon = 1e-1
n_leapfrogs = 6

iter_fixed_point = 6
n_iter = 1000
# n_iter = 100
n_burn_in = 0


df = pd.read_csv('data/diabetes.csv')
df = df.iloc[:50]
y = df.Outcome.values
X = df.drop(columns='Outcome').values
X = np.hstack((X, np.ones((len(X), 1))))

from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, y)
theta_0 = mdl.coef_.squeeze()


# 1D Version
# X = np.random.randn(100, 1)
# w = np.array([2.0])
# z = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
# y = np.random.random(100) <= z
# y = y.astype(float)
# theta_0 = np.array([-100.0])

# 2D Version
# X = np.random.randn(100, 2)
# w = np.array([1.0, 2.0])
# z = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
# y = np.random.random(100) <= z
# y = y.astype(float)
# theta_0 = np.array([10.0, 1.0])
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(int))
# plt.show()


approx_metric_tensor = BayesianEmpiricalMetric(X=X, y=y)
metric_tensor = BayesianMetric(alpha=alpha, X=X)
n_examples, dim = X.shape
p0 = 5*np.random.randn(dim)


def grad_theta_h(theta, p, my_metric_tensor):
    """
    Gradient of H wrt theta
    :param theta:
    :param p:
    :param my_metric_tensor:

    :return:
    """
    posterior_d_deriv = -theta/alpha**2
    posterior_d_deriv += np.dot((y - sigmoid(np.dot(X, theta))).T,
                                X)
    posterior_d_deriv = - posterior_d_deriv

    # We need these 2 followings quantities to compute other derivative terms
    # This part takes the most computations: problematic ...
    dg = my_metric_tensor.derivative(theta)
    g_inv = my_metric_tensor.value_inv(theta)

    prod = np.dot(g_inv, dg)  # d*d*d matrix where the last axis corresponds to the derivative index
    trace_term = 0.5*np.trace(prod, axis1=0, axis2=1)
    left = np.dot(p.T, g_inv)
    right = left.T
    first_mul = np.tensordot(left, dg, axes=(0, 0))
    final = np.tensordot(first_mul, right, axes=(0, 0))
    prod_term = -0.5*final
    return posterior_d_deriv + trace_term + prod_term


def h(theta, p, my_metric_tensor):
    """
    UP TO AN ADDITIVE CONSTANT
    :param theta:
    :param p:
    :param my_metric_tensor:
    :return:
    """
    posterior_term = - 0.5/alpha*np.dot(theta.T, theta)
    prod = np.dot(X, theta)
    # posterior_term += np.dot(y.T, np.log(sigmoid(prod)))
    # posterior_term += np.dot((1.0 - y).T, np.log(sigmoid(-prod)))
    posterior_term += np.dot(y.T, np.log(sigmoid(prod)))
    posterior_term += np.dot((1.0 - y).T, np.log(sigmoid(-prod)))

    g_inv = my_metric_tensor.value_inv(theta)
    middle_term = -0.5*np.log(np.linalg.det(g_inv))  # G_inv considered here hence the neg sign
    last_term = 0.5*np.dot(np.dot(p.T, g_inv), p)

    h = -posterior_term + middle_term + last_term
    return h


fi_grad_theta_h = partial(grad_theta_h, my_metric_tensor=metric_tensor)
fi_h = partial(h, my_metric_tensor=metric_tensor)
afi_grad_theta_h = partial(grad_theta_h, my_metric_tensor=approx_metric_tensor)
afi_h = partial(h, my_metric_tensor=approx_metric_tensor)


if __name__ == '__main__':
    # # Riemann Manifold HMC
    hmcmc = RiemannHMCMC(riemann_metric=metric_tensor,
                         h=fi_h,
                         grad_theta_h=fi_grad_theta_h,
                         p0=p0, theta_0=theta_0, epsilon=epsilon, n_leapfrogs=n_leapfrogs,
                         iter_fixed_point=iter_fixed_point)
    sample, acceptance_rates = hmcmc.sample(n_iter=n_iter, n_burn_in=n_burn_in)

    # Rieman Manifold HMC using approx fisher information
    # hmcmc = RiemannHMCMC(riemann_metric=approx_metric_tensor,
    #                      h=afi_h,
    #                      grad_theta_h=afi_grad_theta_h,
    #                      p0=p0, theta_0=theta_0, epsilon=epsilon, n_leapfrogs=n_leapfrogs,
    #                      iter_fixed_point=iter_fixed_point)
    # sample_afi, acceptance_rates_afi = hmcmc.sample(n_iter=n_iter, n_burn_in=n_burn_in)

    # # Vanilla HMC
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

    with tf.variable_scope('step-size', reuse=True):
        vanilla_hmcmc = VanillaHMC(log_prob=log_posterior,
                                   step_size=0.5,
                                   num_leapfrog_steps=100,
                                   theta_0=theta_0_tensor)

        vanilla_sample, vanilla_acceptance_rates = vanilla_hmcmc.sample(n_iter=n_iter,
                                                                        n_burn_in=n_burn_in)

    sample = np.insert(sample, 0, theta_0, axis=0)
    vanilla_sample = np.insert(vanilla_sample, 0, theta_0, axis=0)


    # # 1D plot
    # from utils import autocorr_function
    #
    # lags, autocorrs = autocorr_function(sample)
    # _, vanilla_autocorrs = autocorr_function(vanilla_sample)
    #
    # plt.plot(lags, autocorrs, label='RHMC')
    # plt.plot(lags, vanilla_autocorrs, label='Vanilla')
    # plt.legend()
    # plt.show()

    # 2D plots
    # Plots and prints

    print(acceptance_rates.mean())
    first = 5
    n_plotted = 100
    plt.title('First few samples drawn (zoomed)')
    plt.plot(sample[first:n_plotted, 0], sample[first:n_plotted, 1], 'bx--', label='RMHMC')
    # plt.show()
    print(vanilla_acceptance_rates.mean())
    plt.plot(vanilla_sample[first:n_plotted, 0], vanilla_sample[first:n_plotted, 1], 'ro--',
             label='Vanilla HMC')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].scatter(sample[100:-10, 0],
                    sample[100:-10, 1], label='RHMC', c='blue')
    axes[0].set_title('RMHMC')
    axes[1].scatter(vanilla_sample[100:-10, 0],
                    vanilla_sample[100:-10, 1], label='Vanilla', c='red')
    axes[1].set_title('HMC Vanilla')

    axes[0].plot(sample[100:-10, 0].mean(),
                 sample[100:-10, 1].mean(), 'ko')
    axes[1].plot(vanilla_sample[100:-10, 0].mean(),
                 vanilla_sample[100:-10, 1].mean(), 'ko')

    axes[0].set(adjustable='box-forced', aspect='equal')
    axes[1].set(adjustable='box-forced', aspect='equal')
    plt.legend()
    plt.title('Sampled Posterior')
    plt.show()

    # plt.scatter(sample_afi[100:-10, 0],
    #             sample_afi[100:-10, 1], label='RHMC')
    # plt.legend()
    # plt.show()

    ## REAL DATASET
    from tensorflow_probability.python.mcmc.diagnostic import effective_sample_size
    with tf.Session() as sess:
        ess = sess.run(effective_sample_size(sample))
        ess_vanilla = sess.run(effective_sample_size(vanilla_sample))
    print(ess)
    print(ess_vanilla)

    from tensorflow.contrib.distributions.python.ops import sample_stats
    auto_correlation = sample_stats.auto_correlation
    with tf.Session() as sess:
        ac = sess.run(auto_correlation(tf.convert_to_tensor(sample), axis=0))
        ac_vanilla = sess.run(auto_correlation(tf.convert_to_tensor(vanilla_sample), axis=0))

    lags = np.arange(1, 100)
    autocorrs = ac[lags].mean(axis=-1)
    vanilla_autocorrs = ac_vanilla[lags].mean(axis=-1)
    plt.plot(lags, autocorrs, label='RMHMC')
    plt.plot(lags, vanilla_autocorrs, label='Vanilla')
    plt.legend()
    plt.title('Autocorrelation (first lags)')
    plt.show()

    # autocorrs_10 = autocorrs.copy()
    # acceptance_10 = acceptance_rates.mean()
    #
    # autocorrs_25 = autocorrs.copy()
    # acceptance_25 = acceptance_rates.mean()
    #
    # autocorrs_50 = autocorrs.copy()
    # acceptance_50 = acceptance_rates.mean()
    #
    # plt.plot(lags, autocorrs_10, label='10 leapfrogs')
    # plt.plot(lags, autocorrs_25, label='25 leapfrogs')
    # plt.plot(lags, autocorrs_50, label='50 leapfrogs')
    #
    # plt.legend()
    # plt.title('Autocorrelation (first lags)')
    # plt.show()
