import numpy as np

from metric_tensor import MetricTensor


class RandomVector:
	def __init__(self, G: MetricTensor, theta, iter_fixed_point=7):
		self.G = G
		self.theta = theta
		self.iter_fixed_point = iter_fixed_point

	def update(self, p, epsilon):
		"""
		Given a value of p(t+epsilon/2), computes an update of theta
		Formally, this method returns theta(tau+epsilon) based on its previous value

		:param p: p corresponding to the half leap
		:param epsilon:
		:return:
		"""
		if not self.G.sep:
			theta_eps = self.theta
			for i in range(self.iter_fixed_point):  # 7 itérations mais dans l'article ils parlent de 6 ou 7,
				# ce paramètre est à faire varier
				theta_eps = self.theta + epsilon / 2.0 * np.dot(
					self.G.value_inv(self.theta) + self.G.value_inv(theta_eps), p)
		else:
			theta_eps = self.theta + epsilon * np.dot(self.G.value_inv(self.theta), p)

		self.theta = theta_eps  #TODO:Baptiste tu valides qu'il faille updater l'attribut theta
		return theta_eps


class PDynamic:
	def __init__(self, grad_theta_H, p, iter_fixed_point=7):
		self.grad_theta_H = grad_theta_H
		self.p = p
		self.p_half_leap = None
		self.iter_fixed_point = iter_fixed_point

	def half_leap_update(self, theta, epsilon):
		"""
		Computes p(\tau+\epsilon/2) based on previous values of \theta(\tau), p(\tau)

		:param theta:
		:param epsilon: leapfrog step
		:return:
		"""
		p_half_leap = self.p.copy()
		p_previous = self.p.copy()
		# if not self.G.sep:

		for i in range(self.iter_fixed_point):
			grad = self.grad_theta_H(theta, p_half_leap)
			p_half_leap = p_previous - epsilon/2.0*grad

		self.p_half_leap = p_half_leap
		return p_half_leap

	def update(self, theta, epsilon):
		"""
		Computes p(\tau+\epsilon) based on values of \theta(\tau+\epsilon), p(\tau+\epsilon/2)
		The half leap value is stored as an attribute and thus does not need to be fed to this
		function

		:param theta: \theta(\tau+\epsilon)
		:param epsilon: leapfrog step
		:return:
		"""
		grad = self.grad_theta_H(theta, self.p_half_leap)
		p_epsilon = self.p_half_leap - epsilon/2.0*grad
		self.p = p_epsilon
		return p_epsilon


