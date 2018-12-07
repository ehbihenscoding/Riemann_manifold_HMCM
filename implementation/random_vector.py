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

		:param p:
		:param epsilon:
		:return:
		"""
		if not self.G.sep:
			theta_eps = self.theta
			for i in range(self.iter_fixed_point):  # 7 itérations mais dans l'article ils parlent de 6 ou 7,
				# ce paramètre est à faire varier
				theta_eps = self.theta + epsilon / 2.0 * np.dot(
					self.G.value_inv(self.theta) + self.G.value_inv(theta_eps), p)
			return theta_eps
		else:
			return self.theta + epsilon * np.dot(self.G.value_inv(self.theta), p)
