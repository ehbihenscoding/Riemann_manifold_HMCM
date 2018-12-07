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
	def __init__(self, G: MetricTensor, p, iter_fixed_point=7):
		self.G = G
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
		if not self.G.sep:
			G_inv = self.G.value_inv(theta)
			grad_part = 0.5*(G_inv + G_inv.T)  #Baptiste: Ginv(\theta) est elle sym?
			for i in range(self.iter_fixed_point):
				p_half_leap = p_previous - epsilon/2.0*np.dot(grad_part, p_half_leap)
		else:
			#TODO:Baptiste je me rappelle plus de notre formule dans ce cas, ca te va si on
			#TODO:voit ca ensemble la prochaine fois?
			raise ValueError

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
		G_inv = self.G.value_inv(theta)
		grad_part = 0.5 * (G_inv + G_inv.T)  # Baptiste: Ginv(\theta) est elle sym?
		p_epsilon = self.p_half_leap - epsilon/2.0*np.dot(grad_part, self.p_half_leap)

		self.p = p_epsilon
		return p_epsilon


