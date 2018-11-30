class random_vector:
	def __init__(self,G,theta):
		self.G = G
		self.theta = theta

	def update(self, p):
		if G.sep():
			theta_eps = self.theta
			for i in range(7):	#7 itérations mais dans l'article ils parlent de 6 ou 7, ce paramètre est à faire varier
				theta_eps = self.theta + epsilon/2 * np.dot(G.value_inv(self.theta) + G.value_inv(theta_eps),p)
			return thera_eps
		else:
			return (self.theta + epsilon * np.dot(G.value_inv(self.theta),p))
