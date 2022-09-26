# Code to compose the class which contains the means and convariance matrices of the GMMs and the perform the corresponding EM procedure on them
from data import data
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
	def __init__(self, n_mixtures, n_dim, kmeans_inference, data):
		self.n_mixtures = n_mixtures
		self.n_dim = n_dim
		self.data = data

		# Initialize the gamma vectors 
		self.gamma = self.initialize_gamma(kmeans_inference)

		# Initialize the means and covariance matrices
		self.means = self.means_step()

		# Initialze the convariance matrices
		self.convariance = self.convariance_step()
		
		# Initialize the W vector
		self.W = self.W_step()

	def initialize_gamma(self, kmeans_inference):
		# Based on the values of KMeans hard clustering, 
		# let us initialize the GMM gammas
		gamma_list = []

		for i in kmeans_inference:
			if i == 1:
				gamma_list.append(np.array([[1, 0, 0]]))
			if i == 2:
				gamma_list.append(np.array([[0, 1, 0]]))
			if i == 3:
				gamma_list.append(np.array([[0, 0, 1]]))
		return np.concatenate(gamma_list, axis = 0)

	def W_step(self):
		# Updating the W parameter
		W_vector = np.zeros(self.n_mixtures)
		for i in range(self.n_mixtures):
			W_vector[i] = np.sum(self.gamma[:,i])/self.gamma.shape[0]

		return W_vector

	def means_step(self):
		# Funda : mu(ith cluster) = gamma[:, i].T * data (output 1*d) 
		means_list = []
		for i in range(self.n_mixtures):
			# print(self.gamma[:,i].T.shape, self.data.shape)
			mean_i = np.matmul(self.gamma[:,i].reshape(1,-1), self.data)/np.sum(self.gamma[:,i])
			means_list.append(mean_i)

		return np.concatenate(means_list, axis = 0) 

	def convariance_step(self):
		# Funda : (J.X).T*(X) [J.X is elementwise multiplication, 
		# followed by matrix multiplication with X]
		convariance_list = []

		for i in range(self.n_mixtures):
			mean_norm_data = self.data - self.means[i,:].reshape(1,-1)
			gamma_reshaped = np.concatenate([self.gamma[:, i].reshape(-1,1)
				, self.gamma[:, i].reshape(-1,1)], axis = 1)
			convariance_i_numerator = np.matmul(np.multiply(gamma_reshaped, mean_norm_data).T, mean_norm_data)

			convariance_i = convariance_i_numerator/np.sum(self.gamma[:,i])
			convariance_list.append(convariance_i.reshape(1,self.n_dim, self.n_dim))
		return np.concatenate(convariance_list, axis = 0)

	# def pdf_multivariate_gaussian(self, x, mu, sigma):
	# 	pdf = np.exp(-0.5 * np.matmul((x-mu), np.matmul(
	# 		np.linalg.pinv(sigma), (x-mu).T)))/np.sqrt(np.pow(2*np.pi, 
	# 		self.n_dim) * np.abs(np.linalg.det(sigma)) )
	# 	return pdf

	def gamma_step(self):
		# Update Gamma parameters in Maximization Step
		gamma_temp = np.zeros(self.gamma.shape)
		for i in range(self.gamma.shape[1]):
			for j in range(self.gamma.shape[0]):
				dist = multivariate_normal(self.means[i], self.convariance[i])
				gamma_temp[j,i] = self.W[i] * dist.pdf(self.data[j])

		gamma_temp = gamma_temp/np.sum(gamma_temp, axis = 1).reshape(-1,1)
		self.gamma = gamma_temp.copy()

	def EM(self, iters):
		# Code for the Expectation Maximization algorithm

		for i in range(iters):
			# M Step
			self.gamma_step()

			# E Step
			self.means = self.means_step()
			self.convariance = self.convariance_step()
			self.W = self.W_step()

			print("Covariance : {} {}".format(self.convariance[0], self.convariance[1]))
			print("Means : {} {}".format(self.means[0], self.means[1]))
			print("====================")
		

if __name__ == "__main__" :
	# Sanity Checks for the written GMM code
	n_mixtures = 3
	n_dim = 2
	kmeans_inference = []
	N = 1500
	p = [0.3, 0.3, 0.4]

	[kmeans_inference.append(1) for i in range(int(N* p[0]))]
	[kmeans_inference.append(2) for i in range(int(N* p[1]))]
	[kmeans_inference.append(3) for i in range(int(N* p[2]))]

	gausian_data = data(2, 3, [np.array([1,0.5]), np.array([0,0.5]), 
		np.array([-1,-1])], [np.array([[2,1],[1,1]]), 
		np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])])
	gausian_data.generate(N, p)

	gmm1 = GMM(n_mixtures, n_dim, kmeans_inference, gausian_data.data)

	for i in range(N):
		print(kmeans_inference[i], gmm1.gamma[i])

	print(gmm1.means)
	print(gmm1.convariance)
	print(gmm1.W)

	# gmm1.means = np.random.randn(3,2)
	# gmm1.convariance = np.concatenate([np.identity(2).reshape(1,2,2), 
	# 	np.identity(2).reshape(1,2,2), np.identity(2).reshape(1,2,2)], 
	# 	axis = 0)

	print("EM Start")
	gmm1.EM(4)



