# Generate multi dimensional gaussian data
import numpy as np
import matplotlib.pyplot as plt

class data:
	def __init__(self, n_dim, n_clusters, means, convariance):
		# Dimensions of the data
		self.n_dim = n_dim
		# Number of Gaussian Clusters
		self.n_clusters = n_clusters
		# Means of the corresponding clusters
		# List of n_dim numpy arrays
		self.means = means
		# Convariance matrices of each of the cluster
		# List of n_dim*n_dim numpy arrays
		self.convariance = convariance

		# initializing data variable
		self.data = None

	def generate_one_cluster(self, n, M, C):
		# Generate data for a given cluster with mean M and convariance C
		# n = number of data points for specific cluster
		data_one_cluster = np.random.multivariate_normal(M, C, n)
		return data_one_cluster

	def generate(self, n_points, proportions):
		data_list = []

		n_cluster_points = [int(n_points*p) for p in proportions]
		for n, M, C in zip(n_cluster_points, self.means, self.convariance):
			data_one_cluster = self.generate_one_cluster(n, M, C)
			data_list.append(data_one_cluster)

		self.data = np.concatenate(data_list, axis = 0)

if __name__ == "__main__" :

	# Sanity Checks
	gausian_1d = data(1, 2, [np.array([5,5]), np.array([0,0])], [np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])])
	gausian_1d.generate(100, [0.5, 0.5])

	print(gausian_1d.data)
	print(gausian_1d.data.shape)

	plt.figure(0)
	plt.title("Two One Dimensional Gaussians")
	plt.scatter(gausian_1d.data[:50,0], gausian_1d.data[:50,1])
	plt.scatter(gausian_1d.data[50:,0], gausian_1d.data[50:,1])
	plt.show()
