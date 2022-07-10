import numba
from numba import jit
import numpy as np
from math import inf

# need to fix a bug in this where the graph becomes disconnected

@jit(nopython=True)
def matmul(A, B):
	# multiplication of two integer matrices
	m = A.shape[0]
	n = A.shape[1]
	p = B.shape[1]
	C = np.zeros((m, p), dtype=numba.int64)
	for i in range(m):
		for j in range(p):
			for k in range(n):
				C[i][j] += A[i][k] * B[k][j]
	return C

@jit(nopython=True)
def softmax(weights):
	normalizing_factor = np.sum(np.exp(weights))
	return np.exp(weights) / normalizing_factor

@jit(nopython=True)
def sample(edge_index, weights, temperature=1, use_softmax=True):
	# samples randomly from the edge index of a graph given an nxn matrix of weights
	seed = np.random.random()
	if use_softmax:
		probabilities = softmax(temperature * weights)
	else:
		probabilities = weights / np.sum(weights)
	N = len(edge_index[0])
	for i in range(N):
		u = edge_index[0][i]
		v = edge_index[1][i]
		seed -= probabilities[u, v]
		if seed < 0:
			return u, v
	u = edge_index[0][N-1]
	v = edge_index[1][N-1]
	return u, v

@jit(nopython=True)
def argmin(values, mask=None):
	# masked argmin; selects the index i minimizing value[i] such that mask[i]=True
	min_index = -1
	min_value = inf
	for i in range(len(values)):
		if values[i] < min_value and mask[i]:
			min_value = values[i]
			min_index = i
	return min_index

@jit(nopython=True)
def flip(edge_index, u, v, i, j):
	# flip operation on a graph
	num_edges = len(edge_index[0])
	for index in range(num_edges):
		if edge_index[0][index] == u:
			edge_index[1][index] = j
		elif edge_index[0][index] == v:
			edge_index[1][index] = i
		elif edge_index[0][index] == i:
			edge_index[1][index] = v
		elif edge_index[0][index] == j:
			edge_index[1][index] = u
	return edge_index

@jit(nopython=True)
def grlef(edge_index, num_iterations=1, temperature=1):
	num_nodes = max(edge_index[0]) + 1
	num_edges = len(edge_index[0])
	adj_matrix = np.zeros((num_nodes, num_nodes), dtype=numba.int64)
	for i in range(num_edges):
		u = edge_index[0][i]
		v = edge_index[1][i]
		adj_matrix[u,v] += 1
	# adjacency matrix squared is used since (A^2)_{ij} = number of triangles based at edge ij
	adj_matrix_squared = adj_matrix ** 2
	for iteration in range(num_iterations):
		sample_weights = 1 / (2 + adj_matrix_squared)
		u, v = sample(edge_index, sample_weights, temperature=temperature)
		adj_u = adj_matrix[u]
		adj_v = adj_matrix[v]
		adj_u_but_not_v = adj_u - adj_u * adj_v
		adj_v_but_not_u = adj_v - adj_u * adj_v

		# define an array which denotes the change in triangles from switching edge iu to iv
		change_in_triangles = np.zeros(num_nodes)
		for i in range(num_nodes):
			change_in_triangles[i] = adj_matrix_squared[i, v] - adj_matrix_squared[i, u]

		# choose the value of i~u which causes the greatest reduction in triangles possible
		i = argmin(change_in_triangles, mask=(adj_u_but_not_v==1))
		if i != -1:
			# choose the value of j~v which causes the greatest reduction in triangles possible
			j = argmin(-change_in_triangles, mask=(adj_v_but_not_u==1))
			if j != -1:
				edge_index = flip(edge_index, u, v, i, j)
				adj_matrix[u][i] = 0
				adj_matrix[i][u] = 0
				adj_matrix[u][j] = 1
				adj_matrix[j][u] = 1
				adj_matrix[v][i] = 1
				adj_matrix[i][v] = 1
				adj_matrix[v][j] = 0
				adj_matrix[j][v] = 0
				
				change_matrix = np.zeros((num_nodes, num_nodes), dtype=numba.int64)
				change_matrix[u][i] = -1
				change_matrix[i][u] = -1
				change_matrix[u][j] = 1
				change_matrix[j][u] = 1
				change_matrix[v][i] = 1
				change_matrix[i][v] = 1
				change_matrix[v][j] = -1
				change_matrix[j][v] = -1
				adj_matrix_squared += (matmul(change_matrix, adj_matrix) + matmul(adj_matrix, change_matrix) + change_matrix ** 2)
				adj_matrix += change_matrix
	return edge_index, adj_matrix, adj_matrix_squared