from torch_geometric.utils import to_networkx
from numba import jit, int64
import networkx as nx
import numpy as np
from math import inf

@jit(nopython=True)
def test(x):
	z = np.zeros(3, dtype=int64)
	y = np.append(z, z)
	return y

@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
	# returns array of degrees of all nodes
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees

@jit(nopython=True)
def choose_edge_to_add(x, edge_index):
	# chooses edge (u, v) to add which minimizes x[u]*x[v]
	n = x.size
	m = edge_index.shape[1]
	d = compute_degrees(edge_index, num_nodes=n)
	differentials = np.zeros((n, n))
	for u in range(n):
		for v in range(u):
			increment = 2 * x[u] * x[v] / ((1 + d[u])**0.5 * (1 + d[v]) ** 0.5)
			differentials[u, v] += increment
			differentials[v, u] += increment
			for I in range(m):
				i = edge_index[0, I]
				j = edge_index[1, I]
				if i != u:
					increment = x[i] * x[v] / d[i]**0.5 * (1 / (1 + d[v])**0.5 - 1 / d[v]**0.5)
					differentials[u, v] += increment
					differentials[v, u] += increment
				if i != v:
					increment = x[i] * x[u] / d[i]**0.5 * (1 / (1 + d[u])**0.5 - 1 / d[u]**0.5)
					differentials[u, v] += increment
					differentials[v, u] += increment
	for I in range(m):
		i = edge_index[0, I]
		j = edge_index[1, I]
		differentials[i, j] = inf
	for i in range(n):
		differentials[i, i] = inf 
	smallest_differential = np.argmin(differentials)
	return (smallest_differential % n, smallest_differential // n)

@jit(nopython=True)
def add_edge(edge_index, u, v):
	new_edge = np.array([[u, v],[v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)

@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
	# given an edge_index, computes Ax, where A is the corresponding adjacency matrix
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y

@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	return 1 - np.linalg.norm(y)

@jit(nopython=True)
def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=5):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, edge_index)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		edge_type = np.append(edge_type, 1)
		edge_type = np.append(edge_type, 1)
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index, edge_type, x

def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	if edge_type is None:
		edge_type = np.zeros(m, dtype=np.int64)
	return _edge_rewire(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)