from torch_geometric.utils import to_networkx
from numba import jit
import networkx as nx
import numpy as np
from math import inf


def choose_edge_to_add(x, A):
	min_prod = inf
	n = len(x)
	u = np.argmin(x)
	v = np.argmax(x)
	min_values = (u, v)
	if not A[u, v]:
		return (u, v)
	for i in range(n):
		for j in range(i):
			if not A[i,j] and x[i]*x[j] < min_prod:
				min_prod = x[i] * x[j]
				min_values = (i, j)
	return min_values

def edge_rewire(G, A=None, x=None, num_iterations=50, initial_power_iters=5):
	n = G.number_of_nodes()
	if x is None:
		x = 2 * np.random.random(n) - 1
	if A is None:
		A = np.array(nx.adjacency_matrix(G).todense())
	degrees = A.dot(np.ones(n))
	for i in range(initial_power_iters):
		x -= x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = A.dot(x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, A)
		A[i, j] = 1
		A[j, i] = 1
		G.add_edge(i, j)
		degrees[i] += 1
		degrees[j] += 1
		x -= x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = A.dot(x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return x, A