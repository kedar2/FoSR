import rewiring, rewiring_rlef
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import inf, log

from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx, from_networkx


qm9 = QM9(root='data')

def path_of_cliques(num_cliques, size_of_clique):
	G = nx.Graph([])
	for i in range(num_cliques):
		for j in range(size_of_clique):
			for k in range(j):
				G.add_edge(i*size_of_clique+j, i*size_of_clique+k)
		if i != num_cliques - 1:
			G.add_edge((i+1)*size_of_clique - 1, (i+1)*size_of_clique)
	return G

def dumbbell_graph(n, dumbbell_length):
	l = []
	for i in range(n):
		l.append((i, 2 * n - 1 + dumbbell_length + 2))
		for j in range(i):
			l.append((i, j))
	for i in range(n, 2*n):
		l.append((i, 2*n))
		for j in range(n, i):
			l.append((i,j))
	for i in range(2 * n, 2 * n - 1 + dumbbell_length + 2):
		l.append((i, i + 1))
	G = nx.Graph(l)
	return G


def ring_of_cliques(n, d):
	# ring of cliques graph with n vertices of degree d
	G = nx.Graph([])
	k = d + 1
	# encodes vertex by which clique it's in
	f = lambda x: (x % k, x // k)
	for x1 in range(n):
		for x2 in range(x1):
			(u1, v1) = f(x1)
			(u2, v2) = f(x2)
			if v1 == v2 and u1 - u2 != k - 1:
				G.add_edge(x1, x2)
			elif u2 - u1 == k - 1 and v1 - v2 == 1:
				G.add_edge(x1, x2)
	G.add_edge(n - 1, 0)
	return G

def tree(depth, b):
	# b-ary tree of a given depth
	num_nodes = (b ** depth - 1) // (b - 1)
	num_non_leaves = (b ** (depth - 1) - 1) // (b - 1)
	G = nx.Graph([])
	for i in range(num_non_leaves):
		for j in range(b):
			G.add_edge(i, b*i+j+1)
	return G

def plot_graph(spectral_values, triangle_values):
	font = {'size'   : 20}
	matplotlib.rc('font', **font)
	fig, ax1 = plt.subplots()
	x_values = list(range(len(spectral_values)))
	ax1.plot(x_values, spectral_values, color='C0')
	ax2 = ax1.twinx()
	ax2.plot(x_values, triangle_values, color='C1')
	fig.tight_layout()
	plt.savefig('filename.png', dpi=1200)
	plt.show()

rewiring_method = "rlef"
graph_type = "path_of_cliques"
depth = 8
b = 4
num_trials = 1
num_iterations = 1000


spectral_values = np.zeros(num_iterations)
triangle_values = np.zeros(num_iterations)

for trial in range(num_trials):
	triangle_data = None
	curvatures = None
	print(trial)
	if graph_type == "tree":
		G = tree(depth=3, b=6)
	elif graph_type == "dumbbell":
		G = dumbbell_graph(25, 1)
	elif graph_type == "ring_of_cliques":
		G = ring_of_cliques(250, 4)
	elif graph_type == "path_of_cliques":
		G = path_of_cliques(3, 10)
	elif graph_type == "qm9":
		graph_data = qm9[3000]
		print(graph_data.edge_index)
		input()
		G = to_networkx(graph_data, to_undirected=True)
		G = nx.Graph(G)
		largest_cc = max(nx.connected_components(G), key=len)
		#print(len(largest_cc))
		G = G.subgraph(largest_cc)
		G = nx.Graph(G)
	for i in range(num_iterations):
		#curvature = rewiring.average_curvature(G, curvatures=curvatures)
		if rewiring_method == "rlef":
			rewiring.rlef(G)
		elif rewiring_method == "grlef":
			triangle_data = rewiring.greedy_rlef_2(G, triangle_data=triangle_data)
		elif rewiring_method == "sdrf":
			#G = sdrf(G, loops=1, removal_bound=0.5)
			G, curvatures = rewiring.sdrf(G, curvatures=curvatures, C_plus=-inf)
		spectral_gap = rewiring.spectral_gap(G)
		num_triangles = rewiring.number_of_triangles(G)
		spectral_values[i] += spectral_gap
		triangle_values[i] += num_triangles
		print(i, spectral_gap, num_triangles)

spectral_values /= num_trials
triangle_values /= num_trials

#df = pd.DataFrame({"spectral values": spectral_values, "num triangles": triangle_values})
#df.to_csv("dumbbell-rlef.csv")

plot_graph(list(spectral_values), list(triangle_values))


#ax1.set_xlabel('Number of iterations')
#ax1.set_ylabel('Spectral Gap', color='tab:red')

#ax2.set_ylabel('Triangle count', color='tab:blue')

#plt.title("RLEF on Dumbbell Graph")
#plt.legend(loc='best')


