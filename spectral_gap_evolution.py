"""
Create plots of evolution of the spectral gap of a graph as it gets rewired.

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import rewiring, sdrf, fosr, digl
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from torch_geometric.datasets import TUDataset

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
collab = list(TUDataset(root="data", name="COLLAB"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}

for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))

max_iterations = 40
all_data = {}

for key in datasets:
    for rewiring_method in ["fosr", "sdrf"]:
        print(key, rewiring_method)
        dataset = datasets[key]
        spectral_gaps = [average_spectral_gap(dataset)]
        for j in range(max_iterations):
            spectral_gaps.append(average_spectral_gap(dataset))
            if rewiring_method == "sdrf":
                for i in range(len(dataset)):
                    # add an edge to each graph once
                    dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=1, remove_edges=False, is_undirected=True)   
            elif rewiring_method == "fosr":
                for i in range(len(dataset)):
                    edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=1)
                    dataset[i].edge_index = torch.tensor(edge_index)
                    dataset[i].edge_type = torch.tensor(edge_type)
        all_data[key + "/" + rewiring_method] = spectral_gaps
df = pd.DataFrame(all_data)
df.to_csv("results/spectral.csv")
    