from attrdict import AttrDict
from experiment import Experiment
import torch
import numpy as np
import rewiring
from torch_geometric.datasets import ZINC, QM9
from torch.nn.functional import one_hot
from torch_geometric.utils import to_networkx, from_networkx

active = True

attribute_names = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]


def produce_rewired_dataset(dataset_source, num_iterations):
    dset = dataset_source(root='data')
    n = len(dset)
    for i in range(n):
        G = to_networkx(dset[i], to_undirected=True)
        for j in range(num_iterations):
            rewiring.sdrf(G)
        dset[i].edge_index = from_networkx(G).edge_index
    return dset.data.edge_index

def produce_labeled_graph(rewirings):
    # takes a set of rewirings of a dataset and merges them into a single graph where each rewiring has its own labeled edges
    edge_index = torch.tensor([[],[]])
    edge_attr = torch.tensor([])
    for i, rewiring in enumerate(rewirings):
        num_edges_in_rewiring = rewiring.size(1)
        current_edge_attr = torch.full((num_edges_in_rewiring,), i)
        edge_attr = torch.concat([edge_attr, current_edge_attr])
        edge_index = torch.concat([edge_index, rewiring], dim=1)
    return edge_index.long(), edge_attr.long()

def log_to_file(message, filename="qm9_results2.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

print("REWIRING...")

rewired0 = QM9(root='data').data.edge_index
rewired1 = produce_rewired_dataset(QM9, 10)
rewired2 = produce_rewired_dataset(QM9, 25)
rewired3 = produce_rewired_dataset(QM9, 50)
rewirings = [rewired0, rewired1, rewired2, rewired3]


print("REWIRED DATASET GENERATED")

if active:

    names = ["qm9"]
    hyperparams = {
    "qm9": AttrDict({"dropout": 0.2, "num_layers": 4, "hidden_dim": 64, "learning_rate": 0.001, "rewired": False})
    }

    num_trials=5
    for i in range(13):
        name = attribute_names[i]
        accuracies = []
        print(f"TESTING: {name} (GRLEF)")
        for trial in range(num_trials):
            print(f"TRIAL {trial+1}")
            qm9 = QM9(root='data')
            qm9.data.edge_index, qm9.data.edge_attr = produce_labeled_graph(rewirings)
            qm9.data.y = qm9.data.y[:,i]
            # only use the current attribute in training
            args = AttrDict({"dataset": qm9, "layer_type": "Rewired-GCN-Concurrent", "display": True, "num_relations": len(rewirings)})
            args += hyperparams["qm9"]
            train_acc, validation_acc, test_acc = Experiment(args).run()
            accuracies.append(test_acc.item())
            torch.cuda.empty_cache()
        log_to_file(f"RESULTS FOR {name} (GRLEF):\n")
        log_to_file(f"average acc: {np.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(num_trials ** 0.5)}\n\n")
    
