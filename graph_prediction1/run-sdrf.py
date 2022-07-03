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

def log_to_file(message, filename="qm9_results.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

print("REWIRING...")
rewired1 = produce_rewired_dataset(QM9, 5)
rewired2 = produce_rewired_dataset(QM9, 10)
rewired3 = produce_rewired_dataset(QM9, 25)
print("REWIRED DATASET GENERATED")

if active:

    names = ["qm9"]
    hyperparams = {
    "qm9": AttrDict({"dropout": 0.2, "num_layers": 4, "dim": 128, "learning_rate": 0.001, "rewired": True})
    }

    num_trials=5
    for i in range(13):
        name = attribute_names[i]
        accuracies = []
        print(f"TESTING: {name} (SDRF)")
        for trial in range(num_trials):
            print(f"TRIAL {trial+1}")
            qm9 = QM9(root='data')
            qm9.data.y = qm9.data.y[:,i]
            qm9.rewired1 = rewired1
            qm9.rewired2 = rewired2
            qm9.rewired3 = rewired3
            # only use the current attribute in training

            args = AttrDict({"data": qm9, "layer_type": "GCN", "display": False})
            args += hyperparams["qm9"]
            train_acc, validation_acc, test_acc = Experiment(args).run()
            accuracies.append(test_acc.item())
            torch.cuda.empty_cache()
        log_to_file(f"RESULTS FOR {name} (SDRF):\n")
        log_to_file(f"average acc: {torch.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * torch.std(accuracies)/(num_trials ** 0.5)}\n\n")
    
