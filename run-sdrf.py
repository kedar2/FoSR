from attrdict import AttrDict
from experiment import Experiment
import rewiring
import torch
import numpy as np
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_undirected
from copy import deepcopy
from sdrf import sdrf

cornell = WebKB(root="data", name="Cornell")[0]
wisconsin = WebKB(root="data", name="Wisconsin")[0]
texas = WebKB(root="data", name="Texas")[0]
chameleon = WikipediaNetwork(root="data", name="chameleon")[0]
squirrel = WikipediaNetwork(root="data", name="squirrel")[0]
actor = Actor(root="data")[0]
cora = Planetoid(root="data", name="cora")[0]
citeseer = Planetoid(root="data", name="citeseer")[0]
pubmed = Planetoid(root="data", name="pubmed")[0]
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}



if __name__ == '__main__':

    names = ["cornell", "texas", "wisconsin"]
    hyperparams = {
    "cornell": AttrDict({"dropout": 0.3060, "num_layers": 1, "dim": 128, "learning_rate": 0.00082, "weight_decay": 0.1570}),
    "texas": AttrDict({"dropout": 0.5, "num_layers": 1, "dim": 128, "learning_rate": 0.000072, "weight_decay": 0.0037}),
    "wisconsin": AttrDict({"dropout": 0.5, "num_layers": 1, "dim": 128, "learning_rate": 0.000281, "weight_decay": 0.1570}),
    "chameleon": AttrDict({"dropout": 0.7304, "num_layers": 1, "dim": 128, "learning_rate": 0.0248, "weight_decay": 0.0936}),
    "squirrel": AttrDict({"dropout": 0.5974, "num_layers": 6, "dim": 64, "learning_rate": 0.0136, "weight_decay": 0.1346}),
    "actor": AttrDict({"dropout": 0.7605, "num_layers": 1, "dim": 64, "learning_rate": 0.0290, "weight_decay": 0.0619}),
    "cora": AttrDict({"dropout": 0.4144, "num_layers": 1, "dim": 64, "learning_rate": 0.0097, "weight_decay": 0.0639}),
    "citeseer": AttrDict({"dropout": 0.7477, "num_layers": 1, "dim": 128, "learning_rate": 0.0251, "weight_decay": 0.4577}),
    "pubmed": AttrDict({"dropout": 0.4013, "num_layers": 1, "dim": 64, "learning_rate": 0.0095, "weight_decay": 0.0448})
    }

    num_trials=1000
    for name in names:
        accuracies = []
        data = datasets[name]
        data.edge_index = to_undirected(data.edge_index)
        original_edge_index = data.edge_index
        print("TESTING: " + name)
        for trial in range(num_trials):
            data.edge_index = original_edge_index
            data.edge_index = sdrf(data, loops=50, is_undirected=True).edge_index
            args = AttrDict({"data": data, "display": False})
            args += hyperparams[name]
            train_acc, validation_acc, test_acc = Experiment(args).run()
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("plus/minus: ", 2 * np.std(accuracies)/(num_trials ** 0.5))
    
