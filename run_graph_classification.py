from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.graph_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl

largest_cc = LargestConnectedComponents()
to_undirected = ToUndirected()

mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
collab = list(TUDataset(root="data", name="COLLAB"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
#datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}
datasets = {"proteins": proteins, "collab": collab}
for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))


def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": False,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "fosr",
    "num_iterations": 10,
    "patience": 100,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001
    })

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2})
}

def run(args=AttrDict({})):
    results = []
    args = default_args + args
    args += get_args_from_input()
    for key in datasets:
        args += hyperparams[key]
        validation_accuracies = []
        test_accuracies = []
        print(f"TESTING: {key} ({args.rewiring})")
        dataset = datasets[key]
        if args.rewiring == "fosr":
            for i in range(len(dataset)):
                edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
        elif args.rewiring == "sdrf":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True)
        elif args.rewiring == "digl":
            for i in range(len(dataset)):
                dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
                m = dataset[i].edge_index.shape[1]
                dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
        for trial in range(args.num_trials):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            validation_accuracies.append(validation_acc)
            test_accuracies.append(test_acc)
        val_mean = 100 * np.mean(validation_accuracies)
        test_mean = 100 * np.mean(test_accuracies)
        val_ci = 100 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
        test_ci = 100 * np.std(test_accuracies)/(args.num_trials ** 0.5)
        log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
        log_to_file(f"average acc: {test_mean}\n")
        log_to_file(f"plus/minus:  {test_ci}\n\n")
        results.append({
            "dataset": key,
            "rewiring": args.rewiring,
            "layer_type": args.layer_type,
            "num_iterations": args.num_iterations,
            "alpha": args.alpha,
            "eps": args.eps,
            "test_mean": test_mean,
            "test_ci": test_ci,
            "val_mean": val_mean,
            "val_ci": val_ci
            })
    df = pd.DataFrame(results)
    with open('results/graph_classification.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)
if __name__ == '__main__':
    run()
