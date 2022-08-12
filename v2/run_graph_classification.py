from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.graph_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, robustness

largest_cc = LargestConnectedComponents()
to_undirected = ToUndirected()

mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
collab = list(TUDataset(root="data", name="COLLAB"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
#datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}
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
    "display": True,
    "num_trials": 30,
    "eval_every": 1,
    "rewiring": "sdrf",
    "num_iterations": 10,
    "num_relations": 2,
    "patience": 100,
    "output_dim": 2
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
        accuracies = []
        print(f"TESTING: {key} ({args.rewiring})")
        dataset = datasets[key]
        if args.rewiring == "edge_rewire":
            for i in range(len(dataset)):
                edge_index, edge_type, _ = robustness.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
        elif args.rewiring == "sdrf":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True)
        #print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
        for trial in range(args.num_trials):
            #print(f"TRIAL {trial+1}")
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            result_dict = {"train_acc": train_acc, "validation_acc": validation_acc, "test_acc": test_acc, "dataset": key}
            results.append(args + result_dict)
            accuracies.append(test_acc)
            print(test_acc)
        avg = 100 * np.mean(accuracies)
        ci = 100 * np.std(accuracies)/(args.num_trials ** 0.5)
        log_to_file(f"RESULTS FOR {key} ({default_args.rewiring}), {args.num_iterations} ITERATIONS:\n")
        log_to_file(f"average acc: {avg}\n")
        log_to_file(f"plus/minus:  {ci}\n\n")
        results.append({
            "dataset": key,
            "rewiring": args.rewiring,
            "num_iterations": args.num_iterations,
            "avg_accuracy": avg,
            "ci": ci
            })
    df = pd.DataFrame(results)
    df.to_csv('results/graph_classification.csv', mode='a')
if __name__ == '__main__':
    run()
