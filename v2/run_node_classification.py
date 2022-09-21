from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, robustness

largest_cc = LargestConnectedComponents()
to_undirected = ToUndirected()

cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
squirrel = WikipediaNetwork(root="data", name="squirrel")
actor = Actor(root="data")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
#datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
datasets = {"cornell": cornell, "wisconsin": wisconsin}

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": False,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "edge_rewire",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100
    })

def run(args=AttrDict({})):
    results = []
    args = default_args + args
    args += get_args_from_input()
    for key in datasets:
        accuracies = []
        print(f"TESTING: {key} ({default_args.rewiring})")
        dataset = datasets[key]
        dataset.data = to_undirected(dataset.data)
        dataset.data = largest_cc(dataset.data)
        if args.rewiring == "edge_rewire":
            print(dataset.data.edge_index)
            edge_index, edge_type, _ = robustness.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
            dataset.data.edge_index = torch.tensor(edge_index)
            dataset.data.edge_type = torch.tensor(edge_type)
            print(dataset.data.edge_index)
            input()
        elif args.rewiring == "sdrf":
            dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, is_undirected=True)
        #print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
        for trial in range(args.num_trials):
            #print(f"TRIAL {trial+1}")
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            accuracies.append(test_acc)

        log_to_file(f"RESULTS FOR {key} ({default_args.rewiring}):\n")
        log_to_file(f"average acc: {np.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
        results.append({
            "dataset": key,
            "rewiring": args.rewiring,
            "num_iterations": args.num_iterations,
            "avg_accuracy": np.mean(accuracies),
            "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5)
            })
        results_df = pd.DataFrame(results)
        with open('results/node_classification.csv', 'a') as f:
            results_df.to_csv(f, mode='a', header=f.tell()==0)
if __name__ == '__main__':
    run()