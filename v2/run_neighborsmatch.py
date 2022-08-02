from attrdict import AttrDict
from torch_geometric.utils import to_networkx, from_networkx
from experiments.nmatch_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, robustness, nmatch

def log_to_file(message, filename="results/neighborsmatch_results.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

def produce_rewired_dataset(dataset_source, num_iterations):
    dset = dataset_source
    n = len(dset)
    for i in range(n):
        edge_index = np.array(dset[i].edge_index)
        G = to_networkx(dset[i], to_undirected=True)
        for j in range(num_iterations):
            rewiring.greedy_rlef_2(G)
        dset[i].edge_index = from_networkx(G).edge_index
    return dset

default_args = AttrDict({
    "dropout": 0.0,
    "num_layers": 6,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "GAT",
    "display": True,
    "num_trials": 375,
    "eval_every": 1,
    "rewiring": "GRLEF",
    "num_iterations": 50
    })

def run(args=AttrDict({})):
    vertices_to_label = list(range(0, 9))
    iteration_counts = list(range(50, 1050, 50))
    

    results = []
    args = default_args + args
    args += get_args_from_input()

    accuracies = []
    print(f"TESTING: NEIGHBORSMATCH ({args.rewiring}), ITERATION COUNT {args.num_iterations}")

    for trial in range(1, args.num_trials + 1):
        G = nmatch.path_of_cliques(3, 10)
        if args.rewiring == "GRLEF":
            for i in range(args.num_iterations):
                rewiring.greedy_rlef_2(G)
        dataset = nmatch.create_neighborsmatch_dataset(G, 29, vertices_to_label, 10000)
        print(f"TRIAL {trial}")
        train_acc = Experiment(args=args, dataset=dataset).run()
        result_dict = {"train_acc": train_acc, "validation_acc": validation_acc, "test_acc": test_acc, "dataset": key}
        results.append(args + result_dict)
        accuracies.append(test_acc)

    log_to_file(f"RESULTS FOR NEIGHBORSMATCH ({args.rewiring}), ITERATION COUNT {args.num_iterations}:\n")
    log_to_file(f"average acc: {100 * np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {200 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/neighborsmatch_results.csv', mode='a')

if __name__ == '__main__':
    run()