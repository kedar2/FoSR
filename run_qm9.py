from attrdict import AttrDict
from experiments.graph_regression import Experiment
import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import ZINC, QM9
from torch.nn.functional import one_hot
from hyperparams import get_args_from_input

zinc = ZINC(root='data')
zinc.data.x = one_hot(zinc.data.x).view(-1, 28)

qm9 = QM9(root='data') 
attributes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
attribute_names = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]

def log_to_file(message, filename="qm9_results.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 8,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 5,
    "eval_every": 1,
    "patience": 5,
    "dataset": "QM9",
    "rewiring": None
    })

def run(args=AttrDict({})):
    results = []
    args = default_args + args
    args += get_args_from_input()
    for i in attributes:
        name = attribute_names[i]
        accuracies = []
        print(f"TESTING: {name} ({default_args.rewiring})")
        for trial in range(args.num_trials):
            print(f"TRIAL {trial+1}")
            qm9 = QM9(root='data')
            qm9.data.y = qm9.data.y[:,i]
            default_args.attribute_name = attribute_names[i]
            # only use the current attribute in training
            
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=qm9).run()
            result_dict = {"train_acc": train_acc, "validation_acc": validation_acc, "test_acc": test_acc, "attribute": name}
            results.append(args + result_dict)
            accuracies.append(test_acc.item())

        log_to_file(f"RESULTS FOR {name} ({default_args.rewiring}):\n")
        log_to_file(f"average acc: {np.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
        results_df = pd.DataFrame(results)
        results_df.to_csv('qm9_results.csv', mode='a')

if __name__ == '__main__':
    run()