from attrdict import AttrDict
from experiment import Experiment
import torch
import numpy as np
from torch_geometric.datasets import ZINC, QM9
from torch.nn.functional import one_hot

active = True

zinc = ZINC(root='data')
zinc.data.x = one_hot(zinc.data.x).view(-1, 28)

qm9 = QM9(root='data')
attributes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
attribute_names = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]
datasets = {'zinc': zinc ,'qm9': qm9}

def log_to_file(message, filename="qm9_results.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

if active:

    names = ["qm9"]
    hyperparams = {
    "qm9": AttrDict({"dropout": 0.2, "num_layers": 3, "dim": 128, "learning_rate": 0.001})
    }

    num_trials=5
    for i in attributes:
        name = attribute_names[i]
        total_error = 0
        print(f"TESTING: {name} (GCN)")
        for trial in range(num_trials):
            print(f"TRIAL {trial+1}")
            qm9 = QM9(root='data')
            qm9.data.y = qm9.data.y[:,i]
            # only use the current attribute in training

            args = AttrDict({"data": qm9, "layer_type": "GCN", "display": True})
            args += hyperparams["qm9"]
            train_acc, validation_acc, test_acc = Experiment(args).run()
            accuracies.append(test_acc.item())
            torch.cuda.empty_cache()
        log_to_file(f"RESULTS FOR {name} (GCN):\n")
        log_to_file(f"average acc: {np.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(num_trials ** 0.5)}\n\n")
    
