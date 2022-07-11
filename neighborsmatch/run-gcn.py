from attrdict import AttrDict
from experiment import Experiment
import torch
import numpy as np
import task
import rewiring
from torch.nn.functional import one_hot

active = True

G = task.path_of_cliques(3, 10)
vertices_to_label = list(range(0, 9))
nmatch = task.create_neighborsmatch_dataset(G, 29, vertices_to_label, 10000)

def log_to_file(message, filename="neighborsmatch2.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

if active:

    hyperparams = {
    "neighborsmatch": AttrDict({"dropout": 0.0, "num_layers": 6, "hidden_dim": 64, "learning_rate": 0.001})
    }

    num_trials=1
    name = "neighborsmatch"
    accuracies = []
    print(f"TESTING: {name} (GCN)")
    for trial in range(num_trials):

        args = AttrDict({"dataset": nmatch, "layer_type": "GAT", "display": True})
        args += hyperparams["neighborsmatch"]
        train_acc, validation_acc, test_acc = Experiment(args).run()
        accuracies.append(train_acc.item())
        torch.cuda.empty_cache()
        log_to_file(f"RESULTS FOR {name} (GCN):\n")
        log_to_file(f"average acc: {np.mean(accuracies)}\n")
        log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(num_trials ** 0.5)}\n\n")
    
