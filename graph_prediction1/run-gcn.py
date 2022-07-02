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

datasets = {'zinc': zinc ,'qm9': qm9}
qm9.data.y = qm9.data.y[:,7] # just using first property for now (mu)
#print(qm9[2000].y)
#input()

# normalize data

mu = torch.mean(qm9.data.y)
std = torch.std(qm9.data.y)

qm9.data.y -= mu
qm9.data.y /= std

print(mu, std)

if active:

    names = ["qm9"]
    hyperparams = {
    "qm9": AttrDict({"dropout": 0.2, "num_layers": 3, "dim": 128, "learning_rate": 0.001, "weight_decay": 0})
    }

    num_trials=20
    for name in names:
        accuracies = []
        print("TESTING: " + name)
        for trial in range(num_trials):
            data = datasets[name]
            #data.edge_index = torch.tensor([[], []]).long()
            args = AttrDict({"data": data, "layer_type": "GCN"})
            args += hyperparams[name]
            train_acc, validation_acc, test_acc = Experiment(args).run()
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("plus/minus: ", 2 * np.std(accuracies)/(num_trials ** 0.5))
    
