import torch
import numpy as np
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from math import inf

from models.gcn_model import GCN

default_args = AttrDict(
    {"learning_rate": 0.005,
    "max_epochs": 1000000,
    "loss_fn": torch.nn.CrossEntropyLoss(),
    "display": True,
    "model": GCN,
    "eval_every": 1,
    "stopping_criterion": "train",
    "stopping_threshold": 1.01,
    "patience": 20,
    "dataset": None,
    "train_fraction": 0.9,
    "validation_fraction": 0.05,
    "test_fraction": 0.05,
    "train_data": None,
    "validation_data": None,
    "test_data": None,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "GCN",
    "sequential_rewiring": False,
    "concurrent_rewiring": True,
    "num_relations": 1
    }
    )

class Experiment:
    def __init__(self, args):
        self.args = default_args + args
        self.learning_rate = self.args.learning_rate
        self.batch_size = self.args.batch_size
        self.dropout = self.args.dropout
        self.weight_decay = self.args.weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = self.args.max_epochs
        self.loss_fn = self.args.loss_fn
        self.display = self.args.display
        self.eval_every = self.args.eval_every
        self.dataset = self.args.dataset
        self.num_layers = self.args.num_layers
        self.stopping_criterion = self.args.stopping_criterion
        self.stopping_threshold = self.args.stopping_threshold
        self.patience = self.args.patience
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.hidden_layers = self.args.hidden_layers
        self.layer_type = self.args.layer_type
        self.num_relations = self.args.num_relations

        if self.hidden_layers is None:
            self.hidden_layers = [self.hidden_dim] * self.num_layers

        if self.input_dim is None:
            self.input_dim = self.dataset[0].x.shape[1]

        output_dim = max([elt.y for elt in self.dataset]) + 1

        self.model = GCN(input_dim=self.input_dim,
            output_dim=output_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            layer_type=self.layer_type,
            num_relations=self.num_relations).to(self.device)
        
    def run(self):
        torch.manual_seed(123)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)

        if self.display:
            print("Starting training")
        best_train_loss = 0
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.dataset)

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epochs):
            self.model.train()            
            total_loss = 0
            sample_size = 0
            optimizer.zero_grad()

            for graph in train_loader:
                #print(graph.x, graph.y)
                #input()
                graph = graph.to(self.device)
                y = graph.y.to(self.device)
                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''

            if epoch % self.eval_every == 0:
                train_loss = self.eval(train_loader, view=True)
                scheduler.step(train_loss)

                if self.stopping_criterion == "train":
                    if train_loss > best_train_loss * self.stopping_threshold:
                        best_train_loss = train_loss
                        epochs_no_improve = 0
                        new_best_str = ' (new best train)'
                    else:
                        epochs_no_improve += 1
                if self.display:
                    print(f'Epoch {epoch}, Train loss: {train_loss} {new_best_str}')
                if epochs_no_improve > self.patience:
                    if self.display:
                        print(f'{self.patience} epochs without improvement, stopping training')
                        print(f'Best train loss: {best_train_loss}')
                    return train_loss

    def eval(self, loader, view=False):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.device)
                y = graph.y.to(self.device)
                out = self.model(graph)
                guess = torch.argmax(out, dim=1)
                acc = sum(y == guess)
                total_correct += acc
                
        return total_correct / sample_size