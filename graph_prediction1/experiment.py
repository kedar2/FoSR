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
    "loss_fn": torch.nn.L1Loss(),
    "display": True,
    "model": GCN,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 5,
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
    "rewired": False,
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
        self.train_data = self.args.train_data
        self.validation_data = self.args.validation_data
        self.test_data = self.args.test_data
        self.train_fraction = self.args.train_fraction
        self.validation_fraction = self.args.validation_fraction
        self.test_fraction = self.args.test_fraction
        self.stopping_criterion = self.args.stopping_criterion
        self.stopping_threshold = self.args.stopping_threshold
        self.patience = self.args.patience
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.hidden_layers = self.args.hidden_layers
        self.layer_type = self.args.layer_type
        self.rewired = self.args.rewired

        if self.hidden_layers is None:
            self.hidden_layers = [self.hidden_dim] * self.num_layers

        if self.input_dim is None:
            self.input_dim = self.dataset[0].x.shape[1]

        self.model = GCN(input_dim=self.input_dim, output_dim=1, hidden_layers=self.hidden_layers, dropout=self.dropout, layer_type=self.layer_type, rewired=self.rewired).to(self.device)

        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_data is None:
            dataset_size = len(self.dataset)
            train_size = int(self.train_fraction * dataset_size)
            validation_size = int(self.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_data, self.validation_data, self.test_data = random_split(self.dataset,[train_size, validation_size, test_size])
        elif self.validation_data is None:
            train_size = int(self.train_fraction * len(self.train_data))
            validation_size = len(self.train_data) - train_size
            self.train_data, self.validation_data = random_split(self.train_data, [train_size, validation_size])
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)

        if self.display:
            print("Starting training")
        best_validation_loss = inf
        best_train_loss = inf
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.train_data)

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
		
        #validation_loss = self.eval(validation_loader)
        #input()

        for epoch in range(self.max_epochs):
            self.model.train()            
            total_loss = 0
            sample_size = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.device)
                y = graph.y.float().to(self.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''

            if epoch % self.eval_every == 0:
                train_loss = self.eval(train_loader)
                validation_loss = self.eval(validation_loader)
                test_loss = self.eval(test_loader)
                scheduler.step(train_loss)

                if self.stopping_criterion == "train":
                    if train_loss < best_train_loss / self.stopping_threshold:
                        best_train_loss = train_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                        epochs_no_improve = 0
                        new_best_str = ' (new best train)'
                    else:
                        epochs_no_improve += 1
                elif self.stopping_criterion == 'validation':
                    if validation_loss < best_validation_loss / self.stopping_threshold:
                        best_train_loss = train_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                        epochs_no_improve = 0
                        new_best_str = ' (new best validation)'
                    else:
                        epochs_no_improve += 1
                if self.display:
                    print(f'Epoch {epoch}, Train loss: {train_loss}, Validation loss: {validation_loss}{new_best_str}, Test loss: {test_loss}')
                if epochs_no_improve > self.patience:
                    if self.display:
                        print(f'{self.patience} epochs without improvement, stopping training')
                        print(f'Best train loss: {best_train_loss}, Best validation loss: {best_validation_loss}, Best test loss: {best_test_loss}')
                    return train_loss, validation_loss, test_loss

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_loss = 0
            for graph in loader:
                graph = graph.to(self.device)
                y = graph.y.float().to(self.device)
                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y) * (len(graph.ptr) - 1)
                total_loss += loss
                
        return total_loss / sample_size