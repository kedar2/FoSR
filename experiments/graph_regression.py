import torch
import numpy as np
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

from models.graph_model import GCN

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 5,
    "train_fraction": 0.9,
    "validation_fraction": 0.05,
    "test_fraction": 0.05,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "GCN",
    "num_relations": 1
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.L1Loss()

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]

        self.model = GCN(self.args).to(self.args.device)

        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
        elif self.validation_dataset is None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(self.args.train_data, [train_size, validation_size])
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        if self.args.display:
            print("Starting training")
        best_validation_loss = inf
        best_train_loss = inf
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.train_dataset)

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.max_epochs):
            self.model.train()
            total_loss = 0
            sample_size = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.float().to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''

            if epoch % self.args.eval_every == 0:
                train_loss = self.eval(train_loader)
                validation_loss = self.eval(validation_loader)
                test_loss = self.eval(test_loader)
                scheduler.step(train_loss)

                if self.args.stopping_criterion == "train":
                    if train_loss < best_train_loss / self.args.stopping_threshold:
                        best_train_loss = train_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                        epochs_no_improve = 0
                        new_best_str = ' (new best train)'
                    elif train_loss < best_train_loss:
                        best_train_loss = train_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_loss < best_validation_loss / self.args.stopping_threshold:
                        best_train_loss = train_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                        epochs_no_improve = 0
                        new_best_str = ' (new best validation)'
                    elif validation_loss < best_validation_loss:
                        best_train_loss = test_loss
                        best_validation_loss = validation_loss
                        best_test_loss = test_loss
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train loss: {train_loss}, Validation loss: {validation_loss}{new_best_str}, Test loss: {test_loss}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train loss: {best_train_loss}, Best validation loss: {best_validation_loss}, Best test loss: {best_test_loss}')
                    return train_loss, validation_loss, test_loss

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_loss = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.float().to(self.args.device)
                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y) * (len(graph.ptr) - 1)
                total_loss += loss
                
        return total_loss / sample_size