import torch
import numpy as np
from attrdict import AttrDict
from torch.utils.data import DataLoader
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
    "stopping_criterion": "validation",
    "stopping_threshold": 0.00001,
    "patience": 100,
    "data": None,
    "train_fraction": 0.6,
    "validation_fraction": 0.2,
    "test_fraction": 0.2,
    "train_samples": None,
    "val_samples": None,
    "test_samples": None,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": None,
    "hidden_layers": None,
    "num_layers": 1,
    "pass_to_largest_cc": True
    }
    )

class Experiment:
    def __init__(self, args):
        self.args = default_args + args
        self.learning_rate = self.args.learning_rate
        self.dropout = self.args.dropout
        self.weight_decay = self.args.weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = self.args.max_epochs
        self.loss_fn = self.args.loss_fn  
        self.display = self.args.display
        self.eval_every = self.args.eval_every
        self.data = self.args.data
        self.num_nodes = len(self.data.y)
        self.num_layers = self.args.num_layers
        self.train_samples = self.args.train_samples
        self.val_samples = self.args.val_samples
        self.test_samples = self.args.test_samples
        self.train_fraction = self.args.train_fraction
        self.validation_fraction = self.args.validation_fraction
        self.test_fraction = self.args.test_fraction
        self.stopping_criterion = self.args.stopping_criterion
        self.stopping_threshold = self.args.stopping_threshold
        self.patience = self.args.patience
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.hidden_layers = self.args.hidden_layers
        self.output_dim = self.args.output_dim

        if self.hidden_layers is None:
            self.hidden_layers = [self.hidden_dim] * self.num_layers
        
        if self.output_dim is None:
            self.output_dim = max(self.data.y).item() + 1

        if self.input_dim is None:
            self.input_dim = self.data.x.shape[1]

        self.model = GCN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, dropout=self.dropout).to(self.device)

        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_samples is None:
            node_indices = list(range(self.num_nodes))
            self.test_fraction = 1 - self.train_fraction - self.validation_fraction
            non_test, self.test_samples = train_test_split(node_indices, test_size=self.test_fraction)
            self.train_samples, self.validation_samples = train_test_split(non_test, test_size=self.validation_fraction/(self.validation_fraction + self.train_fraction))
        elif self.validation_samples is None:
            non_test = [i for i in range(self.num_nodes) if not i in self.test_samples]
            self.train_samples, self.validation_samples = train_test_split(non_test, test_size=self.validation_fraction/(self.validation_fraction + self.train_fraction))

        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)

        if self.display:
            print("Starting training")
        best_validation_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        batch = self.data.to(self.device)
        train_size = len(self.train_samples)
		
        for epoch in range(self.max_epochs):
            self.model.train()            
            total_loss = 0
            num_examples = 0
            train_correct = 0
            optimizer.zero_grad()

            batch = self.data.to(self.device)
            y = self.data.y.to(self.device)
                
            out = self.model(batch)
            loss = self.loss_fn(input=out[self.train_samples], target=y[self.train_samples])
            num_examples += train_size
            total_loss += loss.item()
            _, train_pred = out[self.train_samples].max(dim=1)
            train_correct += train_pred.eq(y[self.train_samples]).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_training_loss = total_loss / num_examples
            train_acc = train_correct / num_examples
            scheduler.step(train_acc)
            new_best_str = ''

            if epoch % self.eval_every == 0:
                validation_acc = self.eval(sample="validation")
                test_acc = self.eval(sample="test")

                if self.stopping_criterion == "train":
                    if train_acc > best_train_acc + self.stopping_threshold:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        new_best_str = ' (new best train)'
                    else:
                        epochs_no_improve += 1
                elif self.stopping_criterion == 'validation':
                    if validation_acc > best_validation_acc + self.stopping_threshold:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        new_best_str = ' (new best validation)'
                    else:
                        epochs_no_improve += 1
                if self.display:
                    print(f'Epoch {epoch}, Train acc: {train_acc}, Validation acc: {validation_acc}{new_best_str}, Test acc: {test_acc}')

                if epochs_no_improve > self.patience:
                    if self.display:
                        print(f'{self.patience} epochs without improvement, stopping training')
                        print(f'Best train acc: {best_train_acc}, Best validation acc: {best_validation_acc}, Best test acc: {best_test_acc}')
                    return train_acc, validation_acc, test_acc

    def eval(self, sample="validation"):
        self.model.eval()
        with torch.no_grad():
            if sample == "validation":
                samples = self.validation_samples
            elif sample == "test":
                samples = self.test_samples
            else:
                return 0
            sample_size = len(samples)
            total_correct = 0
            batch = self.data.to(self.device)
            _, pred = self.model(batch)[samples].max(dim=1)
            total_correct += pred.eq(batch.y[samples]).sum().item()
            acc = total_correct / sample_size
            return acc