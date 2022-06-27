import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from experiment import Experiment
from attrdict import AttrDict

training_data = MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(torch.nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.layers = torch.nn.ModuleList()
		self.layers.append(torch.nn.Linear(in_dim, 64))
		self.layers.append(torch.nn.ReLU())
		self.layers.append(torch.nn.Linear(64, 64))
		self.layers.append(torch.nn.ReLU())
		self.layers.append(torch.nn.Linear(64, out_dim))
	def forward(self, x):
		x = x.view(-1, 784)
		for layer in self.layers:
			x = layer(x)
		return x

N = NeuralNetwork(784, 10)

#D = DataLoader(training_data)

#x = next(iter(D)).float()

args = AttrDict({"model": N, "train_data": training_data, "val_data": test_data, "test_data": test_data})
Experiment(args).run()