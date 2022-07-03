__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GatedGraphConv, GINConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset

class GCN(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 output_dim,
                 dropout: float = 0.5,
                 last_layer_fully_adjacent=False,
                 layer_type="GCN",
                 ):
        super(GCN, self).__init__()
        self.layer_type = layer_type
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        num_features = [input_dim] + hidden_layers + [output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        #print(num_features)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU()))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        batch_size = len(ptr) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        # assign values to each graph in batch
        x = global_mean_pool(x, batch)
        x = x.view(-1)
        return x
