__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset

class SelfLoopGCNConv(torch.nn.Module):
    # R-GCN layer with two classes - the original edge list, and self-loops.
    def __init__(self, in_features, out_features):
        super(SelfLoopGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = GCNConv(in_features, out_features)
        self.layer2 = GCNConv(in_features, out_features)
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        all_nodes = torch.arange(num_nodes)
        only_self_loops = torch.stack([all_nodes, all_nodes]).to(self.device)
        return self.layer1(x, edge_index) + self.layer2(x, only_self_loops)

class GCN(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 output_dim,
                 dropout: float = 0.5,
                 last_layer_fully_adjacent=False,
                 layer_type="GCN",
                 num_relations=1
                 ):
        super(GCN, self).__init__()
        self.num_relations = num_relations
        self.layer_type = layer_type
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        num_features = [input_dim] + hidden_layers + [output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN" or self.layer_type == "Rewired-GCN-Sequential":
            return SelfLoopGCNConv(in_features, out_features)
        elif self.layer_type == "Rewired-GCN-Concurrent":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        batch_size = len(ptr) - 1
        for i, layer in enumerate(self.layers):
            if self.layer_type == "Rewired-GCN-Sequential":
                # rewired graphs treated in separate layers
                rewiring_mask = (graph.edge_attr == i)
                rewired_edge_index = edge_index[:,rewiring_mask]
                x = layer(x, rewired_edge_index)
            elif self.layer_type == "Rewired-GCN-Concurrent":
                x = layer(x, edge_index, graph.edge_attr)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        # assign values to each graph in batch
        x = global_mean_pool(x, batch)
        x = x.view(-1)
        return x
