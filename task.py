import torch
import torch_geometric

from torch_geometric.data import Data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
#from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import homophily, to_undirected, to_networkx, subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected, remove_self_loops
import networkx as nx

class GraphDataset(object):
    
    cornell = WebKB(root="data", name="Cornell")[0]
    wisconsin = WebKB(root="data", name="Wisconsin")[0]
    texas = WebKB(root="data", name="Texas")[0]
    chameleon = WikipediaNetwork(root="data", name="chameleon")[0]
    squirrel = WikipediaNetwork(root="data", name="squirrel")[0]
    actor = Actor(root="data")[0]
    cora = Planetoid(root="data", name="cora")[0]
    citeseer = Planetoid(root="data", name="citeseer")[0]
    pubmed = Planetoid(root="data", name="pubmed")[0]
    dataset_names = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
    
    def __init__(self):
        super(GraphDataset, self).__init__()
        self.criterion = F.cross_entropy

    def generate_data(self, name):
        self.graph = self.dataset_names[name].clone().detach()
        self.graph.edge_index = to_undirected(self.graph.edge_index)
        self.pass_to_largest_cc()
        self.graph.edge_index = remove_self_loops(self.graph.edge_index)[0]
        self.out_dim = max(self.graph.y) + 1
        self.num_nodes = len(self.graph.y)
    
    def decide_training_set(self, train_fraction, validation_fraction):
        node_indices = list(range(self.num_nodes))
        test_fraction = 1 - train_fraction - validation_fraction
        non_test, test = train_test_split(node_indices, test_size=test_fraction)
        train, validation = train_test_split(non_test, test_size=validation_fraction/(validation_fraction + train_fraction))
        return train, validation, test

    def pass_to_largest_cc(self):
        G = to_networkx(self.graph, to_undirected=True)
        largest_cc_vertices = torch.tensor(list(max(nx.connected_components(G), key=len)))
        self.graph = self.graph.subgraph(largest_cc_vertices)
        return self.graph