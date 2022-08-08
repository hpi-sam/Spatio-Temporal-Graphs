import torch
import torch.utils.data
from utils import decode_graphs

class GraphGenLinearGenerator(torch.nn.Module):
    def __init__(self, num_nodes: int, temporal: int=0, activation=torch.nn.PReLU, layers=None):
        super().__init__()
        self.temporal = temporal
        self.num_nodes = num_nodes
        self.layers = torch.nn.Sequential()
        if layers is None:
            layer_size = 1
            while layer_size*2< num_nodes*num_nodes:
                self.layers.append(torch.nn.Linear(layer_size, 2*layer_size))
                layer_size *= 2
                self.layers.append(activation())
            self.layers.append(torch.nn.Linear(layer_size, num_nodes*num_nodes))
        else:
            if len(layers) == 0:
                last_layer = 1
            else:
                last_layer = layers[0]
                self.layers.append(torch.nn.Linear(1, last_layer))
                self.layers.append(activation())
                for layer in layers[1:]:
                    self.layers.append(torch.nn.Linear(last_layer, layer))
                    self.layers.append(activation())
                    last_layer = layer
            if self.temporal == 0:
                self.layers.append(torch.nn.Linear(last_layer, (num_nodes**2)))
            else:
                self.layers.append(torch.nn.Linear(last_layer, (num_nodes**2)*temporal))
        self.layers.append(torch.nn.Sigmoid())
    
    def forward(self, X) -> torch.Tensor:
        return self.layers(X)

class GraphGenLinearDescriminator(torch.nn.Module):
    def __init__(self, num_nodes: int, temporal: int=0, activation=torch.nn.PReLU, layers=None):
        super().__init__()
        self.layers = torch.nn.Sequential()
        layer_size = num_nodes**2
        if temporal > 0:
            layer_size *= temporal
        self.layers.append(torch.nn.Flatten())
        if layers is None:
            while int(layer_size / 2)  > 1:
                self.layers.append(torch.nn.Linear(layer_size, int(layer_size / 2)))
                layer_size = int(layer_size / 2)
                self.layers.append(activation())
                last_layer = layer_size
        else:
            if len(layers) == 0:
                last_layer = layer_size
            else:
                last_layer = layers[0]
                self.layers.append(torch.nn.Linear(layer_size, last_layer))
                self.layers.append(activation())
                for layer in layers[1:]:
                    self.layers.append(torch.nn.Linear(last_layer, layer))
                    self.layers.append(activation())
                    last_layer = layer
        self.layers.append(torch.nn.Linear(last_layer, 1))

    def forward(self, X) -> torch.Tensor:
        return self.layers(X)
