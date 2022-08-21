import torch
import torch.utils.data

class GraphGenGanGenerator(torch.nn.Module):
    def __init__(self, num_nodes: int, activation=torch.nn.PReLU, layers=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(self.num_nodes**2 + 1, self.num_nodes**2))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(self.num_nodes**2, self.num_nodes**2))
        self.layers.append(torch.nn.Sigmoid())
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        out = [noise.new_zeros(noise.size(0), self.num_nodes, self.num_nodes)]
        for _ in range(self.num_nodes - 1):
            next = torch.concat([torch.flatten(out[-1], start_dim=1), noise], dim=1)
            out.append(self.layers(next))
        return torch.concat(out[1:], dim=1)

class GraphGenGanDescriminator(torch.nn.Module):
    def __init__(self, num_nodes: int, temporal: int=0, activation=torch.nn.Tanh, layers=None):
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
