import torch
import torch.utils.data

class GraphGenLSTMGenerator(torch.nn.Module):
    def __init__(self, num_nodes: int, activation=torch.nn.PReLU, layers=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = torch.nn.Sequential()
        self.hidden_size = self.num_nodes
        self.lstm = torch.nn.LSTM(input_size=self.num_nodes**2 + 1, hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        self.layers.append(torch.nn.Linear(self.hidden_size, self.num_nodes**2))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(self.num_nodes**2, self.num_nodes**2))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        batch_size = noise.size(0)
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        out = []
        input = torch.cat((torch.zeros(batch_size, self.num_nodes**2), noise), dim=1)
        for _ in range(self.num_nodes - 1):
            pred, (h, c) = self.lstm(input.unsqueeze(0), (h,c))
            pred = self.layers(pred.squeeze(0))
            out.append(pred)
            input = torch.concat((pred, noise), dim=1)
        return torch.concat(out, dim=-1)

class GraphGenLSTMDescriminator(torch.nn.Module):
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
