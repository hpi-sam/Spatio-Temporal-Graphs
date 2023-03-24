from zipfile import BadZipFile
import torch
import torch.utils.data

class GraphGenLSTMGeneratorNodeSoftmax(torch.nn.Module):
    def __init__(self, num_nodes: int, activation=torch.nn.PReLU, hidden_size=None, layers=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = torch.nn.Sequential()
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            self.hidden_size = self.num_nodes
        self.lstm = torch.nn.LSTM(input_size=((self.num_nodes + 1) * 2), hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        self.layers.append(torch.nn.Linear(hidden_size, (self.num_nodes + 1) * 2))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear((self.num_nodes + 1) * 2, (self.num_nodes + 1) * 2))

    def forward(self, noise: torch.Tensor, num_predictions=None) -> torch.Tensor:
        if num_predictions is None:
            num_predictions = self.num_nodes
        batch_size = noise.size(0)
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        out = []
        input = noise
        for i in range(num_predictions):
            pred, (h, c) = self.lstm(input.unsqueeze(0), (h,c))
            pred = self.layers(pred.squeeze(0))
            pred[:,pred.shape[1] // 2 :] = torch.nn.functional.softmax(pred[:,pred.shape[1] // 2:], dim=1)
            pred[:,:pred.shape[1] // 2] = torch.nn.functional.softmax(pred[:,:pred.shape[1] // 2], dim=1)
            node = torch.multinomial(pred[:,:pred.shape[1] // 2], 1)
            nextnode = torch.multinomial(pred[:,pred.shape[1] // 2:], 1)
            pred[:,pred.shape[1] // 2 :] = torch.nn.functional.one_hot(node, pred.shape[1] // 2).squeeze().float()
            pred[:,:pred.shape[1] // 2] = torch.nn.functional.one_hot(nextnode, pred.shape[1] // 2).squeeze().float()
            out.append(pred)
            input = pred
        return torch.stack(out, dim=1)

    def sequence_forward(self, noise, sequence, num_predictions=None) -> torch.Tensor:
        if num_predictions is None:
            num_predictions = self.num_nodes
        batch_size = noise.size(0)
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        out = []
        input = noise
        for i in range(num_predictions):
            pred, (h, c) = self.lstm(input.unsqueeze(dim=0), (h,c))
            pred = self.layers(pred.squeeze())
            out.append(pred)
            input = sequence[:,i,:]
        return torch.concat(out, dim=-1)

    def raw_forward(self, input: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.lstm(input, (h,c))

class GraphGenLSTMDescriminatorSoftmax(torch.nn.Module):
    def __init__(self, num_nodes: int, temporal: int=0, activation=torch.nn.Tanh, layers=None):
        super().__init__()
        layer_size = (num_nodes + 1)*max(1, temporal)

        self.layers = torch.nn.Sequential()
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
