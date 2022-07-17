from datetime import datetime
import gzip
import os
from pathlib import Path
import time
from typing import List
from matplotlib import pyplot as plt
import matplotlib.animation as plta
import pandas
import teneto
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
from torchviz import make_dot

from correlation import decode_graphs

def save_model_graph(generator, discriminator, outdir="."):
    x = torch.normal(0, 1, size=(1,1))
    y_hat = generator(x)
    make_dot(y_hat, params=dict(list(generator.named_parameters()))).render(Path(outdir) / "generator_arch", format="png")
    y_hat = discriminator(y_hat.detach())
    make_dot(y_hat, params=dict(list(discriminator.named_parameters()))).render(Path(outdir) / "discriminator_arch", format="png")
    y_hat = discriminator(generator(x))
    make_dot(y_hat, params=dict(list(generator.named_parameters()) + list(discriminator.named_parameters()))).render(Path(outdir) / "complete_arch", format="png")


class GraphGen(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int=1,
        bidirectional: bool=False,
        dropout: float = 0.0,
        output_size: int=1,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.node = torch.nn.Linear(hidden_size, output_size)
        self.matrix_embedding = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=hidden_size
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Reshape X so that is a sequence of vectors
        original_shape = X.shape
        X = X.flatten(start_dim=1)
        output, (h_n, c_n) = self.lstm(X)
        output = torch.sigmoid(self.node(output))
        return output.reshape(*original_shape)

class GraphGenGanGenerator(torch.nn.Module):
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

class GraphGenGanDescriminator(torch.nn.Module):
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

def encode_graph(graph: nx.Graph):
    edge_dfs = list(nx.edge_dfs(graph))
    node_length = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    encoded = torch.zeros(num_edges + 2, node_length, node_length) # +2 for the origin and end encodings
    for index, edge in enumerate(edge_dfs):
        encoded[index + 1][edge] = 1
    encoded[0][edge_dfs[0][0], edge_dfs[0][0]] = 1
    encoded[-1,:,:] = 1
    return encoded

def load_graphs(dataset) -> torch.utils.data.DataLoader:
    with gzip.open(dataset, "rb") as file_handle:
        encoded = np.load(file_handle)
    decoded_graphs = decode_graphs(encoded)
    return [torch.tensor(nx.to_numpy_array(graph, dtype=np.float32)) for graph in decoded_graphs]
        

def trainGraphGen():
    outdir = Path("./output") / datetime.now().time().strftime("%H-%M-%S")
    dataset = "100k_frontend_graphs.npy"
    os.makedirs(outdir, exist_ok=False)
    num_nodes = 17
    num_training_steps = 50
    batch_size = 100
    step = 0
    discriminator = GraphGenGanDescriminator(num_nodes, activation=torch.nn.Tanh, layers=[128])
    generator = GraphGenGanGenerator(num_nodes, activation=torch.nn.Tanh, layers=[128])
    save_graph(generator, discriminator, outdir)
    print(generator)
    print(discriminator)
    save_model_graph(generator, discriminator, outdir)
    graphs: List[torch.Tensor] = load_graphs(dataset)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lr = 1e-4
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    with open(outdir/"setup.txt", "w") as write_handle:
        write_handle.write(repr(generator))
        write_handle.write(repr(discriminator))
        write_handle.write(f"\nBatch size: {batch_size}\n")
        write_handle.write(f"Learning rate: {lr}\n")
        write_handle.write(f"Dataset: {dataset}\n")
    log = {
        "d_generated": [],
        "d_real": [],
        "d_loss": [],
    }
    while step <= num_training_steps:
        bar = tqdm(loader)
        for graph_batch in bar:
            # Train discriminator
            discriminator.train()
            generator.train()
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device)
            generated: torch.Tensor = generator(random_data)
            d_generated = discriminator(generated.detach()).mean()
            d_real: torch.Tensor = discriminator(graph_batch).mean()
            d_loss: torch.Tensor = d_real - d_generated
            d_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device)
            generated: torch.Tensor = generator(random_data)
            d_generated: torch.Tensor = discriminator(generated).mean()
            d_generated.backward()
            generator_optimizer.step()
            log["d_generated"].append(d_generated.detach().cpu().item())
            log["d_loss"].append(d_loss.detach().cpu().item())
            log["d_real"].append(d_real.detach().cpu().item())
            bar.set_postfix({
                "d_generated": d_generated.detach().cpu().item(),
                "d_real": d_real.detach().cpu().item(),
                "d_loss": d_loss.detach().cpu().item(),
            })
        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        df.plot(xlabel="step", ylabel="value", title=f"One-Shot GAN Graph Generation")
        plt.savefig(outdir/"log.png")
        plt.close()
        torch.save(generator.state_dict(), outdir/"generator.ph")
        torch.save(discriminator.state_dict(), outdir/"discriminator.ph")
        save_graph(generator, discriminator, outdir)
        step += 1

def trainGraphGenTemporal():
    outdir = Path("./output") / datetime.now().time().strftime("%H-%M-%S")
    dataset = "100k_frontend_graphs-temporal.npz"
    os.makedirs(outdir, exist_ok=False)
    num_nodes = 17
    time_steps = 16
    evaluate_size = 1000
    num_training_steps = 50
    batch_size = 100
    step = 0
    discriminator = GraphGenGanDescriminator(num_nodes, temporal=time_steps, activation=torch.nn.Tanh, layers=[128])
    generator = GraphGenGanGenerator(num_nodes, temporal=time_steps, activation=torch.nn.Tanh, layers=[128])
    # save_temporal_graph(generator, discriminator, outdir)
    print(generator)
    print(discriminator)
    save_model_graph(generator, discriminator, outdir)
    t1 = time.perf_counter()
    if os.path.exists(dataset + ".torch"):
        graphs = torch.load(dataset + ".torch")
    else:
        graphs: List[torch.Tensor] = list(map(lambda x: torch.tensor(x, dtype=torch.float), np.load(dataset).values()))
        torch.save(torch.stack(graphs), dataset + ".torch")
    print(f"loaded data in {time.perf_counter() - t1:.2f}s")
   
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lr = 1e-4
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    with open(outdir/"setup.txt", "w") as write_handle:
        write_handle.write(repr(generator))
        write_handle.write(repr(discriminator))
        write_handle.write(f"\nBatch size: {batch_size}\n")
        write_handle.write(f"Learning rate: {lr}\n")
        write_handle.write(f"Dataset: {dataset}\n")
    log = {
        "d_generated": [],
        "d_real": [],
        "d_loss": [],
    }
    while step <= num_training_steps:
        bar = tqdm(loader)
        for graph_batch in bar:
            # Train discriminator
            discriminator.train()
            generator.train()
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device)
            generated: torch.Tensor = generator(random_data)
            d_generated = discriminator(generated.detach()).mean()
            d_real: torch.Tensor = discriminator(graph_batch).mean()
            d_loss: torch.Tensor = d_real - d_generated
            d_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device)
            generated: torch.Tensor = generator(random_data)
            d_generated: torch.Tensor = discriminator(generated).mean()
            d_generated.backward()
            generator_optimizer.step()
            log["d_generated"].append(d_generated.detach().cpu().item())
            log["d_loss"].append(d_loss.detach().cpu().item())
            log["d_real"].append(d_real.detach().cpu().item())
            bar.set_postfix({
                "d_generated": d_generated.detach().cpu().item(),
                "d_real": d_real.detach().cpu().item(),
                "d_loss": d_loss.detach().cpu().item(),
            })
        # Evaluate
        with torch.no_grad():
            random_data = torch.normal(0, 1, size=(evaluate_size, 1), device=device)
            generated = generator(random_data) > 0.5
            generated = generated.reshape(-1, num_nodes, num_nodes, time_steps).cpu().numpy()
            measures = []
            for index in range(generated.shape[0]):
                graph = generated[index]
                measures.append(teneto.networkmeasures.temporal_degree_centrality(teneto.TemporalNetwork(from_array=graph, nettype="bu"), calc="overtime"))
            mean = np.mean(measures, axis=0)
            real_mean = np.mean(np.load("tdc.npy"), axis=0)
            print(mean-real_mean)

        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        df.plot(xlabel="step", ylabel="value", title=f"One-Shot GAN Graph Generation")
        plt.savefig(outdir/"log.png")
        plt.close()
        torch.save(generator.state_dict(), outdir/"generator.ph")
        torch.save(discriminator.state_dict(), outdir/"discriminator.ph")
        step += 1
    save_temporal_graph(generator, discriminator, outdir)

def tainLSTM():
    num_nodes = 18
    num_edge_features = 0
    batch_size = 1
    training_samples = 1000
    validation_samples = 100
    loss_f = torch.nn.BCELoss()
    model = GraphGen(input_size=num_nodes**2, hidden_size=128 ,output_size=num_nodes**2, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    # The main idea is to encode the failure propagation network as a adjecency matrix
    # The temporal propagation will be encoded as a sequence of adjecency matrices
    # The first adjecency matrix will only encode a single node - the origin node of the failure
    # The end of the failure propagation will be encoded as a single adjecency matrix with all ones
    # The rest of the adjecency matrices will encode the propagation of the failure

    graphs = []
    val_graphs = []
    for _ in tqdm(range(training_samples), leave=False):
        graph = nx.fast_gnp_random_graph(num_nodes, 0.1)
        graphs.append(graph)
    encoded_graphs = [encode_graph(graph) for graph in graphs]

    for _ in tqdm(range(validation_samples), leave=False):
        val_graph = nx.fast_gnp_random_graph(num_nodes, 0.1)
        val_graphs.append(val_graph)
    encoded_validation_graphs = [encode_graph(val_graph) for val_graph in val_graphs]
    # Training
    for epoch in tqdm(range(epochs)):
        for graph in tqdm(encoded_graphs, leave=False):
            optimizer.zero_grad()
            generated = model(graph[:-1,:,:])
            loss = loss_f(generated, graph[1:,:,:])
            loss.backward()
            optimizer.step()
        eval(model, encoded_graphs, loss_f)

def main():
    trainGraphGenTemporal()
    
def save_graph(generator, discriminator, outdir="."):
    nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']
    node_num = len(nodes)
    random_data = torch.normal(0, 1, (1,1))
    generated: torch.Tensor = (generator(random_data).reshape((node_num,node_num)) > 0.5).squeeze().numpy()
    graph: nx.DiGraph = nx.from_numpy_array(generated, create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph, { k: nodes[i] for i, k in enumerate(graph.nodes) })
    nx.draw_circular(graph, with_labels=True)
    plt.savefig(Path(outdir)/"generated.png")
    plt.close()

def save_temporal_graph(generator, discriminator, outdir="."):
    nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']
    node_num = len(nodes)
    temporal_steps = 16
    random_data = torch.normal(0, 1, (1,1))
    generated: torch.Tensor = (generator(random_data).reshape((node_num,node_num, temporal_steps)) > 0.5).squeeze().numpy()
    fig, ax = plt.subplots(figsize=(16,9))
    def draw(frame):
        t = frame % temporal_steps
        graph: nx.DiGraph = nx.from_numpy_array(generated[:,:,t], create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, { k: nodes[i] for i, k in enumerate(graph.nodes) })
        nx.draw_circular(graph, with_labels=True, ax=ax)
    ani = plta.FuncAnimation(fig, draw, frames=temporal_steps, interval=1000, repeat=True)
    ani.save(Path(outdir)/"generated.gif")
    plt.close(fig)

def eval(model: torch.nn.Module, graphs: List[nx.Graph], loss_f: torch.nn.Module):
    losses = []
    with torch.no_grad():
        for graph in graphs:
            generated = model(graph[:-1,:,:])
            losses.append(loss_f(generated, graph[1:,:,:]))
        print("Mean validation loss:", np.mean(losses))

if __name__ == "__main__":
    main()