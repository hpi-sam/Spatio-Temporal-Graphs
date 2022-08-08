import argparse
from datetime import datetime
import logging
from math import inf
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

def load_graphs(dataset) -> torch.utils.data.DataLoader:
    loaded = np.load(dataset)
    return [torch.tensor(graph) for graph in loaded.values()]

def trainGraphGenTemporalOneShot(
    dataset="../generated/100k_frontend_graphs-temporal_unique.npy",
    outdir=Path("./output") / "LSTMAutoregressive"
):
    outdir = Path(outdir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(outdir, exist_ok=False)
    num_nodes = 17
    time_steps = 16
    evaluate_size = 1000
    epochs = 100
    batch_size = 100
    step = 0
    clip=True
    clip_value=1.0
    discriminator = GraphGenLSTMDescriminator(num_nodes, activation=torch.nn.Tanh, temporal=num_nodes-1, layers=[128])
    generator = GraphGenLSTMGenerator(num_nodes, activation=torch.nn.Tanh, layers=[128])
    best_g_loss = inf
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
        write_handle.write(f"Grandient clipping: {clip}\n")
        write_handle.write(f"Grandient clipping value: {clip_value}\n")
    log = {
        "d_generated": [],
        "d_real": [],
        "d_loss": [],
    }
    mean_differences = []
    while step <= epochs:
        bar = tqdm(loader, desc=f"[{step}/{epochs}]")
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
            if clip:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            generator_optimizer.step()
            log["d_generated"].append(d_generated.detach().cpu().item())
            log["d_loss"].append(d_loss.detach().cpu().item())
            log["d_real"].append(d_real.detach().cpu().item())
            bar.set_postfix({
                "d_generated": d_generated.detach().cpu().item(),
                "d_real": d_real.detach().cpu().item(),
                "d_loss": d_loss.detach().cpu().item(),
            })
            if d_generated.detach().cpu().item() < best_g_loss:
                best_g_loss = d_generated.detach().cpu().item()
                torch.save(generator.state_dict(), outdir/"best-loss-generator.ph")
                torch.save(discriminator.state_dict(), outdir/"best-loss-discriminator.ph")
        # Evaluate
        with torch.no_grad():
            random_data = torch.normal(0, 1, size=(evaluate_size, 1), device=device)
            generated = generator(random_data) > 0.5
            generated = generated.reshape(-1, num_nodes, num_nodes, time_steps).cpu().numpy()
            measures = []
            for index in range(generated.shape[0]):
                graph = generated[index]
                measures.append(teneto.networkmeasures.temporal_degree_centrality(teneto.TemporalNetwork(from_array=graph, nettype="bu"), calc="overtime"))
            try:
                mean = np.mean(measures, axis=0)
                real_mean = np.mean(np.load("tdc.npy"), axis=0)
                mean_differences.append(mean-real_mean)
            except Exception as e:
                print(e)
                mean_differences.append(np.array([inf]*num_nodes))
            print(mean_differences[-1])
        if sum(mean_differences[-1]) == min(map(sum, map(abs, mean_differences))):
            print("minimum")
            torch.save(generator.state_dict(), outdir/"best-generator.ph")
            torch.save(discriminator.state_dict(), outdir/"best-discriminator.ph")
        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        df.plot(xlabel="step", ylabel="value", title=f"Autoregressive GAN Graph Generation")
        plt.savefig(outdir/"log.png")
        plt.close()
        df2 = pandas.DataFrame.from_records(mean_differences, columns=['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master'])
        df2.to_csv(outdir/"tdegree-differences.csv")
        df2.plot(xlabel="step", ylabel="difference", title=f"Autoregressive GAN Graph Generation - TDegree Difference")
        plt.savefig(outdir/"tdegree-differences.png")
        plt.close()
        torch.save(generator.state_dict(), outdir/"latest-generator.ph")
        torch.save(discriminator.state_dict(), outdir/"latest-discriminator.ph")
        step += 1
    save_temporal_graph(generator, discriminator, outdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-best", action="store_true", help="Run best model from last run.")
    parser.add_argument("--val-exp", type=str, help="Path to experiment directory to run the validation from.")
    args = parser.parse_args()
    if args.run_best:
        if not args.val_exp:
            raise Exception("Specify experiment dir.")
        path = Path(args.val_exp)
        laod_and_test(path)
    else:
        logging.info("Starting Training")
        trainGraphGenTemporalOneShot()
    
def laod_and_test(dir):
    eval_dir = dir/"eval"
    os.makedirs(eval_dir, exist_ok=True)
    num_nodes = 17
    discriminator = GraphGenLSTMDescriminator(num_nodes, activation=torch.nn.Tanh, temporal=num_nodes-1, layers=[128])
    generator = GraphGenLSTMGenerator(num_nodes, activation=torch.nn.Tanh, layers=[128])
    discriminator.load_state_dict(torch.load(dir/"best-discriminator.ph"))
    generator.load_state_dict(torch.load(dir/"best-generator.ph"))
    save_temporal_graph(generator, discriminator, eval_dir)

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
