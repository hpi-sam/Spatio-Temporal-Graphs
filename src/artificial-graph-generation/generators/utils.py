from pathlib import Path
from typing import List
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as plta
import networkx as nx
from torchviz import make_dot

def get_dataloader(dataset: str, batch_size: int):
    if os.path.exists(dataset + ".torch"):
        graphs = torch.load(dataset + ".torch")
    elif dataset.endswith(".torch") and os.path.exists(dataset):
        graphs = torch.load(dataset)
    elif dataset.endswith(".npy"):
        graphs = torch.from_numpy(np.load(dataset))
        torch.save(graphs, dataset + ".torch")
    else:
        graphs: torch.Tensor = torch.stack(list(map(lambda x: torch.tensor(x, dtype=torch.float), np.load(dataset).values())))
        torch.save(graphs, dataset + ".torch")
    dataloader = DataLoader(graphs.float(), shuffle=False, num_workers=0, batch_size=batch_size)
    return dataloader

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

def eval(model: torch.nn.Module, graphs: List[nx.Graph], loss_f: torch.nn.Module):
    losses = []
    with torch.no_grad():
        for graph in graphs:
            generated = model(graph[:-1,:,:])
            losses.append(loss_f(generated, graph[1:,:,:]))
        print("Mean validation loss:", np.mean(losses))

def save_model_graph(generator, discriminator, outdir="."):
    x = torch.normal(0, 1, size=(1,1))
    y_hat = generator(x)
    make_dot(y_hat, params=dict(list(generator.named_parameters()))).render(Path(outdir) / "generator_arch", format="png")
    y_hat = discriminator(y_hat.detach())
    make_dot(y_hat, params=dict(list(discriminator.named_parameters()))).render(Path(outdir) / "discriminator_arch", format="png")
    y_hat = discriminator(generator(x))
    make_dot(y_hat, params=dict(list(generator.named_parameters()) + list(discriminator.named_parameters()))).render(Path(outdir) / "complete_arch", format="png")
