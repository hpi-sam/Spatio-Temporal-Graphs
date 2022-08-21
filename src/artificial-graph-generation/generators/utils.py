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
from tqdm import tqdm

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

def get_sequence_dataloader(dataset: str, batch_size: int):
    if os.path.exists(dataset+".sequence.torch"):
        sequence_graphs = torch.load(dataset+".sequence.torch")
    else:
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
        next = torch.cumsum(graphs, dim=1) == 1
        sequence_graphs = []
        for graph in tqdm(next):
            sequence_graph = []
            for step in graph:
                items = torch.nonzero(step)
                if len(items) == 0:
                    step_encoding = torch.zeros(((graphs.shape[-2]+ 1) * 2))
                    step_encoding[(len(step_encoding) // 2) - 1] = 1
                    step_encoding[-1] = 1
                else:
                    step_encoding = torch.zeros(((graphs.shape[-2]+ 1) * 2))
                    step_encoding[items[0][0]] = 1
                    step_encoding[(graphs.shape[-2] + 1) + items[0][1]] = 1
                sequence_graph.append(step_encoding)
            sequence_graphs.append(torch.stack(sequence_graph, dim=0))
        sequence_graphs = torch.stack(sequence_graphs, dim=0)
        torch.save(sequence_graphs, dataset+".sequence.torch")
    print(sequence_graphs.shape)
    dataloader = DataLoader(sequence_graphs.float(), shuffle=False, num_workers=0, batch_size=batch_size)
    return dataloader

def save_temporal_graph(generator, discriminator, outdir=".", temporal=False):
    nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']
    node_num = len(nodes)
    temporal_steps = 16
    if not temporal:
        random_data = torch.normal(0, 1, (1,1))
        generated: torch.Tensor = (generator(random_data).reshape((node_num,node_num, temporal_steps)) > 0.5).squeeze().numpy()
    else:
        random_data = torch.normal(0.5, 0.5, (1,(node_num + 1) * 2))
        generated: torch.Tensor = sequence_to_adj(generator(random_data))
        generated = (generated.reshape((node_num,node_num, temporal_steps)) > 0.5).squeeze().numpy()
    fig, ax = plt.subplots(figsize=(16,9))
    def draw(frame):
        t = frame % temporal_steps
        graph: nx.DiGraph = nx.from_numpy_array(generated[:,:,t], create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, { k: nodes[i] for i, k in enumerate(graph.nodes) })
        nx.draw_circular(graph, with_labels=True, ax=ax)
    ani = plta.FuncAnimation(fig, draw, frames=temporal_steps, interval=1000, repeat=True)
    ani.save(Path(outdir)/"generated.gif")
    plt.close(fig)

def sequence_to_adj(seq: torch.Tensor, batched = True):
    if not batched:
        seq = seq.unsqueeze(0)
    graphs = torch.zeros((seq.shape[0], seq.shape[-1] // 2 - 1, seq.shape[-1] // 2 - 1, (seq.shape[-1] // 2) - 2))
    for graph_index, graph_seq in enumerate(seq):
        graph = graphs[graph_index]
        for index, item in enumerate(graph_seq[:-1]):
            if index != 0:
                graph[:,:, index] = graph[:,:, index - 1]
            node = torch.multinomial(item[:int((len(item)) // 2)], 1).item()
            nextnode = torch.multinomial(item[int((len(item)) // 2):], 1).item()
            if nextnode == ((len(item) // 2) - 1) or node == ((len(item) // 2) - 1):
                if index != 0:
                    prv = graph[:,:, index - 1]
                    for set_index in range(index, graph.size(-1)):
                        graph[:,:, set_index] = prv
                break
            try:
                graph[node, nextnode, index] = 1
            except Exception as e:
                print(node, nextnode, index)
                raise e
    return graphs

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
