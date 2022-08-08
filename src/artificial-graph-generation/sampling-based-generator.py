import argparse
import gzip
import random
import warnings
from typing import List
import networkx
import numpy as np
import pandas as pd
from tqdm import tqdm

def encode_graphs(graphs: List[networkx.Graph], node_order: List[str]) -> np.ndarray:
    """
    Encodes a list of graphs into a array.
    """
    graph_seperator = 255
    numbers = []
    for graph in graphs:
        for edge in graph.edges():
            numbers.append(edge[0])
            numbers.append(edge[1])
        numbers.append(graph_seperator)
    return np.array(numbers, dtype=np.uint8)

def decode_graphs(encoded_graphs: np.ndarray, node_order: List[str]) -> List[networkx.Graph]:
    """
    Decodes a list of graphs from a array.
    """
    graph_seperator = 255
    graphs = []
    current_graph = networkx.DiGraph()
    current_graph.add_nodes_from(list(range(len(node_order))))
    last_number = None
    for number in encoded_graphs:
        if number == graph_seperator:
            if last_number is not None:
                raise ValueError("Invalid encoded graph")
            graphs.append(current_graph)
            current_graph = networkx.DiGraph()
            current_graph.add_nodes_from(list(range(len(node_order))))
        else:
            if last_number is not None:
                current_graph.add_edge(last_number, number)
                last_number = None
                continue
            last_number = number
    return graphs

def build_causal_graph(path: str, node_order: List[str]) -> networkx.DiGraph:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = networkx.read_graphml(path, node_type=str, edge_key_type=str, force_multigraph=True)
    nodes = {old:node_data["id"] for old, node_data in dict(g.nodes(data=True)).items()}
    networkx.relabel_nodes(g, nodes, copy=False)  
    def is_gc_edge(n1, n2, k):
        return g[n1][n2][k]["label"] == 'CONNECTED' and g[n1][n2][k].get("causal", 'false') == 'true'
    cg = networkx.subgraph_view(g, filter_edge=is_gc_edge)
    return cg

def make_temporal(graph: networkx.DiGraph, node_order: List[str]) -> np.array:
    start_node_name = "front-end"
    start_node = node_order.index(start_node_name)
    # Creates a temporal adjacency matrixes
    neighbors = list(map(lambda x: (start_node, x), graph.neighbors(start_node)))
    included = [start_node]
    time_steps = []
    all_edges = []
    while len(neighbors) != 0:
        new_edge = random.choice(neighbors)
        neighbors.remove(new_edge)
        new_node = new_edge[1]
        all_edges.append(new_edge)
        included.append(new_node)
        new_subgraph = networkx.DiGraph()
        new_subgraph.add_nodes_from(range(len(node_order)))
        new_subgraph.add_edges_from(all_edges)
        time_steps.append(new_subgraph)
        neighbors += list(map(lambda x: (new_node, x), filter(lambda x: x not in included, graph.neighbors(new_node)))) 
    if len(time_steps) != 0:
        time_steps.extend([time_steps[-1]]*(len(node_order) - len(time_steps) - 1))
    else:
        new_subgraph = networkx.DiGraph()
        new_subgraph.add_nodes_from(range(len(node_order)))
        time_steps = [new_subgraph] * (len(node_order) - 1)
    arrays = list(map(lambda x: networkx.to_numpy_array(x), time_steps))
    return np.stack(arrays, axis=-1)

def main():
    node_order = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']
    parser = argparse.ArgumentParser(description="Sample from correlation graph")
    parser.add_argument("--data", type=str, required=True, help="Path to the data")
    parser.add_argument("--output", type=str, required=True, help="Path to the output")
    parser.add_argument("--num-samples", "-n", type=int, required=False, default=10_000, help="Number of samples")
    parser.add_argument("--compress", action="store_true", help="Compress output with gzip")
    parser.add_argument("--use-causality", type=str, help="Use causality from given graphml to sample new graphs")
    parser.add_argument("--temporal", action="store_true", help="Turn on temporal_generation")
    args = parser.parse_args()

    full_data = pd.read_csv(args.data, index_col=0)
    data_without_time = full_data.drop("Time", axis=1)

    num_graphs = args.num_samples
    c = data_without_time.corr()
    d = c.to_numpy()
    np.fill_diagonal(d, 0)
    if args.use_causality:
        order = data_without_time.columns.to_list() + ['worker1', 'worker2', 'master']
        print(order)
        causal_graph = build_causal_graph(args.use_causality, node_order)
        assert set(order) == set(causal_graph.nodes()), f"{set(order)}, {set(causal_graph.nodes())}, {set(order) - set(causal_graph.nodes())}, {set(causal_graph.nodes()) - set(order)}"
        networkx.to_numpy_array(causal_graph, order)
    graphs = []
    correlation_graph = networkx.from_numpy_array(d, create_using=networkx.Graph)
    for edge in correlation_graph.edges():
        correlation_graph.remove_edge(*edge)
        correlation_graph.add_edge(edge[0], edge[1], weight=d[edge[0], edge[1]])
    correlation_graph: networkx.Graph = correlation_graph
    for _ in tqdm(range(num_graphs)):
        new_graph = networkx.DiGraph()
        initial_node = list(correlation_graph.nodes)[np.random.randint(0, len(correlation_graph.nodes))]
        initial_node = node_order.index("front-end")
        new_graph.add_node(initial_node)
        previous_random_neighbors = [initial_node]
        edge_labels = networkx.get_edge_attributes(correlation_graph, "weight")
        reverse_edge_labels = {(key[1], key[0]): value for key, value in edge_labels.items()}
        edge_labels = {**edge_labels, **reverse_edge_labels}
        while len(previous_random_neighbors):
            new_random_neighbors = []
            for node in previous_random_neighbors:
                # Get random edge from node to another node in the graph using weight as probability
                genrated_propagation = [edge[1] for edge in correlation_graph.edges(node) if edge_labels[edge] > np.random.random() and edge[1] not in new_graph.nodes]
                new_random_neighbors += genrated_propagation
                for neighbor in genrated_propagation:
                    new_graph.add_edge(node, neighbor)
            previous_random_neighbors = new_random_neighbors
            np.random.shuffle(previous_random_neighbors)
        graphs.append(new_graph)
    if args.temporal:
        temporal_graphs = map(lambda x: make_temporal(x, node_order), tqdm(graphs))
        np.savez_compressed(args.output, *temporal_graphs)
    else:
        encoded_graphs = encode_graphs(graphs, node_order)
        if args.compress:
            with gzip.open(args.output, "wb") as file_handle:
                np.save(file_handle, encoded_graphs)
        else:
            np.save(args.output, encoded_graphs)

if __name__ == '__main__':
    main()
