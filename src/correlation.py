import argparse
from http.client import GATEWAY_TIMEOUT
from typing import List
import networkx
import pandas as pd
import numpy as np
from tqdm import tqdm
import gzip

def encode_graphs(graphs: List[networkx.Graph]) -> np.ndarray:
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

def encode_temporal_graphs(graphs: List[networkx.Graph]) -> np.ndarray:
    """
    Encodes a list of graphs into a array.
    """
    graph_seperator = 255
    edge_separator = 254
    numbers = []
    for graph in graphs:
        for edge in networkx.to_edgelist(graph):
            numbers.append(edge[0])
            numbers.append(edge[1])
            numbers.append(edge[2]['timestep'])
            numbers.append(edge_separator)
        numbers.append(graph_seperator)
    return np.array(numbers, dtype=np.uint8)

def decode_temporal_graphs(encoded_graphs: np.ndarray) -> List[networkx.Graph]:
    """
    Decodes a list of graphs from a array.
    """
    graph_seperator = 255
    edge_separator=254
    graphs = []
    current_graph = networkx.DiGraph()
    edge_start = None
    edge_end = None
    edge_timestep = None
    for number in encoded_graphs:
        if number == graph_seperator:
            if edge_start is not None or edge_end is not None or edge_timestep is not None:
                raise ValueError("Invalid encoded graph")
            graphs.append(current_graph)
            current_graph = networkx.DiGraph()
        else:
            if edge_start is None:
                edge_start = number
                continue
            if edge_end is None:
                edge_end = number
                continue
            if edge_timestep is None:
                edge_timestep = number
                continue
            if number == edge_separator:
                if edge_start is None or edge_end is None or edge_timestep is None:
                    raise ValueError("Invalid encoded graph")
                current_graph.add_edge(edge_start, edge_end, timestep=edge_timestep)
                edge_start = None
                edge_end = None
                edge_timestep = None
                continue
            raise ValueError("Invalid encoded graph")
    return graphs

def decode_graphs(encoded_graphs: np.ndarray) -> List[networkx.Graph]:
    """
    Decodes a list of graphs from a array.
    """
    graph_seperator = 255
    graphs = []
    current_graph = networkx.DiGraph()
    last_number = None
    for number in encoded_graphs:
        if number == graph_seperator:
            if last_number is not None:
                raise ValueError("Invalid encoded graph")
            graphs.append(current_graph)
            current_graph = networkx.DiGraph()
        else:
            if last_number is not None:
                current_graph.add_edge(last_number, number)
                last_number = None
                continue
            last_number = number
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Sample from correlation graph")
    parser.add_argument("--data", type=str, required=True, help="Path to the data")
    parser.add_argument("--output", type=str, required=True, help="Path to the output")
    parser.add_argument("--num-samples", "-n", type=int, required=False, default=10_000, help="Number of samples")
    parser.add_argument("--compress", action="store_true", help="Compress output with gzip")
    args = parser.parse_args()

    full_data = pd.read_csv(args.data, index_col=0)
    data_without_time = full_data.drop("Time", axis=1)

    num_graphs = args.num_samples
    c = data_without_time.corr()
    d = c.to_numpy()
    np.fill_diagonal(d, 0)
    graphs = []
    correlation_graph = networkx.from_numpy_array(d, create_using=networkx.Graph)
    for edge in correlation_graph.edges():
        correlation_graph.remove_edge(*edge)
        correlation_graph.add_edge(edge[0], edge[1], weight=d[edge[0], edge[1]])
    correlation_graph: networkx.Graph = correlation_graph
    for i in tqdm(range(num_graphs)):
        new_graph = networkx.DiGraph()
        initial_node = list(correlation_graph.nodes)[np.random.randint(0, len(correlation_graph.nodes))]
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
    encoded_graphs = encode_graphs(graphs)
    if args.compress:
        with gzip.open(args.output, "wb") as file_handle:
            np.save(file_handle, encoded_graphs)
    else:
        np.save(args.output, encoded_graphs)

if __name__ == '__main__':
    main()
