from typing import List
import networkx
import numpy as np

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
