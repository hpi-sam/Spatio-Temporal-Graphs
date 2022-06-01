import traceback
from typing import Any, Dict, List
import xml.etree.ElementTree as ElementTree
from cv2 import STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL
import matplotlib.pyplot as plt
import numpy as np
from spot import bidSPOT
import datetime
import pandas as pd
import networkx as nx


def read_graph(filename) -> nx.Graph:
    """
    Reads a graph from a file.
    """
    tree = ElementTree.parse(filename)
    print(tree)
    root = tree.getroot()
    nodes = set()
    G = nx.DiGraph()
    for child in root.iter("vertex"):
        node = child.attrib["id"]
        if node not in nodes:
            nodes.add(node)
            G.add_node(node)
        for adj in child.iter("adjacent"):
            adjacent_node = adj.attrib["vertex"]
            if adjacent_node not in nodes:
                nodes.add(adjacent_node)
                G.add_node(adjacent_node)
            G.add_edge(node, adjacent_node)
    return G

def find_analomies(data, init_data, column_name: str):
    init_data = init_data[column_name].to_numpy()
    data = data[column_name].to_numpy()
    q = 1e-5 # risk parameter
    d = 10	# depth
    s = bidSPOT(q,d) # bidSPOT object
    s.fit(init_data, data) # data import
    s.initialize(verbose=False) # initialization step
    results = s.run() 	# run
    alarms = results["alarms"]
    if len(alarms) > 0:
        num_analomies = ((alarms[1:] - np.roll(np.array(alarms), 1)[1:]) > 1).sum() + 1 # +1 because we do not count the first analomy this way
    else:
        num_analomies = 0
    results["num_analomies"] = num_analomies
    return results

def build_anamoly_subgraphs(graph: nx.DiGraph, alarms: Dict[str, Dict[str, Any]], length: int, max_lag: int = 1) -> List[nx.DiGraph]:
    mat = np.zeros((len(graph.nodes), length))
    for node_index, node in enumerate(graph.nodes):
        if node in alarms:
            for alarm_index in alarms[node]["alarms"]:
                mat[node_index, alarm_index] = 1
    print(mat.sum(axis=1))
    # Extend errors to account for allowed lag
    for index in range(length):
        for node_index in range(len(graph.nodes)):
            if mat[node_index, index] == 1:
                for lag in range(1, max_lag + 1):
                    if index + lag < length:
                        if mat[node_index, index + lag] == 0:
                            mat[node_index, index + lag] = 1
    # Todo: split by causality and failure
    # get index of zero columns
    zero_column_indices = np.where(np.sum(mat, axis=0) == 0)[0]
    # get consecutive nonzero columns
    mat_alarm_sequences = np.split(mat, zero_column_indices, axis=1)
    # remove zero columns from alarm sequences
    total_index = 0
    new_sequences = []
    for sequence in mat_alarm_sequences:
        sequence_length = sequence.shape[1]
        to_delete_indices = zero_column_indices[(zero_column_indices >= total_index) & (zero_column_indices < total_index + sequence_length)] - total_index
        total_index += sequence_length
        if sequence_length == len(to_delete_indices):
            continue
        new_sequence = np.delete(sequence, to_delete_indices, axis=1)
        new_sequences.append(new_sequence)
    mat_alarm_sequences = new_sequences

    print(f"Found {len(mat_alarm_sequences)} sequences of alarms from which subgraphs can be built.")
    #print(mat_alarm_sequences)
    subgraphs = []
    # for each row remove row that is not zero from graph
    for col_sequence in mat_alarm_sequences:
        subgraph = graph.copy()
        for node_index in np.where(np.sum(col_sequence, axis=1) == 0)[0]:
            subgraph.remove_node(list(graph.nodes)[node_index])
        # Todo: split subgraph by connected components
        nx.draw(subgraph)
        plt.show()
        subgraphs += [subgraph.subgraph(c).copy() for c in nx.weakly_connected_components(subgraph)]
    return subgraphs

def main():
    # Load data and graph
    full_data = pd.read_csv("./data/loadtest.csv", index_col=0)
    dates = full_data['Time'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date())
    graph = read_graph("data/Sock-shop-xml.txt")

    # Prepare data
    data_dates = (dates == datetime.date(2022,3,9)) | (dates == datetime.date(2021,10,19))
    init_data_dates = (dates == datetime.date(2021,10,14)) | (dates == datetime.date(2021,10,20))
    data = full_data[data_dates].sort_values(by=['Time'])
    init_data = full_data[init_data_dates].sort_values(by=['Time'])
    data_length = len(data)
    print(f"actual data length: {len(data)}")
    print(f"init data lenght: {len(init_data)}")

    # Todo: Split data into chunks with a time delta of 1 minute and run bidSPOT on each chunk
    
    # Find analomies
    result_strings = []
    all_results = {}
    csv_nodes = data.columns.drop(["Time"])
    print("Node names from data csv: ", sorted(csv_nodes))
    graph_nodes = graph.nodes
    print("Node names from graph: ", sorted(graph_nodes))
    print("Nodes in graph but not in data csv: ", sorted(set(graph_nodes) - set(csv_nodes)))
    print("Nodes in data csv but not in graph: ", sorted(set(csv_nodes) - set(graph_nodes)))
    for col in csv_nodes:
        try:
            results = find_analomies(data, init_data, col)
            all_results[col] = results
            result_strings.append(f"{col} analomies: {results['num_analomies']}")
        except Exception:
            print(traceback.format_exc())
            result_strings.append(f"{col} error")
    print("\n".join(result_strings))

    # Build subgraphs from anamolies
    # Todo: Maybe exchange connectivity graph with causality graph
    graphs = build_anamoly_subgraphs(graph, all_results, data_length, max_lag=0)
    print(f"Found {len(graphs)} anamoly subgraphs")
    for graph in graphs:
        nx.draw(graph, with_labels=True)
        plt.show()
    # results = find_analomies(data, init_data, "orders")

if __name__ == "__main__":
    main()
