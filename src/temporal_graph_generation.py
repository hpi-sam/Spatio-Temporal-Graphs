import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta
import networkx.algorithms.isomorphism as iso

def sample_subgraph(super_graph, start_node: str, bfs_prob=0.5, max_nodes=None, random_seed=42) -> nx.DiGraph():
    np.random.seed(random_seed)
    subgraph=nx.DiGraph()
    subgraph.add_node(start_node)
    nodes_to_visit = [start_node]
    if max_nodes is None:
        max_nodes = len(super_graph.nodes())
    assert max_nodes>0, 'enter positive max_nodes'
    for node in nodes_to_visit:
        potential_next_nodes = []
        potential_next_nodes = [n for n in list(super_graph.neighbors(node)) if not n in nodes_to_visit]
        if len(potential_next_nodes) > 0:
            if np.random.choice([0, 1], p=[1-bfs_prob, bfs_prob]):
                # bfs way
                next_nodes = potential_next_nodes
            else:
                # dfs way
                next_nodes = [potential_next_nodes[np.random.choice(len(potential_next_nodes))]]
            
            num_nodes_to_add = np.minimum(max_nodes-len(subgraph.nodes()), len(next_nodes))
            next_nodes = next_nodes[:num_nodes_to_add]
            nodes_to_visit += next_nodes
            for next_node in next_nodes:
                subgraph.add_node(next_node)
                subgraph.add_edge(node, next_node)
        if len(subgraph.nodes())>=max_nodes:
            return subgraph
    return subgraph

def _has_received_propagation(subgraph, node):
    in_edges = list(subgraph.in_edges(node))
    all_timesteps = nx.get_edge_attributes(subgraph, "timestep")
    in_edges_with_timesteps = [edge for edge in in_edges if edge in all_timesteps]
    return len(in_edges_with_timesteps) > 0

# in-place modification
def add_timesteps_to_graph(subgraph, start_node, random_seed=42):
    np.random.seed(random_seed)
    timestep = 0
    edge_pool = {edge:1 for edge in list(subgraph.out_edges(start_node))}

    while edge_pool:
        edge_to_sample_idx = np.random.choice(range(len(edge_pool.keys())),p=np.array(list(edge_pool.values()))/np.sum(list(edge_pool.values())))
        edge_to_sample = list(edge_pool.keys())[edge_to_sample_idx]
        del edge_pool[edge_to_sample]
        nx.set_edge_attributes(subgraph, {edge_to_sample: {"timestep": timestep}})
        for key in edge_pool.keys():
            edge_pool[key] += 1
        new_node = edge_to_sample[1]
        # don't propagate to the same node twice
        edge_pool.update({edge:1 for edge in list(subgraph.out_edges(new_node)) if not _has_received_propagation(subgraph, edge[1])})
        timestep += 1


def _propagation_timestamp_next_node(graph, subgraph, current_node, neighbor, nodes_to_visit, datetime_format, maxlag):
    if neighbor in nodes_to_visit:
        return None
    #print(graph.nodes[neighbor])
    neighbor_data = graph.nodes[neighbor]
    if not neighbor_data.get("anomaly"):
        return None
    current_timestamp = subgraph.nodes[current_node]["propagation_timestamp"]
    current_timestamp = datetime.strptime(current_timestamp, datetime_format)
    neighbor_anomaly_timestamps = np.array([datetime.strptime(timestamp, datetime_format) for timestamp in neighbor_data["anomaly_timestamps"]])
    max_propagation_timestamp = None
    
    time_diff = neighbor_anomaly_timestamps - current_timestamp
    propagation_timestamps = neighbor_anomaly_timestamps[np.logical_and((timedelta(minutes=0) <= time_diff), (time_diff <= timedelta(minutes=maxlag)))]
    if not len(propagation_timestamps):
        return None
    max_propagation_timestamp = datetime.strftime(np.max(propagation_timestamps), datetime_format)
    return max_propagation_timestamp


def sample_subgraph_with_timestamp_constraint(super_graph, start_node: str, start_timestamp: str, datetime_format, maxlag, bfs_prob=0.5, max_nodes=None, random_seed=42) -> nx.DiGraph():
    np.random.seed(random_seed)
    subgraph=nx.DiGraph()
    subgraph.add_node(start_node, propagation_timestamp=start_timestamp)
    nodes_to_visit = [start_node]
    if max_nodes is None:
        max_nodes = len(super_graph.nodes())
    assert max_nodes>0, 'enter positive max_nodes'
    for node in nodes_to_visit:
        potential_next_nodes = []
       
        for n in list(super_graph.neighbors(node)):
            propagation_timestamp = _propagation_timestamp_next_node(super_graph, subgraph, node, n, nodes_to_visit, datetime_format, maxlag)
            if propagation_timestamp is not None:
                potential_next_nodes.append((n, propagation_timestamp))
    
        if len(potential_next_nodes) > 0:
            if np.random.choice([0, 1], p=[1-bfs_prob, bfs_prob]):
                # bfs way
                next_nodes = potential_next_nodes
            else:
                # dfs way
                next_nodes = [potential_next_nodes[np.random.choice(len(potential_next_nodes))]]
            
            num_nodes_to_add = np.minimum(max_nodes-len(subgraph.nodes()), len(next_nodes))
            next_nodes = next_nodes[:num_nodes_to_add]
            nodes_to_visit += [n for n, timestamp in next_nodes]

            for next_node, p_timestamp in next_nodes:
                subgraph.add_node(next_node, propagation_timestamp=p_timestamp)
                subgraph.add_edge(node, next_node, propagation_timestamp=p_timestamp)
        
        if len(subgraph.nodes())>=max_nodes:
            return subgraph
    return subgraph


def sample_temporal_subgraphs(super_graph, start_node, max_nodes=None, seed=np.random.seed(111), num_sub_seeds=1000):
    if not max_nodes:
        max_nodes = len(super_graph.nodes()) + 1
    tn_subgraphs = []
    for seed in tqdm(np.random.choice(10000000, num_sub_seeds)):
        for max_nodes_no in range(2, max_nodes):
            for bfs_prob in [0.1, 0.5, 0.9]:
                tn_subgraph = sample_subgraph(super_graph, start_node, max_nodes=max_nodes_no, bfs_prob=bfs_prob, random_seed=seed)
                add_timesteps_to_graph(tn_subgraph, start_node, random_seed=seed)
                if tn_subgraphs:
                    if not np.any([nx.utils.graphs_equal(tn_subgraph, s) for s in tn_subgraphs]):
                        tn_subgraphs.append(tn_subgraph)
                else:
                    tn_subgraphs.append(tn_subgraph)
    return tn_subgraphs

def get_complete_teneto_edgelist(graph):
    add_edges = []

    edge_df = nx.to_pandas_edgelist(graph)
    edge_df.columns = ['i','j','t']
    edge_df = edge_df.apply(pd.to_numeric)

    max_time = np.max(edge_df['t'])
    for i in edge_df.index:
        row = edge_df.iloc[i]
        times_to_add = np.arange(row["t"], max_time+1)
        for t_to_add in times_to_add:
            add_edges.append([row["i"], row["j"], t_to_add])
    
    fill_edges = pd.DataFrame(add_edges, columns=edge_df.columns)
    return fill_edges
