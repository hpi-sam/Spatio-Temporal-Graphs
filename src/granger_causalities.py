import networkx as nx
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

MAXLAG = 2

def read_graph(filename) -> nx.Graph:
    """
    Reads a graph from a file.
    """
    tree = ET.parse(filename)
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

def main():
    graph = read_graph("data/Sock-shop-xml.txt")

    f = 'data/loadtest.csv'
    P = pd.read_csv(f)
    nx.set_edge_attributes(graph, False, name="causal")
    print(graph.edges.data())
    #perform Granger-Causality test
    for source, target in graph.edges:
        if source in P.columns and target in P.columns:
            gc_dicts = grangercausalitytests(P[[source, target]], MAXLAG, verbose=False)
            p_values = [lag_result[0]['ssr_ftest'][1] for lag_result in gc_dicts.values()]
            print(gc_dicts)
            if np.min(p_values) < 0.05:
                nx.set_edge_attributes(graph, {(source, target): {"causal": True}})
    print(graph.edges.data())
    colors = []
    for u,v in graph.edges:
        if u in P.columns and v in P.columns:
            if graph[u][v]["causal"]:
                colors.append("r")
            else:
                colors.append("b")
        else:
            colors.append("g")

    nx.layout.spring_layout(graph)
    nx.draw_networkx(graph, edge_color=colors)
    plt.show()
    #nx.write_graphml(graph, f"data/Sock-shop-gc-maxlag{MAXLAG}.graphml")

    



if __name__ == '__main__':
    main()
