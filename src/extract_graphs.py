import networkx
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def read_graph(filename) -> networkx.Graph:
    """
    Reads a graph from a file.
    """
    tree = ET.parse(filename)
    print(tree)
    root = tree.getroot()
    nodes = set()
    G = networkx.DiGraph()
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
    networkx.layout.spring_layout(graph)
    networkx.draw_networkx(graph)
    plt.show()

if __name__ == '__main__':
    main()
