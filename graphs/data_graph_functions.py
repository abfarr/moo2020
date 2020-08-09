import networkx as nx
import matplotlib.pyplot as plt
import random


def load_graph(file_loc='../data/inf-power/inf-power.mtx'):
    """Load graph from adjacency list file

    :type file_loc: str
    :param file_loc: Location of adjacency list file for data graph
    :rtype G: networkx.Graph
    :return G: Networkx Graph loaded from file with attributes 'B' and 'D' and node names 0 to size of graph-1
    """

    G = nx.read_adjlist(file_loc)

    # Rename nodes as integers (0 to size of graph)
    mapping = dict(zip(G, range(0, len(G))))
    G = nx.relabel_nodes(G, mapping)

    # Give each node a burnt and defended status attribute and set all to 0 to start
    init_bd_status = [0 for n in G.nodes()]
    nx.set_node_attributes(G, 0, 'B')
    nx.set_node_attributes(G, 0, 'D')

    return G


def knbrs(G, start, k):
    """Gets all nodes in graph G at or within k hops of start node

    :type G: networkx.Graph
    :param G: Networkx graph of interest
    :type start: int, string
    :param start:  Name of initial node to use to build the extended neighborhood from
    :type k: int
    :param k: Number of hops from start node to go out to build extended neighborhood

    :rtype nbrs: set
    :return nbrs: Neighbors of start node at or within k hops away
    """

    nbrs = {start}
    for l in range(k):
        new_nbrs = set((nbr for n in nbrs for nbr in G[n]))
        nbrs = new_nbrs | nbrs
    return nbrs


def xsubgraphs(G, x, k, seed=42):
    """Randomly selects x number of subgraphs from G, where k is the number of steps from central nodes to use
    to create a connected subgraph
    
    :type G: networkx.Graph
    :param G: Networkx graph of interest
    :type x: int
    :param x: Number of subgraphs to build
    :type k: int
    :param k: Number of hops from start node to go out to build extended neighborhood
    :type seed: int
    :param seed: Number used to set the random seed to make results reproducible

    :rtype subgraphs: list
    :return subgraphs: List containing x number of randomly selected subgraphs from G
    """

    random.seed(seed)
    start_nodes = random.sample(G.nodes, x)

    subgraphs = []
    for n in start_nodes:
        subgraph_nodes = knbrs(G, start=n, k=k)
        subgraphs.append(G.subgraph(subgraph_nodes))

    return subgraphs

# # Example
G = load_graph(file_loc='../data/inf-power/inf-power.mtx')
# subgraphs = xsubgraphs(G, x=5, k=5, seed=42)
# nx.draw_spring(subgraphs[0], node_size=200, width=1.5, with_labels=True)
# plt.show()
