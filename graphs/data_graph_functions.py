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


def xsubgraphs(G, x, k, seed=42):
    """Randomly selects x number of subgraphs from G, where k is the number of steps from the central node
    of each subgraphto use

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
        g = nx.ego_graph(G, n, k)
        d = {list(g.nodes)[i]: 0 for i in range(len(list(g.nodes)))}
        d[n] = 1
        nx.set_node_attributes(g, d, 'start_nodes')
        subgraphs.append(g)

    return subgraphs


def plot_T0(G):
    colors = []
    for n in G.nodes:
        if G.nodes[n]['start_nodes'] == 1:
            colors.append('r')
        else:
            colors.append('b')

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, node_color=colors,
                                with_labels=True, node_size=300)
    lb = nx.draw_networkx_labels(G, pos, font_size=10)
    # plt.colorbar(nc)
    plt.axis('off')
    plt.show()


def get_fig(graph, time, graph_type_proper, algorithm_proper, f, k, title_ONOFF):
    neutral = []
    safe = []
    burned = []
    for n in graph:
        if nx.get_node_attributes(graph, 'b/n/d')[n] == 0:
            neutral.append(n)
        elif nx.get_node_attributes(graph, 'b/n/d')[n] == 1:
            safe.append(n)
        else:
            burned.append(n)

    nodes = {}
    for n in list(graph.nodes):
        nodes[n] = (n)

    # pos = nx.get_node_attributes(graph, 'pos')
    pos = nx.spring_layout(graph)
    nx.set_node_attributes(graph, pos, 'pos')

    fig = plt.figure(figsize=(6,7.2))#pylab.figure(figsize=(6,6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'pos'), nodelist=neutral, node_size=200,
                           node_color='b', label='unburnt/\nundefended')
    nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'pos'), nodelist=safe, node_size=200,
                           node_color='g', label='defended')
    nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'pos'), nodelist=burned, node_size=200,
                           node_color='r', label='burnt')
    nx.draw_networkx_edges(graph, nx.get_node_attributes(graph, 'pos'), width=2, edge_color='k')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.1, box.width, box.height*0.8])
    #nx.draw_networkx_labels(graph, pos, font_size=10, font_color='w')
    plt.legend(bbox_to_anchor=(0.5, -0.3),loc='lower center', fontsize=12)
    if title_ONOFF == 'ON':
        if f == 1 and k == 1:
            plt.title('FP with ${}$ for MAXSAVE and $t$-FIRE\non {}\nat Time $t$={} for {} Fire and {} Firefighter'.format(
                algorithm_proper,graph_type_proper,time, f, k), fontsize=15)
        elif f == 1 and k > 1:
            plt.title('FP with ${}$ for $t$-FIRE on {} \nat Time $t$={} for {} Fire and {} Firefighters'.format(
                algorithm_proper,graph_type_proper, time, f,k), fontsize=15)
        elif f > 1 and k==1:
            plt.title('FP with ${}$ for $t$-FIRE on {} \nat Time $t$={} for {} Fires and {} Firefighter'.format(
                algorithm_proper,graph_type_proper,time, f,k), fontsize=15)
        else:
            plt.title('FP with ${}$ for $t$-FIRE on {} \nat Time $t$={} for {} Fires and {} Firefighters'.format(
                algorithm_proper, graph_type_proper, time, f, k), fontsize=15)

    return fig


## DON'T USE UNLESS YOU FIX IT
def xsubgraphs2(G, x, p, k, seed=42):
    """Randomly selects x number of subgraphs from G, where k is the number of steps from the central node
    of each subgraphto use
    !!!! WILL NOT NECESSARILY CREATE CONNECTED GRAPH, DO NOT USE UNLESS YOU FIX IT

    :type G: networkx.Graph
    :param G: Networkx graph of interest
    :type x: int
    :param x: Number of subgraphs to build
    :type p: int
    :param p: number of start_nodes to use
    :type k: int
    :param k: Number of hops from start node to go out to build extended neighborhood
    :type seed: int
    :param seed: Number used to set the random seed to make results reproducible

    :rtype subgraphs: list
    :return subgraphs: List containing x number of randomly selected subgraphs from G
    """

    random.seed(seed)

    subgraphs = []

    for sg in range(x):
        # Select p number of start nodes at random
        start_nodes = random.sample(G.nodes, p)

        subs = []
        for n in start_nodes:
            # Get ego graph (k steps out) of each start node
            g = nx.ego_graph(G, n, k)
            # Append nodes of g to list of nodes in subgraph
            subs.append(list(g.nodes))

        # Flatten list, keep only unique, and get subgraph including all nodes
        sub = list(set([item for sublist in subs for item in sublist]))
        g = nx.subgraph(G, sub)

        # Create dictionary for defining start_node attributes (0 if not in start nodes)
        d = {list(g.nodes)[i]: 0 for i in range(len(list(g.nodes)))}
        for n in start_nodes:
            d[n] = 1 #1 if in start nodes

        # Set start_node attributes
        nx.set_node_attributes(g, d, 'start_nodes')

        # Append graph to list of subgraphs
        subgraphs.append(g)

    return subgraphs

# # Example
# G = load_graph(file_loc='../data/inf-power/inf-power.mtx')
# subgraphs = xsubgraphs(G, x=5, k=2, seed=42)
# print(nx.get_node_attributes(subgraphs[0], 'start_nodes'))
# for g in subgraphs:
#     plot_T0(g)


