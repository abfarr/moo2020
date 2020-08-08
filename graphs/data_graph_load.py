import networkx as nx
import matplotlib.pyplot as plt
<<<<<<< HEAD
from random import sample
=======
import random

random.seed(42)

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        newnbrs = set((nbr for n in nbrs for nbr in G[n]))
        nbrs=newnbrs |nbrs
    return nbrs
>>>>>>> 7a164bf6cd2f790801f77a40fb6ac9241c97e21f

G = nx.read_adjlist('../data/inf-power/inf-power.mtx')
random_subnodes = sample(G.nodes,20)
g = G.subgraph(random_subnodes)

<<<<<<< HEAD
nx.draw_spring(g)
plt.show()
=======
# subgraph_main_node = random.sample(G.nodes, 5)

mapping = dict(zip(G, range(0, len(G))))
G = nx.relabel_nodes(G, mapping)
print(G.nodes)
# print(knbrs(G, start=4, k=1))

# nx.draw_spring(knbrs(G, main_node=4, k=2), node_size=200, width=1.5, with_labels=True)
# plt.show()
# knbrs(G, main_node, k=1)
>>>>>>> 7a164bf6cd2f790801f77a40fb6ac9241c97e21f
