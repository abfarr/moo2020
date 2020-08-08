import networkx as nx
import matplotlib.pyplot as plt
from random import sample

G = nx.read_adjlist('../data/inf-power/inf-power.mtx')
random_subnodes = sample(G.nodes,20)
g = G.subgraph(random_subnodes)

nx.draw_spring(g)
plt.show()