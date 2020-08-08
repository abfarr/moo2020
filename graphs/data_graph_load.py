import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_adjlist('../data/inf-power/inf-power.mtx')

nx.draw(G)
plt.show()