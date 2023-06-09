import networkx as nx
import matplotlib.pyplot as plt


G = nx.powerlaw_cluster_graph(4, 2, 0, seed=None)
nx.draw(G)
plt.show()