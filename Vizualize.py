'''Visualize UV-space population'''
from chart_studio import plotly as py 
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display
from tabulate import tabulate

import utils
import Simulate_old as sim

params = utils.get_config()

def viz_UV(UV_space):
    #\\FIXME Most likely some recording of the UV space is incorrect as there is no dynamic change in the UV space 
    t=0
    df = pd.DataFrame(UV_space[0][t], columns=['U'])
    df['V'] = UV_space[1][t]

    fig = px.scatter(df, x="U", y="V")
    fig = px.density_contour(df, x="U", y="V", marginal_x="histogram", marginal_y="histogram")
    #fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    fig.show()

def viz_deg_distr(G):

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()



def heatmap(N, rounds, steps):
    '''Compute heatmap for various U and V values for payoffs'''

    U = np.linspace(0,1,10)
    V = np.linspace(0,1,10)
    heat_map = []
    data = []

    for u in U:
        temp = []
        temp2 = []
        for v in V:
            Payoffs = sim.simulateANDWealth(N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,u,v))
            #heatmap[u][v] = gini_new(np.array(Payoffs[-1]))
            temp.append(sim.gini_new(np.array(Payoffs[-1])))
            temp2.append(Payoffs[-1])
        heat_map.append(temp)
        data.append(temp2)

def viz_effect_clustering_coef(name, p, data):
    fig, ax = plt.subplots()
    ax.scatter(p, data, c="green", alpha=0.5, marker='x')
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel(name)
    ax.legend()
    plt.show()


def measures_over_timesteps(df):

    # Get steps, used networks and used measures from dataframe
    steps = np.arange(df['Step'].nunique())
    networks = df['Network'].unique()

    # Only first two columns of dataframe do not contain measures
    measures = list(df.columns)[2:]
    mean_measures = measures[:len(measures)//2]

    fig, axs = plt.subplots(len(networks), len(mean_measures))

    # Make plot per measure and networktype
    for row in range((len(networks))):
        for col in range(len(mean_measures)):
            mean = df.loc[df['Network'] == networks[row]][measures[col]]
            std = df.loc[df['Network'] == networks[row]][measures[col+len(mean_measures)]]
            axs[row, col].plot(steps, mean)
            axs[row, col].errorbar(steps, mean, xerr=0.5, yerr=2*std,
               linestyle='--', c='blue', alpha = 0.7)

    # Share x per column and y per row
    for ax in fig.get_axes():
        ax.label_outer()

    # Set common x label
    fig.text(0.5, 0.04, 'Step', ha='center')

    # Set y labels
    for ax, row in zip(axs[:,0], mean_measures):
        ax.set_ylabel(row)

    # Make clear which network belongs to which graph
    for ax, col in zip(axs[0,:], networks):
        ax.set_title(col)
    
    plt.show()









