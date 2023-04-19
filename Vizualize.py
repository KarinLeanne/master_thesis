'''Visualize UV-space population'''
from chart_studio import plotly as py 
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import Simulate as sim


def vizualize_UV(UV_space):
    #\\FIXME Most likely some recording of the UV space is incorrect as there is no dynamic change in the UV space 
    t=0
    df = pd.DataFrame(UV_space[0][t], columns=['U'])
    df['V'] = UV_space[1][t]

    fig = px.scatter(df, x="U", y="V")
    fig = px.density_contour(df, x="U", y="V", marginal_x="histogram", marginal_y="histogram")
    #fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    fig.show()


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


