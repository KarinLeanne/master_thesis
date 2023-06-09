import numpy as np
import pandas as pd
import itertools
from IPython.display import display
import GamesModel as gm
from tabulate import tabulate

def gini_new(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    array = array.flatten() #all values are treated equally, arrays must be 1d
    #make array into floats
    array = array.astype(float)
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    #array = np.sum(array, 0.0000001, dtype=np.float)
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements

    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 

def simulate(rewiring_p, alpha, beta, N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None), network=('RR',4,1)):

    #agent_data = pd.DataFrame()
    network_data = pd.DataFrame(columns=['Round'])
    agent_data = pd.DataFrame(columns=['Round'])

    for round in range(rounds):
        print("round", round)
        model = gm.GamesModel(N, alpha, beta, netRat, partScaleFree, alwaysSafe,UV,network)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        network_data = pd.concat([network_data, model.datacollector.get_model_vars_dataframe()])
        network_data['Round'] = network_data['Round'].fillna(round)

    network_data.index.name = 'Step'
    return network_data, agent_data


