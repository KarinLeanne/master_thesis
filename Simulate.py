### Simulate.py
# Runs the model for a certain number of rounds and steps using the given parameters
###

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

def simulate(N, rewiring_p, alpha, beta, network, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, alwaysSafe = False, UV=(True,None,None)):
    '''
    Description: Run the simulation for a certain number of rounds and steps using the given parameters
    Inputs:
        - N: The number of agents in a model
        - rewiring_p: The probability that an agent rewires a connection for each timestep
        - alpha: alpha is the homophilic parameter 
        - beta: beta controls homophily together with alpha
        - network: A truple sepcifying the type of network
        - netRat: A parameter of agent rationality
        - partScaleFree: 
        - AlwaysOwn: Boolean, if true always choose the own strategy
        - UV: A truple specifing whether the UV space should be generated randomly as well as the default values for U and V
    Outputs:
        - network_data: The data collected in network level
        - agent_data: The data collected on agent level
    '''

    #agent_data = pd.DataFrame()
    network_data = pd.DataFrame(columns=['Round'])
    agent_data = pd.DataFrame(columns=['Round'])

    for round in range(rounds):
        #print("round", round)
        
        # Keep track of how edges are made

        Track_edges = [[0,0], [0,0]]

        model = gm.GamesModel(N, rewiring_p, alpha, beta, network, netRat, partScaleFree, alwaysOwn, UV)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        #print(model.datacollector.model_vars)
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        network_data = pd.concat([network_data, model.datacollector.get_model_vars_dataframe()])
        network_data['Round'] = network_data['Round'].fillna(round)
        #print(tabulate(network_data, headers = 'keys', tablefmt = 'psql'))
    
    network_data = network_data.reset_index(drop=False, names="Step")

    return network_data, agent_data


