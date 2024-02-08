### Simulate.py
# Runs the model for a certain number of rounds and steps using the given parameters
###

import numpy as np
import pandas as pd
from tabulate import tabulate
#from IPython.display import display

import GamesModel as gm
import utils

params = utils.get_config()

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

def simulate(N = params.n_agents, 
             rewiring_p = params.rewiring_p, 
             alpha = params.alpha, 
             beta = params.beta, 
             network = params.default_network, 
             rounds = params.n_rounds, 
             steps = params.n_steps, 
             netRat = 0.1, 
             partScaleFree = 1, 
             alwaysOwn = False, 
             alwaysSafe = False, 
             UV=(True,None,None,False), 
             risk_distribution = "uniform", 
             utility_function = "isoelastic"):
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
        - model_data: The data collected in network level
        - agent_data: The data collected on agent level
    '''

    #agent_data = pd.DataFrame()
    model_data = pd.DataFrame(columns=['Round'])
    agent_data = pd.DataFrame(columns=['Round'])

    for round in range(rounds):
        print("round", round)
        model = gm.GamesModel(N, rewiring_p, alpha, beta, network, netRat, partScaleFree, alwaysOwn, UV, risk_distribution, utility_function)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        model_data = pd.concat([model_data, model.datacollector.get_model_vars_dataframe()])
        model_data['Round'] = model_data['Round'].fillna(round)
    

    # Split the MultiIndex into separate columns for agent data
    agent_data.reset_index(inplace=True)
    agent_data[['Step', 'Players']] = pd.DataFrame(agent_data['index'].to_list(), index=agent_data.index)

    # Reorder the columns with 'steps' and 'players' immediately after the index for agnet data
    index_columns = ['Step', 'Players']
    agent_data = agent_data[index_columns + [col for col in agent_data.columns if col not in index_columns]]

    # Drop the original 'index' column for agent data
    agent_data.drop(columns=['index'], inplace=True)

    #agent_data['UV'] = agent_data['UV'].apply(ast.literal_eval)

    #print(tabulate(agent_data, headers = 'keys', tablefmt = 'psql'))
    
    # For network data, reset the index and rename the index column to "step"
    model_data.reset_index(inplace=True)
    model_data.rename(columns={"index": "Step"}, inplace=True)
    #model_data['Unique Games'] = model_data['Unique Games'].apply(lambda x: eval(x))
    #print(model_data["Unique Games"][0])
    #print(tabulate(model_data, headers = 'keys', tablefmt = 'psql'))


    

    return model_data, agent_data


