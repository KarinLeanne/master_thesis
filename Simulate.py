'''
Simulate.py
This file contains a function simulate that runs a simulation of a model with given parameters, 
collecting data at both network and agent levels.
'''

import pandas as pd
from tabulate import tabulate

import GamesModel as gm
import utils

params = utils.get_config()

def simulate(N = params.n_agents, 
             rewiring_p = params.rewiring_p, 
             alpha = params.alpha, 
             beta = params.beta, 
             rat = params.rat,
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
        print("Round:", round)
        model = gm.GamesModel(
                 N = N, 
                 rewiring_p = rewiring_p, 
                 alpha = alpha, 
                 beta = beta, 
                 rat = rat,
                 network = network, 
                 partScaleFree = partScaleFree, 
                 alwaysOwn = alwaysOwn, 
                 UV = (True, None, None, False), 
                 risk_distribution = risk_distribution, 
                 utility_function = utility_function)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        model_data = pd.concat([model_data, model.datacollector.get_model_vars_dataframe()])
        model_data['Round'] = model_data['Round'].fillna(round)
    

    # Split the MultiIndex into separate columns for agent data
    agent_data.reset_index(inplace=True)
    # Subtract 1 from the first element of each tuple in the "index" column
    agent_data[['Step', 'Players']] = pd.DataFrame(agent_data['index'].to_list(), index=agent_data.index)

    # Reorder the columns with 'steps' and 'players' immediately after the index for agnet data
    index_columns = ['Step', 'Players']
    agent_data = agent_data[index_columns + [col for col in agent_data.columns if col not in index_columns]]

    # Drop the original 'index' column for agent data
    agent_data.drop(columns=['index'], inplace=True)
    
    # Subtract 1 from each value in the "Step" column
    agent_data['Step'] = agent_data['Step'] - 1

    # For network data, reset the index and rename the index column to "step"
    model_data.reset_index(inplace=True)
    model_data.rename(columns={"index": "Step"}, inplace=True)



    

    return model_data, agent_data
