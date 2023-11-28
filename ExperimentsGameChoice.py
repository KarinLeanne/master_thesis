import numpy as np
import pandas as pd
import utils
import Simulate as sim

params = utils.get_config()

"""
def measures_per_timestep(measures_of_interest = [], networks = params.networks):

    # Create dataframe in which data for all network will be stored
    df_model_full = pd.DataFrame()
    df_data_mean_std = pd.DataFrame()

    for network in networks:
        # Data_model_partial contains data from one network
        _, df_agent_data_partial = sim.simulate(N = params.n_agents, rewiring_p = params.rewiring_p, alpha = params.alpha, beta = params.beta, network = params.network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2))
        # Add column specifying which network the data is for
        df_model_partial['Network'] = network[0]
        # Add partial data to the network that needsa to contain the full data
        df_model_full = pd.concat([df_model_full, df_model_partial])


def effect_rationality_on_UV_space():
    df_rewiring_p = pd.DataFrame()
    netRat = np.linspace(0, 1, 8)
"""