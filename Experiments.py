import numpy as np
import Simulate as sim
import Vizualize as viz
import utils
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st


params = utils.get_config()

def mean_measures_per_timestep():

    # Create dataframe in which data for all network will be stored
    df_model_full = pd.DataFrame()
    df_data_mean_std = pd.DataFrame()

    for network in params.networks:
        # Data_model_partial contains data from one network
        df_model_partial, _ = sim.simulate(N = params.n_agents, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=(network,4,1))
        # Add column specifying which network the data is for
        df_model_partial['Network'] = network
        # Add partial data to the network that needsa to contain the full data
        df_model_full = pd.concat([df_model_full, df_model_partial])

    df_data_mean_std = df_model_full.groupby(["Step", "Network"])[["Mean Degree", "Var of Degree", "Avg Clustering"]].mean().reset_index()
    measures = list(df_model_full.columns)[1:-1]
    for measure in measures:
        df_data_mean_std[f"STD {measure}"] = df_model_full.groupby(["Step", "Network"])[measure].std().reset_index()[measure]

    #print(tabulate(df_data_mean_std, headers = 'keys', tablefmt = 'psql'))
    return df_data_mean_std




def effect_success_probability():

    # Create dataframe in which data for all network will be stored
    df_agent_full = pd.DataFrame()

    for network in params.networks:
        # Data_model_partial contains data from one network
        _, df_agent_partial = sim.simulate(N = params.n_agents, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=(network,4,1))
        # Add column specifying which network the data is for
        df_agent_partial['Network'] = network
        # Add partial data to the network that needsa to contain the full data
        df_agent_full = pd.concat([df_agent_full, df_agent_partial])

    print(tabulate(df_agent_full, headers = 'keys', tablefmt = 'psql'))

    


def effect_clustering_coef_on_topology():
    mean_degree = []
    variance = []
    clustering = []
    
    probs = np.linspace(0.0, 1.0, 1)
    for p in probs:
        data_model, _ = sim.simulate(N = params.n_agents, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=("HK",4,p))
        
    
        mean_degree.append(data_model.iloc[-1]["Mean Degree"])
        variance.append(data_model.iloc[-1]["Var of Degree"])
        clustering.append(data_model.iloc[-1]["Avg Clustering"])

    print(probs)

    viz.viz_effect_clustering_coef("Mean degree", probs, mean_degree)
    viz.viz_effect_clustering_coef("Variance", probs, variance)
    viz.viz_effect_clustering_coef("Clustering", probs, clustering)
    

        