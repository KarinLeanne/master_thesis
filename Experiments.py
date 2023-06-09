import numpy as np
import Simulate as sim
import Vizualize as viz
import utils
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st


params = utils.get_config()

def mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest = ["Mean Degree", "Var of Degree", "Avg Clustering"]):

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

    df_data_mean_std = df_model_full.groupby(["Step", "Network"])[measures_of_interest].mean().reset_index()


    measures = list(df_data_mean_std.columns)[2:]
    for measure in measures:
        df_data_mean_std[f"STD {measure}"] = df_model_full.groupby(["Step", "Network"])[measure].std().reset_index()[measure]

    #print(tabulate(df_data_mean_std, headers = 'keys', tablefmt = 'psql'))
    return df_data_mean_std


def effect_of_rewiring_prob_on_mean_degree(alpha, beta):
    df_rewiring_prob = pd.DataFrame()
    
    rewiring_probs = np.linspace(0.0, 1.0, 3)

    for rewiring_p in rewiring_probs:
        # Data_model_partial contains data from one network
        df_data_mean_std = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["Mean Degree"])
        df_data_mean_std['Rewiring p'] = rewiring_p
        # Add partial data to the network that needsa to contain the full data
        df_rewiring_prob = pd.concat([df_rewiring_prob, df_data_mean_std])

    return df_rewiring_prob
    



def effect_clustering_coef_on_topology(rewiring_p, alpha, beta, ):
    mean_degree = []
    variance = []
    clustering = []
    
    probs = np.linspace(0.0, 1.0, 1)
    for p in probs:
        data_model, _ = sim.simulate(rewiring_p, alpha, beta, N = params.n_agents, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=("HK",4,p))
        
    
        mean_degree.append(data_model.iloc[-1]["Mean Degree"])
        variance.append(data_model.iloc[-1]["Var of Degree"])
        clustering.append(data_model.iloc[-1]["Avg Clustering"])

    print(probs)

    viz.viz_effect_clustering_coef("Mean degree", probs, mean_degree)
    viz.viz_effect_clustering_coef("Variance", probs, variance)
    viz.viz_effect_clustering_coef("Clustering", probs, clustering)
    

        