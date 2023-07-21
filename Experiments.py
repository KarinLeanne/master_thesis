import numpy as np
import Simulate as sim
import Vizualize as viz
import utils
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st


params = utils.get_config()

def mean_measures_per_timestep(p_rewiring, alpha, beta, measures_of_interest = ["M: Mean Degree", "M: Var of Degree", "M: Avg Clustering"], networks = params.networks):

    # Create dataframe in which data for all network will be stored
    df_model_full = pd.DataFrame()
    df_data_mean_std = pd.DataFrame()

    for network in networks:
        # Data_model_partial contains data from one network
        df_model_partial, df_agent_partial = sim.simulate(N = params.n_agents, p_rewiring = p_rewiring, alpha=alpha, beta=beta, network=network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2))
        # Add column specifying which network the data is for
        df_model_partial['Network'] = network[0]
        # Add partial data to the network that needsa to contain the full data
        df_model_full = pd.concat([df_model_full, df_model_partial])

    

    df_data_mean_std = df_model_full.groupby(["Step", "Network"])[measures_of_interest].mean().reset_index()

    
    measures = list(df_data_mean_std.loc[:, df_data_mean_std.columns.str.contains('M:')].columns)
    for measure in measures:
        df_data_mean_std[f"STD:{measure.replace('M:', '')}"] = df_model_full.groupby(["Step", "Network"])[measure].std().reset_index()[measure]

    #print(tabulate(df_data_mean_std, headers = 'keys', tablefmt = 'psql'))
    return df_data_mean_std


def effect_of_rewiring_p_on_variance_and_clustering(alpha, beta):
    df_rewiring_p = pd.DataFrame()
    rewiring_ps = np.linspace(0.0, 1.0, 5)

    for rewiring_p in rewiring_ps:
        # Data_model_partial contains data from one network
        df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering"])
        df['rewiring_p'] = rewiring_p
        # Add partial data to the network that needsa to contain the full data
        df_rewiring_p = pd.concat([df_rewiring_p, df])

    #print(tabulate(df_rewiring_p, headers = 'keys', tablefmt = 'psql'))

    return df_rewiring_p


def effect_of_triangle_prob_on_variance_and_clustering(rewiring_p, alpha, beta):
    df_triangle_prob = pd.DataFrame()

    triangle_probs = np.linspace(0.0, 1.0, 5)

    for triangle_p in triangle_probs:
         # Data_model_partial contains data from one network
        df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering"], networks=[("HK", 4, triangle_p)])
        df['triangle_p'] = triangle_p
        # Add partial data to the network that needsa to contain the full data
        df_triangle_prob = pd.concat([df_triangle_prob, df])
    return df_triangle_prob
        

def effect_of_alpha_beta_on_variance_and_clustering(rewiring_p, alpha, beta):
    
    df_alpha = pd.DataFrame()
    alphas = np.linspace(0.0, 1.0, 5)

    for alpha in alphas:
        # Data_model_partial contains data from one network
        df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering"])
        df['alpha'] = alpha
        # Add partial data to the network that needsa to contain the full data
        df_alpha = pd.concat([df_alpha, df])

    df_beta = pd.DataFrame()    
    betas= np.linspace(0.0, 1.0, 5)

    for beta in betas:
        # Data_model_partial contains data from one network
        df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering"])
        df['beta'] = beta
        # Add partial data to the network that needsa to contain the full data
        df_beta = pd.concat([df_beta, df])

    return df_alpha, df_beta


        