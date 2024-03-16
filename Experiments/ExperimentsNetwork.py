### Experiments.py
# Contains the different experiment that can be run with the model
###

import numpy as np
import Simulate as sim
import utils
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st
import os
import Experiments.OFAT as OFAT

import Experiments.vizNetworks as viz

params = utils.get_config()

def mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest = ["M: Mean Degree", "M: Var of Degree", "M: Avg Clustering"], networks = params.networks):
    '''
    Description: Runs the model simulation for alle specified network and adds the collected data to a dataframe
    Input: 
        - rewiring_p: The probability that an agent rewires a connection for each timestep 
        - alpha: The homophilic parameter 
        - beta: Controls homophily together with alpha
        - measures_of_interest: The data measures to collect in the dataframe
        - networks: The networks for which to run the simulation
    Output: A dataframe containing the sepcified results of the simulations
    '''
    # Create dataframe in which data for all network will be stored
    df_model_full = pd.DataFrame()
    df_data_mean_std = pd.DataFrame()

    for network in networks:
        # Data_model_partial contains data from one network
        df_model_partial, _ = sim.simulate(N = params.n_agents, rewiring_p = rewiring_p, alpha=alpha, beta=beta, network=network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False)
        # Add column specifying which network the data is for
        df_model_partial['Network'] = network[0]
        # Add partial data to the network that needsa to contain the full data
        df_model_full = pd.concat([df_model_full, df_model_partial])

    #print(tabulate(df_model_full, headers = 'keys', tablefmt = 'psql'))
    df_data_mean_std = df_model_full.groupby(["Step", "Network"])[measures_of_interest].mean().reset_index()
    
    measures = list(df_data_mean_std.loc[:, df_data_mean_std.columns.str.contains('M:')].columns)
    for measure in measures:
        df_data_mean_std[f"STD:{measure.replace('M:', '')}"] = df_model_full.groupby(["Step", "Network"])[measure].std().reset_index()[measure]
    return df_data_mean_std



def effect_of_rewiring_p_on_variance_and_clustering(alpha = params.alpha, beta = params.beta):
    '''
    Description: Experiment with the effect of different values of the rewiring probability on the variance and clustering of the network. Mean should be maintained.
    Input: 
        - alpha: The homophilic parameter 
        - beta: Controls homophily together with alpha
    '''
    path = utils.make_path("Data", "Networks", "Influence_Rewiring_p_On_Network")
    if os.path.isfile(path):
        df_rewiring_p = pd.read_excel(path)
    else:
        df_rewiring_p = pd.DataFrame()
        rewiring_ps = np.linspace(0.0, 1.0, 5)

        for rewiring_p in rewiring_ps:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering", "Gini Coefficient"])
            df['rewiring_p'] = rewiring_p
            # Add partial data to the network that needs to contain the full data
            df_rewiring_p = pd.concat([df_rewiring_p, df])

        #print(tabulate(df_rewiring_p, headers = 'keys', tablefmt = 'psql'))

        df_rewiring_p.to_excel(path, index=False)

    # Vizualize results of experiment
    viz.viz_effect_variable_on_network_measures(df_rewiring_p, 'rewiring_p')
    


def effect_of_triangle_prob_on_variance_and_clustering(rewiring_p = params.rewiring_p, alpha = params.alpha, beta = params.beta):
    '''
    Description: Experiment with the effect of different values of the triangle prob on the variance and clustering of the network. 
    Input: 
        - rewiring_p: The probability that an agent rewires a connection for each timestep 
        - alpha: The homophilic parameter 
        - beta: Controls homophily together with alpha
    '''
    path = utils.make_path("Data", "Networks", "Influence_Triangle_p_On_Network")
    if os.path.isfile(path):
        df_triangle_prob = pd.read_excel(path)
    else:
        df_triangle_prob = pd.DataFrame()

        triangle_probs = np.linspace(0.0, 1.0, 5)

        for triangle_p in triangle_probs:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering"], networks=[("HK", 4, triangle_p)])
            df['triangle_p'] = triangle_p
            # Add partial data to the network that needsa to contain the full data
            df_triangle_prob = pd.concat([df_triangle_prob, df])
        
        df_triangle_prob.to_excel(path, index=False)

    # Vizualize results of experiment
    viz.viz_effect_variable_on_network_measures(df_triangle_prob, 'triangle_p')
        

def effect_of_alpha_beta_on_variance_and_clustering(rewiring_p = params.rewiring_p, alpha = params.alpha, beta = params.beta):
    '''
    Description: Experiment with the effect alpha and beta on the variance and clustering of the network. 
    Input: 
        - rewiring_p: The probability that an agent rewires a connection for each timestep 
        - alpha: The default homophilic parameter 
        - beta: Controls homophily together with alpha, this is the default value
    '''
    path_alpha = utils.make_path("Data", "Networks", "Influence_alpha_On_Network")
    path_beta = utils.make_path("Data", "Networks", "Influence_beta_On_Network")

    if os.path.isfile(path_alpha):
        df_alpha = pd.read_excel(path_alpha)
        df_beta = pd.read_excel(path_beta)
    else:
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

        df_alpha.to_excel(path_alpha, index=False)
        df_beta.to_excel(path_beta, index=False)

    # Vizualize results experiment
    viz.viz_effect_variable_on_network_measures(df_alpha, 'alpha')
    viz.viz_effect_variable_on_network_measures(df_beta, 'beta')

def time_series_mean_network_measures():

    # Only obtain data if data does not already exist
    path = utils.make_path("Data", "Networks", "Time_Series_Mean_Network_Measures")
    if os.path.isfile(path):
        df_measures_over_timesteps = pd.read_excel(path)
    else:
        df_measures_over_timesteps = mean_measures_per_timestep(params.rewiring_p, params.alpha, params.beta) 
        df_measures_over_timesteps.to_excel(path, index=False)
    

    # Vizualize results experiment
    viz.viz_network_measures_over_timesteps(df_measures_over_timesteps)


def run_default_data():
    # Only obtain data if data does not already exist
    path_agent = utils.make_path("Data", "Networks", "Default_Sim_Agent")
    path_model = utils.make_path("Data", "Networks", "Default_Sim_Model")
    if os.path.isfile(path_agent) and os.path.isfile(path_model):
        df_agent = pd.read_excel(path_agent)
        df_model = pd.read_excel(path_model)
    else:
        df_model, df_agent = sim.simulate(N = params.n_agents, rewiring_p = params.rewiring_p, alpha=params.alpha, beta=params.beta, network=params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False)
        df_agent.to_excel(path_agent, index=False)
        df_model.to_excel(path_model, index=False)

    viz.viz_histogram_over_time(df_agent, "Games played", bins=40)
    viz.viz_Degree_Distr(df_model, "Degree Distr", bins=40)

def run_ofat_network():
    path_gini = utils.make_path("Data", "Networks", "ofat_gini")
    path_wealth = utils.make_path("Data", "Networks", "ofat_wealth")
    path_risk_aversion = utils.make_path("Data", "Networks", "ofat_risk_aversion")

    # For Gini Coefficient (model-level)
    if os.path.isfile(path_gini):
        df_ofat_gini = pd.read_excel(path_gini)
    else:
        gini_reporters = {"Gini Coefficient": lambda m: m.get_gini_coef()}
        df_ofat_gini = OFAT.ofat(model_reporters = gini_reporters, level='model')
        df_ofat_gini.to_excel(path_gini, index=False)

    # For Wealth (agent-level)
    if os.path.isfile(path_wealth):
        df_ofat_wealth = pd.read_excel(path_wealth)
    else:
        wealth_reporters = {"Wealth": "wealth"}
        df_ofat_wealth = OFAT.ofat(agent_reporters = wealth_reporters, level='agent')
        df_ofat_wealth.to_excel(path_wealth, index=False)

    # For Player risk aversion (agent-level)
    if os.path.isfile(path_risk_aversion):
        df_ofat_risk_aversion = pd.read_excel(path_risk_aversion)
    else:
        risk_reporters = {"Player risk aversion": "eta"}
        df_ofat_risk_aversion = OFAT.ofat(agent_reporters = risk_reporters, level='agent')
        df_ofat_risk_aversion.to_excel(path_risk_aversion, index=False)

    # Vizualize the ofat
    OFAT.plot_vs_independent('Networks', df_ofat_gini, "Gini Coefficient")
    OFAT.plot_vs_independent('Networks', df_ofat_wealth, "Wealth")
    OFAT.plot_vs_independent('Networks', df_ofat_risk_aversion, "Player risk aversion")






        