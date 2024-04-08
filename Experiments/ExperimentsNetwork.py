'''
ExperimentsNetwork.py
Contains the different experiment that can be run with the model on the network dynamics
'''

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
import Experiments.StatisticsNetworks as stN

params = utils.get_config()

def mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest = ["M: Mean Degree", "M: Var of Degree", "M: Avg Clustering", "M: Avg Path Length"], networks = params.networks):
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
        df_rewiring_p = pd.read_csv(path)
    else:
        df_rewiring_p = pd.DataFrame()
        rewiring_ps = np.linspace(0.0, 1.0, 5)

        for rewiring_p in rewiring_ps:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering", "M: Avg Path Length", "Gini Coefficient"])
            df['rewiring_p'] = rewiring_p
            # Add partial data to the network that needs to contain the full data
            df_rewiring_p = pd.concat([df_rewiring_p, df])

        df_rewiring_p.to_csv(path, index=False)

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
        df_triangle_prob = pd.read_csv(path)
    else:
        df_triangle_prob = pd.DataFrame()

        triangle_probs = np.linspace(0.0, 1.0, 5)

        for triangle_p in triangle_probs:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering", "M: Avg Path Length"], networks=[("HK", 4, triangle_p)])
            df['triangle_p'] = triangle_p
            # Add partial data to the network that needsa to contain the full data
            df_triangle_prob = pd.concat([df_triangle_prob, df])
        
        df_triangle_prob.to_csv(path, index=False)

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
        df_alpha = pd.read_csv(path_alpha)
        df_beta = pd.read_csv(path_beta)
    else:
        df_alpha = pd.DataFrame()
        alphas = np.linspace(0.0, 1.0, 5)
        for alpha in alphas:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering", "M: Avg Path Length"])
            df['alpha'] = alpha
            # Add partial data to the network that needsa to contain the full data
            df_alpha = pd.concat([df_alpha, df])

        df_beta = pd.DataFrame()    
        betas= np.linspace(0.0, 1.0, 5)

        for beta in betas:
            # Data_model_partial contains data from one network
            df = mean_measures_per_timestep(rewiring_p, alpha, beta, measures_of_interest= ["M: Var of Degree", "M: Avg Clustering", "M: Avg Path Length"])
            df['beta'] = beta
            # Add partial data to the network that needsa to contain the full data
            df_beta = pd.concat([df_beta, df])

        df_alpha.to_csv(path_alpha, index=False)
        df_beta.to_csv(path_beta, index=False)

    # Vizualize results experiment
    viz.viz_effect_variable_on_network_measures(df_alpha, 'alpha')
    viz.viz_effect_variable_on_network_measures(df_beta, 'beta')

def time_series_mean_network_measures():
    '''
    Description: 
    Calculate and visualize time series of mean network measures over timesteps.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''

    # Only obtain data if data does not already exist
    path = utils.make_path("Data", "Networks", "Time_Series_Mean_Network_Measures")
    if os.path.isfile(path):
        df_measures_over_timesteps = pd.read_csv(path)
    else:
        df_measures_over_timesteps = mean_measures_per_timestep(params.rewiring_p, params.alpha, params.beta) 
        df_measures_over_timesteps.to_csv(path, index=False)
    

    # Vizualize results experiment
    viz.viz_network_measures_over_timesteps(df_measures_over_timesteps)

    # Statistics
    stN.adf_test_for_combinations(df_measures_over_timesteps)
    stN.sliding_window_rmsd(df_measures_over_timesteps)
    

def run_default_data():
    '''
    Description: 
    Run default simulation and visualization for network data.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    # Only obtain data if data does not already exist
    path_agent = utils.make_path("Data", "Networks", "Default_Sim_Agent")
    path_model = utils.make_path("Data", "Networks", "Default_Sim_Model")
    if os.path.isfile(path_agent) and os.path.isfile(path_model):
        df_agent = pd.read_csv(path_agent)
        df_model = pd.read_csv(path_model)
    else:
        df_model, df_agent = sim.simulate(N = params.n_agents, rewiring_p = params.rewiring_p, alpha=params.alpha, beta=params.beta, network=params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False)
        df_agent.to_csv(path_agent, index=False)
        df_model.to_csv(path_model, index=False)

    # Vizualize
    viz.viz_histogram_over_time(df_agent, "Games played", bins=40)
    viz.viz_Degree_Distr(df_model, "Degree Distr", bins=10)
    viz.analyze_wealth_distribution_by_game_group(df_agent)

    # Statistical analysis
    stN.compare_distributions(df_agent, "Wealth")
    stN.compare_distributions(df_agent, "Recent Wealth")

def run_ofat_network():
    '''
    Description: 
    Run One-Factor-At-a-Time (OFAT) analysis for network simulation data and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    path_ofat_model = utils.make_path("Data", "Networks", "df_ofat_model")
    path_ofat_agent = utils.make_path("Data", "Networks", "df_ofat_agent")

    if (os.path.isfile(path_ofat_agent) and os.path.isfile(path_ofat_model)):
        df_ofat_agent = pd.read_csv(path_ofat_agent)
        df_ofat_model = pd.read_csv(path_ofat_model)
    else:
        # Define model reporters
        model_reporters = {
            "M: Mean Degree": lambda m: m.get_mean_degree(),
            "M: Var of Degree": lambda m: m.get_variance_degree(),
            "M: Avg Clustering": lambda m: m.get_clustering_coef(),
            "M: Avg Path Length": lambda m: m.get_average_path_length(),
            "Gini Coefficient": lambda m: m.get_gini_coef()
            }

        agent_reporters = {
            "Wealth": "wealth", 
            "Player Risk Aversion": "eta",
            "Recent Wealth": "recent_wealth"
            }
        
        # Define the problem
        distinct_samples = 6 
        problem = {
        'num_vars': 3,
        'names': ['rewiring_p', 'alpha', 'beta'],
        'bounds': [[0, 1], [0, 1], [0, 1]]
        }

        # Run OFAT for all reporters
        df_ofat_model, df_ofat_agent = OFAT.ofat(problem, distinct_samples, model_reporters=model_reporters, agent_reporters=agent_reporters)

        # Save the results to respective paths
        df_ofat_model.to_csv(path_ofat_model, index=False)
        df_ofat_agent.to_csv(path_ofat_agent, index=False)


    OFAT.plot_network_measures(df_ofat_model, "rewiring_p", "Rewiring Pr", "Networks")
    OFAT.plot_network_measures(df_ofat_model, "alpha", "alpha", "Networks")
    OFAT.plot_network_measures(df_ofat_model, "beta", "beta", "Networks")
    OFAT.plot_ofat_wealth_measures('Networks', df_ofat_model, "Gini Coefficient", ['alpha', 'beta'])
    OFAT.plot_ofat_wealth_measures('Networks', df_ofat_model, "Gini Coefficient", ['rewiring_p'], ["Rewiring Pr"])


    stN.kruskal_wallis_test(df_ofat_model, "rewiring_p", "M: Avg Clustering")
    stN.kruskal_wallis_test(df_ofat_model, "rewiring_p", "M: Var of Degree")
    stN.kruskal_wallis_test(df_ofat_model, "rewiring_p", "M: Avg Path Length")
    stN.kruskal_wallis_test(df_ofat_model, "rewiring_p", "Gini Coefficient")

    stN.kruskal_wallis_test(df_ofat_model, "alpha", "M: Avg Clustering")
    stN.kruskal_wallis_test(df_ofat_model, "alpha", "M: Var of Degree")
    stN.kruskal_wallis_test(df_ofat_model, "alpha", "M: Avg Path Length")
    stN.kruskal_wallis_test(df_ofat_model, "alpha", "Gini Coefficient")

    stN.kruskal_wallis_test(df_ofat_model, "beta", "M: Avg Clustering")
    stN.kruskal_wallis_test(df_ofat_model, "beta", "M: Var of Degree")
    stN.kruskal_wallis_test(df_ofat_model, "beta", "M: Avg Path Length")
    stN.kruskal_wallis_test(df_ofat_model, "beta", "Gini Coefficient")











        