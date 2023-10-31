### Main.py
# Specifies which experiments to run and which vizualizations to make
###


import Simulate as sim
import Vizualize as viz
from IPython.display import display
import utils
import Experiments as exp
import os
import pandas as pd

import networkx as nx

params = utils.get_config()

def main():
    # Run the following experiments

    
    if not os.path.isfile(f"data/df_measures_over_timesteps_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx"):
        exp.mean_measures_per_timestep(params.p_rewiring, params.alpha, params.beta) 
    df_measures_over_timesteps = pd.read_excel(f"data/df_measures_over_timesteps_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx")

    
    if not os.path.isfile(f"data/df_influence_rewiring_prob_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx"):
        exp.effect_of_rewiring_p_on_variance_and_clustering(params.alpha, params.beta) 
    df_influence_rewiring_prob= pd.read_excel(f"data/df_influence_rewiring_prob_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx")

    
    if not os.path.isfile(f"data/df_influence_triangle_prob_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx"):
        exp.effect_of_triangle_prob_on_variance_and_clustering(params.p_rewiring, params.alpha, params.beta)
    df_influence_triangle_prob = pd.read_excel(f"data/df_influence_triangle_prob_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx")


    
    if not os.path.isfile(f"data/df_influence_alpha_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx"):
        exp.effect_of_alpha_beta_on_variance_and_clustering(params.p_rewiring, params.alpha, params.beta)
    df_influence_alpha = pd.read_excel(f"data/df_influence_alpha_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx")
    df_influence_beta = pd.read_excel(f"data/df_influence_beta_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx")
    


    # Vizualize data 
    viz.network_measures_over_timesteps(df_measures_over_timesteps)
    viz.effect_variable_on_network_measures(df_influence_rewiring_prob, 'rewiring_p')
    viz.effect_variable_on_network_measures(df_influence_triangle_prob, 'triangle_p')
    viz.effect_variable_on_network_measures(df_influence_alpha, 'alpha')
    viz.effect_variable_on_network_measures(df_influence_beta, 'beta')
    


    """
    network_data, agent_data = sim.simulate(params.n_agents, params.p_rewiring, params.alpha, params.beta, network = ('RR',4,1), rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,1,2))
    viz.viz_UV(network_data)
    """

if __name__ == "__main__":
    main()
