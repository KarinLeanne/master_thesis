import Simulate as sim
import Vizualize as viz
from IPython.display import display
import utils
import Experiments as exp

import networkx as nx

params = utils.get_config()

def main():
    # Run the following experiments
    #df_measures_over_timesteps = exp.mean_measures_per_timestep(params.p_rewiring, params.alpha, params.beta)
    #df_influence_rewiring_prob = exp.effect_of_rewiring_p_on_variance_and_clustering(params.alpha, params.beta)
    #df_influence_triangle_prob = exp.effect_of_triangle_prob_on_variance_and_clustering(params.p_rewiring, params.alpha, params.beta)
    df_influence_alpha, df_influence_beta = exp.effect_of_alpha_beta_on_variance_and_clustering(params.p_rewiring, params.alpha, params.beta)
    

    # Vizualize data 
    #viz.network_measures_over_timesteps(df_measures_over_timesteps)
    #viz.effect_variable_on_network_measures(df_influence_rewiring_prob, 'rewiring_p')
    #viz.effect_variable_on_network_measures(df_influence_triangle_prob, 'triangle_p')
    viz.effect_variable_on_network_measures(df_influence_alpha, 'alpha')
    viz.effect_variable_on_network_measures(df_influence_beta, 'beta')
    

if __name__ == "__main__":
    main()
