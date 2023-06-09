import Simulate as sim
import Vizualize as viz
from IPython.display import display
import utils
import Experiments as exp

import networkx as nx

params = utils.get_config()

def main():
    # Run the following experiments
    #df_measures_over_timesteps = exp.mean_measures_per_timestep()
    #df_influence_succes_prob = exp.effect_success_probability()
    df_influence_rewiring_prob = exp.effect_of_rewiring_prob_on_mean_degree()
    

    # Vizualize data 
    #viz.measures_over_timesteps(df_measures_over_timesteps)
    viz.effect_rewiring_prob_network_measures(df_influence_rewiring_prob)

if __name__ == "__main__":
    main()
