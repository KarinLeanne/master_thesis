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
    df_influence_succes_prob = exp.effect_success_probability()
    

    # Vizualize data 
    #viz.measures_over_timesteps(df_measures_over_timesteps)

if __name__ == "__main__":
    main()
