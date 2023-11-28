### Main.py
# Specifies which experiments to run and which vizualizations to make
###


import Simulate as sim
import Vizualize as viz
from IPython.display import display
import utils
import ExperimentsNetwork as expN
import ExperimentsGameChoice as expGC
import os
import pandas as pd

import networkx as nx

params = utils.get_config()

def main():
    # Run the following experiments for chapter 1
    """
    expN.time_series_mean_network_measures()
    expN.effect_of_alpha_beta_on_variance_and_clustering()
    expN.effect_of_rewiring_p_on_variance_and_clustering()
    expN.effect_of_triangle_prob_on_variance_and_clustering()
    """

    # Run the following experiments for chapter 2
    
    expGC.baselineExperiments()

    

if __name__ == "__main__":
    main()
