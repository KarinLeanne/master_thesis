### Main.py
# Specifies which experiments to run and which vizualizations to make
###

from IPython.display import display
import os
import pandas as pd
import networkx as nx

import utils
import Experiments.ExperimentsNetwork as expN
import Experiments.ExperimentsGameChoice as expGC
import Experiments.ExperimentsUpdating as expU
import Experiments.GSA as GSA
import Experiments.OFAT as OFAT

params = utils.get_config()

def run_experiment(experiment_function):
    try:
        experiment_function()
    except Exception as e:
        print(f"Error in {experiment_function.__name__}: {e}")




def main():   

    # Experiments on Gamechoice
    run_experiment(expGC.baselineExperiments)
    run_experiment(expGC.baselineExperiments_NH)
    run_experiment(expGC.effect_of_risk_distribution_on_wealth)
    run_experiment(expGC.effect_of_utility_function_on_wealth)
    run_experiment(expGC.effect_of_rationality_on_QRE)
    run_experiment(expGC.track_num_games_in_pop)
    run_experiment(expGC.gini_over_time)
    run_experiment(expGC.run_ofat_GC)

    # Experiments on Updating Mechanisms
    run_experiment(expU.speedOfUpdatingRewiring_vs_UV)

    # Experiments on Network
    run_experiment(expN.run_default_data)
    run_experiment(expN.time_series_mean_network_measures)
    run_experiment(expN.effect_of_alpha_beta_on_variance_and_clustering)
    run_experiment(expN.effect_of_rewiring_p_on_variance_and_clustering)
    run_experiment(expN.effect_of_triangle_prob_on_variance_and_clustering)
    run_experiment(expN.run_ofat_network)

    # Run global sensitivity analysis
    run_experiment(GSA.global_sensitivity_analysis)

    
    



    
    
    

if __name__ == "__main__":
    main()
