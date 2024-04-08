'''
Main.py
Specifies which experiments to run and which vizualizations to make
'''

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

    print(f"Start Experiment: {experiment_function.__name__}")

    try:
        experiment_function()
        print(f"Experiment {experiment_function.__name__} completed successfully.\n")
    except Exception as e:
        print(f"Error in {experiment_function.__name__}: {e}\n")

def main(): 

    "Relevant experiment to put in thesis"

    # Game Choice experiments
    run_experiment(expGC.effect_of_risk_distribution_on_wealth)
    run_experiment(expGC.baselineExperiment_without_normalization)
    run_experiment(expGC.baselineExperiments)
    run_experiment(expGC.effect_of_risk_and_rationality_on_QRE)
    run_experiment(expGC.gini_over_time)
    
    # Experiments on Updating Mechanisms
    run_experiment(expU.speedOfUpdatingRewiring_vs_UV)
    
    # Network Experiments
    run_experiment(expN.run_ofat_network)
    run_experiment(expN.run_default_data)
    run_experiment(expN.time_series_mean_network_measures)

    # # Run global sensitivity analysis
    run_experiment(GSA.global_sensitivity_analysis)





    
    
     

if __name__ == "__main__":
    main()
