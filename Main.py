### Main.py
# Specifies which experiments to run and which vizualizations to make
###


from IPython.display import display
import os
import pandas as pd
import networkx as nx


import utils
import ExperimentsNetwork as expN
import ExperimentsGameChoice as expGC
import ExperimentsUpdating as expU
import Sobol
import OFAT

params = utils.get_config()

def main():

    
    # Experiments on Gamechoice
    #expGC.baselineExperiments()
    #expGC.baselineExperiments_NH()
    #expGC.effect_of_risk_distribution_on_wealth()
    #expGC.effect_of_utility_function_on_wealth()
    #expGC.effect_of_rationality_on_QRE()
    #expGC.track_num_games_in_pop()
    #expGC.gini_over_time()
    

    # Experiments on Updating Mechanisms
    #expU.speedOfUpdatingRewiring_vs_UV()

    # Experiments on Network
    #expN.run_default_data()
    #expN.time_series_mean_network_measures()
    #expN.effect_of_alpha_beta_on_variance_and_clustering()
    #expN.effect_of_rewiring_p_on_variance_and_clustering()
    #expN.effect_of_triangle_prob_on_variance_and_clustering()\
    expN.run_ofat_network()

  

    
    


    # Run global sensitivity analysis
    #Sobol.global_sensitivity_analysis()






    
    
    

if __name__ == "__main__":
    main()
