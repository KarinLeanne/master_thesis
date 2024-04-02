'''
Experiments.py
Contains the different experiments that can be run with the models updating mechansism
'''

import numpy as np
import Simulate as sim
import Experiments.vizUpdating as viz
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st
import os
import utils
import ast

import Experiments.vizUpdating as viz

params = utils.get_config()

def speedOfUpdatingRewiring_vs_UV():
    '''
    Description: 
    Simulate and visualize the speed of updating and rewiring mechanism versus the UV parameter.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    rewiring_p = np.linspace(0.0, 1.0, 5)

    path = utils.make_path("Data", "UpdatingMechanism", "SpeedUVvsNetwork")
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

        for p in rewiring_p:
            df_network, _ = sim.simulate(rewiring_p=p)
            df_network['rewiring_p'] = p
            

            # Add partial data to the dataframe
            df = pd.concat([df_network, df])
        df["Number of Unique Games"] = df['Unique Games'].apply(lambda x: len(x))
        df["Speed_Ratio"] = df["e_g"] / df["e_n"]
        df.to_csv(path, index=False)

    # Replace spaces with underscores in column names
    df.columns = df.columns.str.replace(' ', '_')

    # Vizualize
    viz.viz_time_series_y_varying_rewiring_p(df, rewiring_p, "Speed_Ratio")
    viz.viz_time_series_y_varying_rewiring_p(df, rewiring_p, "Gini_Coefficient")
    viz.viz_time_series_y_varying_rewiring_p(df, rewiring_p, "Number_of_Unique_Games")


    
