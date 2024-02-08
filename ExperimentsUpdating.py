### Experiments.py
# Contains the different experiments that can be run with the models updating mechansism
###

import numpy as np
import Simulate as sim
import Vizualize as viz
import pandas as pd
from tabulate import tabulate
from IPython.display import display
import scipy.stats as st
import os
import utils
import ast


params = utils.get_config()

def speedOfUpdatingRewiring_vs_UV():
    rewiring_p = np.linspace(0.0, 1.0, 11)

    path = utils.make_path("Data", "UpdatingMechanism", "SpeedUVvsNetwork")
    if os.path.isfile(path):
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame()

        for p in rewiring_p:
            df_network, _ = sim.simulate(rewiring_p=p)
            df_network['rewiring_p'] = p
            

            # Add partial data to the dataframe
            df = pd.concat([df_network, df])
        df["Number of Unique Games"] = df['Unique Games'].apply(lambda x: len(x))
        df["Speed_Ratio"] = df["e_g"] / df["e_n"]
        df.to_excel(path, index=False)

    # Replace spaces with underscores in column names
    df.columns = df.columns.str.replace(' ', '_')

    # Vizualize
    viz.plot_speed_network_vs_speed_games(df, "Gini_Coefficient")
    viz.plot_speed_network_vs_speed_games(df, "Number_of_Unique_Games")
    viz.plot_speed_ratio_time_series(df, rewiring_p)

    
