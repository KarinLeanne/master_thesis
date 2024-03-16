from chart_studio import plotly as py 
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display
from tabulate import tabulate
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d
import matplotlib.cm as cm
import seaborn as sns
import ast
import json


import utils

params = utils.get_config()


def viz_network_measures_over_timesteps(df):

    #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

    # Get steps, used networks and used measures from dataframe
    steps = np.arange(df['Step'].nunique())
    networks = df['Network'].unique()

    # Only first two columns of dataframe do not contain measures
    mean_measures = list(df.loc[:, df.columns.str.contains('M:')].columns)
    std_measures = list(df.loc[:, df.columns.str.contains('STD:')].columns)

    fig, axs = plt.subplots(len(networks), len(mean_measures), sharex='col', sharey='row')


    # Make plot per measure and networktype
    for mean_measure, std_measure, row in zip(mean_measures, std_measures, range(len(mean_measures))):
        for network, col in zip(networks, range(len(networks))):
            mean = df.loc[df['Network'] == network][mean_measure]
            std = df.loc[df['Network'] == network][std_measure]
            axs[row, col].plot(steps, mean)
            axs[row, col].fill_between(steps, mean+2*std, mean-2*std, alpha = 0.2)

    # Set common x label
    fig.text(0.5, 0.04, 'Step', ha='center')

    # Set y labels
    for ax, row in zip(axs[:,0], mean_measures):
        ax.set_ylabel(row)

    # Make clear which network belongs to which graph
    for ax, col in zip(axs[0,:], networks):
        ax.set_title(col)

    fig.suptitle("Network measures over time")

    path = utils.make_path("Figures", "Networks", "Network_Measures_over_Time")
    plt.savefig(path)
    plt.close()


def viz_effect_variable_on_network_measures(df, variable):


    # Get steps, used networks and used measures from dataframe
    steps = np.arange(df['Step'].nunique())
    networks = df['Network'].unique()
    variable_vals = df[variable].unique()

    # Only first two columns of dataframe do not contain measures
    mean_measures = list(df.loc[:, df.columns.str.contains('M:')].columns)

    for measure in mean_measures:
     
        fig, axs = plt.subplots(len(networks), sharex='col', sharey='col', squeeze=False)
        for row in range(len(networks)):
            df_network = df[df['Network'] == networks[row]]
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(variable_vals))))
            
            for var in variable_vals:
                c = next(color)
                data = df_network[df_network[variable] == var]
                mean = data[measure]
                axs[row, 0].plot(steps, mean, c=c, label=f"{variable} = {var}")
                std = data[measure.replace('M:', 'STD:')]
                axs[row, 0].fill_between(steps, mean+2*std, mean-2*std, alpha = 0.2, color=c)
                
            axs[row, 0].legend()

        # Set common x label
        fig.text(0.5, 0.04, 'Step', ha='center')

        # Set y labels
        for ax in axs:
            ax[0].set_ylabel(measure.replace("M:", ""))

        # Make clear which network belongs to which graph
        for ax, col in zip(axs, networks):
            ax[0].set_title(col)

        # displaying the title
        fig.suptitle(f"Effect of {variable} on {measure.replace('M:', '')}")
        path = utils.make_path("Figures", "Networks", f"Effect_of_{variable}_on_{measure.replace('M:', '')}")
        plt.savefig(path)
        plt.close()




def viz_histogram_over_time(df, measure, bins=10):
    stepsPlot = [*range(int(params.n_steps / 4) - 1, params.n_steps + 1, int(params.n_steps / 4))]

    # Create a figure with four subplots for plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs array to iterate over subplots
    axs = axs.flatten()

    # Iterate over selected steps and create plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]
        axs[idx].hist(subset_data[measure], edgecolor='black', bins=bins, density = True)
        axs[idx].set_xlabel(measure)
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(f'Step {step}')


    # Adjust layout to prevent clipping of labels
    fig.tight_layout()

    path = utils.make_path("Figures", "Networks", f"Histogram_{measure}")
    plt.savefig(path)

    plt.close()
    

def viz_Degree_Distr(df, measure, bins=10):
    stepsPlot = [*range(int(params.n_steps / 4) - 1, params.n_steps + 1, int(params.n_steps / 4))]

    # Create a figure with four subplots for plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs array to iterate over subplots
    axs = axs.flatten()    

    # Iterate over selected steps and create plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]
        combined_degree_distribution = []
        data = utils.string_to_list(subset_data['Degree Distr'])
        combined_degree_distribution.extend(data)


        axs[idx].hist(combined_degree_distribution, edgecolor='black', bins=bins)
        axs[idx].set_xlabel(measure)
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(f'Step {step}')


    # Adjust layout to prevent clipping of labels
    fig.tight_layout()

    # Save Plots
    path = utils.make_path("Figures", "Networks", f"Distr_{measure}")
    plt.savefig(path)
    plt.close()


