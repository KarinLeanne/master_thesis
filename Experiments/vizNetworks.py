from chart_studio import plotly as py 
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import networkx as nx
from IPython.display import display
from tabulate import tabulate
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d
import matplotlib.cm as cm
import seaborn as sns
import textwrap
import ast
import json


import utils

params = utils.get_config()


def viz_network_measures_over_timesteps(df):
    # Get steps, used networks, and used measures from dataframe
    steps = np.arange(df['Step'].nunique())
    networks = df['Network'].unique()

    # Only first two columns of dataframe do not contain measures
    mean_measures = [col.split(':')[1].strip().replace('M:', 'Average') for col in df.columns if 'M:' in col]
    std_measures = [col.split(':')[1].strip().replace('STD:', 'STD') for col in df.columns if 'STD:' in col]

    fig, axs = plt.subplots(len(mean_measures), len(networks), sharex='col', sharey='row', figsize=(14, 10))

    # Make plot per measure and network type
    for mean_measure, std_measure, row in zip(mean_measures, std_measures, range(len(mean_measures))):
        for network, col in zip(networks, range(len(networks))):
            mean = df.loc[df['Network'] == network][f'M: {mean_measure}']
            std = df.loc[df['Network'] == network][f'STD: {std_measure}']
            axs[row, col].plot(steps, mean, color='purple', linewidth=2)
            axs[row, col].fill_between(steps, mean + 1.96 * std, mean - 1.96 * std, alpha=0.2, color='purple')
            axs[row, col].tick_params(axis='both', which='major', labelsize=16)  # Increase tick size for all axes

    # Set common x label
    fig.text(0.5, 0.04, 'Step', ha='center', fontsize=20)

    # Set y labels with line breaks and rotation
    for ax, row in zip(axs[:, 0], mean_measures):
        wrapped_labels = textwrap.fill(row, width=10)  # Adjust the width as per your preference
        ax.set_ylabel(wrapped_labels, rotation=90, fontsize=16, labelpad=10, ha='center', rotation_mode='anchor')
        ax.yaxis.set_label_coords(-0.25, 0.5)  # Adjust label position to center after rotation

    # Make clear which network belongs to which graph
    for ax, col in zip(axs[0, :], networks):
        ax.set_title(col, fontsize=20)

    fig.suptitle("Network Measures Over Time", fontsize=22)

    # Adjust layout to prevent overlapping y-axis labels
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Save and close the figure
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

        # Adjust layout to prevent overlapping y-axis labels
        plt.subplots_adjust(wspace=0.3, hspace=0.8)

        # displaying the title
        fig.suptitle(f"Effect of {variable} on {measure.replace('M:', '')}")
        path = utils.make_path("Figures", "Networks", f"Effect_of_{variable}_on_{measure.replace('M:', '')}")
        fig.tight_layout()
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



def analyze_wealth_distribution_by_game_group(data):
    # Normalize wealth and recent wealth
    data = data[data['Step'] == data['Step'].max()]
    data['Normalized Wealth'] = data['Wealth'] / data['Wealth'].sum()
    data['Normalized Recent Wealth'] = data['Recent Wealth'] / data['Recent Wealth'].sum()

    low_game_players = data[data['Games played'] <= data['Games played'].quantile(0.33)]
    medium_game_players = data[(data['Games played'] > data['Games played'].quantile(0.33)) & 
                            (data['Games played'] <= data['Games played'].quantile(0.66))]
    high_game_players = data[data['Games played'] > data['Games played'].quantile(0.66)]


    # Create a figure with two subplots for plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the normalized wealth distributions for different game groups
    sns.kdeplot(data=low_game_players['Normalized Wealth'], color='blue', fill=True, alpha=0.1, label='Low Part Players', linewidth=2,  ax=axs[0])
    sns.kdeplot(data=medium_game_players['Normalized Wealth'], color='purple', fill=True, alpha=0.1, label='Medium Part Players', linewidth=2, ax=axs[0])
    sns.kdeplot(data=high_game_players['Normalized Wealth'], color='pink', fill=True, alpha=0.1, label='High Part Players', linewidth=2, ax=axs[0])
    axs[0].set_xlabel('Normalized Wealth', fontsize=params.labelsize)
    axs[0].set_ylabel('Density', fontsize=params.labelsize)
    axs[0].set_title('Normalized Wealth Distribution Across Game Groups', fontsize=params.titlesize)
    axs[0].tick_params(axis='both', which='major', labelsize=params.ticksize)  
    axs[0].legend(fontsize = params.labelsize)

    # Plot the normalized recent wealth distributions for the same game groups
    sns.kdeplot(data=low_game_players['Normalized Recent Wealth'], color='blue', fill=True, alpha=0.1, label='Low Part Players', linewidth=2, ax=axs[1])
    sns.kdeplot(data=medium_game_players['Normalized Recent Wealth'], color='purple', fill=True, alpha=0.1, label='Medium Part Players', linewidth=2, ax=axs[1])
    sns.kdeplot(data=high_game_players['Normalized Recent Wealth'], color='pink', fill=True, alpha=0.1, label='High Part Players', linewidth=2, ax=axs[1])
    axs[1].set_xlabel('Normalized Recent Wealth', fontsize=params.labelsize)
    axs[1].set_ylabel('Density', fontsize=params.labelsize)
    axs[1].set_title('Normalized Recent Wealth Distribution Across Game Groups', fontsize=params.titlesize)
    axs[1].tick_params(axis='both', which='major', labelsize=params.ticksize)  
    axs[1].legend(fontsize = params.labelsize)

    plt.tight_layout()

    path = utils.make_path("Figures", "Networks", "Wealth_Distribution_by_Game_Group")
    plt.savefig(path)
    plt.close()