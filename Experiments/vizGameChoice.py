

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


def viz_time_series_agent_data_single(df, measure, name_measure, NH = False):
    # Calculate mean and standard deviation of risk aversion  per step
    grouped_data = df.groupby('Step')[measure].agg(['mean', 'std'])

    # Create a new subplot for the variance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the time series with mean and standard deviation
    ax1.plot(grouped_data.index, grouped_data['mean'], label='Mean', color='m')
    ax1.fill_between(grouped_data.index,
                    grouped_data['mean'] - grouped_data['std'],
                    grouped_data['mean'] + grouped_data['std'],
                    color='m', alpha=0.1, label='Standard Deviation')

    # Add labels and title to the first subplot
    ax1.set_ylabel(name_measure)
    ax1.set_title(f"Time Series of {name_measure} Mean")
    ax1.legend()

    # Calculate and plot the time series of the variance
    variance = df.groupby('Step')[measure].var()
    ax2.plot(variance.index, variance, label='Variance', color='purple')

    # Add labels and title to the second subplot
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Variance')
    ax2.set_title(f"Time Series of {name_measure} Variance")
    ax2.legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save plot
    path = utils.make_path("Figures", "GameChoice", "Influence_Rewiring_p_On_Network")
    if NH:
        path = utils.make_path("Figures", "GameChoice", f"{name_measure}_NH")
        plt.savefig(path)
    else:
        path = utils.make_path("Figures", "GameChoice", f"{name_measure}")
        plt.savefig(path)

    # Close the figure to prevent overlap
    plt.close() 


def viz_time_series_agent_data_rationality_single(df, NH = False):
    viz_time_series_agent_data_single(df, "Player Risk Aversion", "Player Risk Aversion", NH = False)


def viz_time_series_agent_data_pay_off_single(df, NH = False):
    viz_time_series_agent_data_single(df, "Wealth", "Player Wealth", NH = False)


def viz_time_series_agent_data_multiple(df, measure, name_measure, indep_var):
    # Create a new subplot for the time series
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use sautomatic color cycling
    sns.set_palette("husl")

    # Iterate over unique values of indep_var
    for value in df[indep_var].unique():
        # Filter the DataFrame for each value of indep_var
        subset_data = df[df[indep_var] == value]
        
        # Calculate mean and standard deviation of the measure per step
        grouped_data = subset_data.groupby('Step')[measure].agg(['mean', 'std'])
        
        # Plot the time series with mean and standard deviation
        ax.plot(grouped_data.index, grouped_data['mean'], label=f'{indep_var}={value}')
        ax.fill_between(grouped_data.index,
                        grouped_data['mean'] - grouped_data['std'],
                        grouped_data['mean'] + grouped_data['std'],
                        alpha=0.1)

    # Add labels and title to the plot
    ax.set_xlabel('Step')
    ax.set_ylabel(name_measure)
    ax.set_title(f"Time Series of {name_measure} for Different {indep_var} Values")
    ax.legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save plot
    path = utils.make_path("Figures", "GameChoice", f"{name_measure}_{indep_var}_time_series")
    plt.savefig(path)
    


    # Close the figure to prevent overlap
    plt.close()


def viz_time_series_agent_data_rationality_for_rat_dist(df):
    viz_time_series_agent_data_multiple(df, "Player Risk Aversion", "Player Risk Aversion", "Rationality Distribution")

def viz_time_series_agent_data_pay_off_for_rat_dist(df):
    viz_time_series_agent_data_multiple(df, "Wealth", "Player Wealth", "Rationality Distribution")

def viz_time_series_agent_data_rationality_for_util(df):
    viz_time_series_agent_data_multiple(df, "Player Risk Aversion", "Player Risk Aversion", "Utility Function")

def viz_time_series_agent_data_pay_off_for_util(df):
    viz_time_series_agent_data_multiple(df, "Wealth", "Player Wealth", "Utility Function")




def viz_wealth_distribution(df, NH = False):

    stepsPlot = [*range(int(params.n_steps / 4) - 1, params.n_steps + 1, int(params.n_steps / 4))]

    # Create figures with four subplots
    fig_violin, axs_violin = plt.subplots(2, 2, figsize=(12, 8))
    fig_kde, axs_kde = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs_violin array to iterate over subplots
    axs_violin = axs_violin.flatten()
    axs_kde = axs_kde.flatten()

    # Iterate over selected steps and create violin and KDE plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Normalize wealth over total wealth
        wealth = subset_data['Wealth']
        total_wealth = wealth.sum()
        normalized_wealth = wealth / total_wealth

        # Violin plot
        sns.violinplot(x=subset_data['Step'], y=normalized_wealth, ax=axs_violin[idx], color='m')
        axs_violin[idx].set_title(f'Step {step}')
        axs_violin[idx].set_xlabel('Step')
        axs_violin[idx].set_ylabel('Player Payoff')

        # KDE plot
        sns.kdeplot(normalized_wealth, ax=axs_kde[idx], color='m', fill=True)
        axs_kde[idx].set_title(f'Step {step}')
        axs_kde[idx].set_xlabel('Player Payoff')
        axs_kde[idx].set_ylabel('Density')


    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    if NH:
        # Save Plots
        path = utils.make_path("Figures", "GameChoice", "Wealth_Violin_NH")
        plt.savefig(path)
        plt.close()

        path = utils.make_path("Figures", "GameChoice", "Wealth_KDE_NH")
        plt.savefig(path)
        plt.close()
    else:
        # Save Plots
        path = utils.make_path("Figures", "GameChoice", "Wealth_Violin")
        plt.savefig(path)
        plt.close()

        path = utils.make_path("Figures", "GameChoice", "Wealth_KDE")
        plt.savefig(path)
        plt.close()


def cumulative_distribution_plot(ax, data, step):
    ax.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False), label=f'Step {step}')
    ax.set_title(f'Step {step}')
    ax.set_xlabel('Player Payoff')
    ax.set_ylabel('Density')


def viz_cml_wealth(df, NH=False):
    steps_plot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create a single figure for all CML plots
    fig_cml, ax_cml = plt.subplots(figsize=(12, 8))

    # Iterate over selected steps and create cumulative distribution plots
    for step in steps_plot:
        subset_data = df[df['Step'] == step]
        wealth = subset_data['Wealth']
        
        # Normalize wealth over total wealth
        total_wealth = wealth.sum()
        normalized_wealth = wealth / total_wealth
        
        cumulative_distribution_plot(ax_cml, normalized_wealth, step)

    # Add legend
    ax_cml.legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save Cumulative Distribution Plot
    if NH:
        path = utils.make_path("Figures", "GameChoice", "Wealth_CML_NH")
        plt.savefig(path)
    else:
        path = utils.make_path("Figures", "GameChoice", "Wealth_CML")
        plt.savefig(path)
    plt.close()



def viz_corrrelation(df, NH = False):

    stepsPlot = [*range(int(params.n_steps / 4) - 1, params.n_steps + 1, int(params.n_steps / 4))]

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # Iterate over selected steps and create scatter plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Scatter plot with regression line
        sns.regplot(x='Player Risk Aversion', y='Wealth', data=subset_data, scatter_kws={'alpha': 0.5}, ax=axs[idx])
        axs[idx].set_title(f'Scatter Plot at Step {step}')
        axs[idx].set_xlabel('Risk Averseness')
        axs[idx].set_ylabel('Pay-off')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    if NH:
        path = utils.make_path("Figures", "GameChoice", "Correlation_Wealth_Risk_aversion_NH")
        plt.savefig(path)
    else:
        path = utils.make_path("Figures", "GameChoice", "Correlation_Wealth_Risk_aversion")
        plt.savefig(path)
    # Close the figure to prevent overlap
    plt.close()  


def viz_UV_scatter(df):
    '''
    Description: Vizualize the UV spave 
    Input: 
        - df: Agent data collected during simulations 
    '''
    
    df = df.loc[df['Step'] == df['Step'].max()]


    #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    U = df['UV'].apply(lambda x: x[0])
    V = df['UV'].apply(lambda x: x[1])

    ax = plt.subplot()
    ax.scatter(U, V)
    ax.set_xlabel("U")
    ax.set_ylabel("V")

    path = utils.make_path("Figures", "GameChoice", "UV_scatter")
    plt.savefig(path)
    # Close the figure to prevent overlap
    plt.close()  


def viz_UV_heatmap(df_agent, df_model, NH=False, only_start = False):

    df_agent_r0 = df_agent[df_agent['Round'] == 0]
    df_game_info_r0 = df_model[df_model['Round'] == 0][['Step', 'Game data']]

    if only_start == True:
        stepsPlot = [int(8 * i / 8) - 1 for i in range(1, 9)]
    else:
        stepsPlot = [int(params.n_steps * i / 8) - 1 for i in range(1, 9)]


    # Create a 2x4 grid of subplots with equal column widths
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # Initialize empty lists to store U and V values for all steps
    all_U_values = []
    all_V_values = []

    for idx, step in enumerate(stepsPlot):
        df_agentPlot = df_agent_r0.loc[df_agent_r0['Step'] == step]
        U_step = df_agentPlot['UV'].apply(lambda x: x[0]).tolist()
        V_step = df_agentPlot['UV'].apply(lambda x: x[1]).tolist()
        all_U_values.append(U_step)
        all_V_values.append(V_step)

        #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))


    # Flatten the lists to get values for all steps
    U_all = [item for sublist in all_U_values for item in sublist]
    V_all = [item for sublist in all_V_values for item in sublist]


    # Calculate common extent based on the minimum and maximum values of U and V
    common_extent = [min(U_all), max(U_all), min(V_all), max(V_all)]

    for idx, step in enumerate(stepsPlot):
        
        dfPlot = df_agent_r0.loc[df_agent_r0['Step'] == step]

        U_step = dfPlot['UV'].apply(lambda x: x[0]).tolist()
        V_step = dfPlot['UV'].apply(lambda x: x[1]).tolist()

        # Ensure that the number of steps is compatible with a 2x4 grid
        ax = axes[idx // 4, idx % 4]  # Get the current subplot

        heatmap, xedges, yedges = np.histogram2d(U_step, V_step, bins=30)
        sigma = 30
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Normalize heatmap by total count
        total_count = np.sum(heatmap)
        heatmap = heatmap / total_count

        # Plot heatmap
        im = ax.imshow(heatmap.T, extent=common_extent, origin='lower', cmap='viridis', interpolation='none', aspect='auto')
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.set_title("Step: {}".format(step+1))

        # Overlay scatter plot of individual data points
        ax.scatter(U_step, V_step, color='black', s=10, alpha=0.5)

        # Add UV bubbles
        df_game_info_step = df_game_info_r0[df_game_info_r0['Step'] == step]

        # Calculate the total play count for all games
        if isinstance(df_game_info_step['Game data'].values[0], str):
            game_data_list = df_game_info_step['Game data'].apply(ast.literal_eval).tolist()[0]
        else:
            game_data_list =  df_game_info_step['Game data'].values[0]


        total_play_count = sum(game[1] for game in game_data_list)

        # Find the three most played games
        most_played_games = sorted(game_data_list, key=lambda x: x[1], reverse=True)[:3]

        # Get UV data for the three most played games
        most_played_UV = [game[3] for game in most_played_games]

        # Calculate the size of each bubble based on the percentage of play count
        bubble_sizes = [game[1] / total_play_count * 100 for game in most_played_games]

        # Plot bubbles for the three most played games with larger size
        for UV, size in zip(most_played_UV, bubble_sizes):
            ax.scatter(UV[0], UV[1], s=size*50, alpha=0.7, label="Most Played Game", edgecolor='black', linewidth=1.5)

    # Adjust layout to prevent clipping of titles and colorbar
    plt.tight_layout(rect=[0, 0, 0.91, 1])


    if NH:
        path = utils.make_path("Figures", "GameChoice", f"UV_heatmap_NH_only_start_{only_start}")
    else:
        path = utils.make_path("Figures", "GameChoice", f"UV_heatmap_only_start_{only_start}")
    plt.savefig(path)
    plt.close()

def viz_measure_over_time(df, measure):

    #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    # Count the unique games at each timestep
    
    df_grouped = df.groupby(['Step', 'Round'])[measure].mean().reset_index()

    # Calculate mean and std over simulations
    mean_df = df_grouped.groupby('Step')[measure].mean().reset_index()
    std_df = df_grouped.groupby('Step')[measure].std().reset_index()

    # Plot mean and std
    # Plot mean and shaded area for std
    plt.plot(mean_df['Step'], mean_df[measure], label='Mean Games', color='b')
    plt.fill_between(mean_df['Step'], mean_df[measure] - std_df[measure], mean_df[measure] + std_df[measure], color='b', alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel(measure)
    plt.legend()

    path = utils.make_path("Figures", "GameChoice", f"{measure}_over_time")
    plt.savefig(path)
    plt.close()

def viz_effect_of_rationality_on_QRE(df):
    # Extract data from DataFrame
    lambda1_values = df['Lambda1']
    lambda2_values = df['Lambda2']
    qre_results = df['QRE Result']

    # Define the range of Lambda1 and Lambda2 values
    lambda1_min, lambda1_max = lambda1_values.min(), lambda1_values.max()
    lambda2_min, lambda2_max = lambda2_values.min(), lambda2_values.max()

    # Create a 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(lambda1_values, lambda2_values, bins=50)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, extent=[lambda1_min, lambda1_max, lambda2_min, lambda2_max], origin='lower', cmap='viridis')
    plt.colorbar(label='QRE Result')
    plt.xlabel('Lambda1')
    plt.ylabel('Lambda2')
    plt.title('Effect of Rationality on QRE')
    
    path = utils.make_path("Figures", "GameChoice", "effect_of_rationality_on_QRE")
    plt.savefig(path)
    plt.close()



    