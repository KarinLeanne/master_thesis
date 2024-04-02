

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp2d
import matplotlib.cm as cm
import seaborn as sns
import ast
import json
import seaborn as sns
import inequalipy   


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

def viz_time_series_agent_data_recent_wealth_single(df, NH = False):
    viz_time_series_agent_data_single(df, "Recent Wealth", "Recent Player Wealth", NH = False)


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


def viz_cml_wealth(df, NH=False):
    
    time_steps = [1, 10, 30, 60, 120, 240, 480]

    # Create a figure with two subplots for all CML plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Iterate over selected steps and create cumulative distribution plots for wealth and recent wealth
    for ax, var in zip(axs, ["Wealth", 'Recent Wealth']):
        for step in time_steps:
            subset_data = df[df['Step'] == step-1]
            wealth = subset_data[var]

            # Normalize wealth over total wealth
            total_wealth = wealth.sum()
            normalized_wealth = wealth / total_wealth

            # Plot cumulative distribution
            ax.plot(np.sort(normalized_wealth), np.linspace(0, 1, len(normalized_wealth), endpoint=False), 
                        label=f'Step {step}', linewidth=2)  # Adjust linewidth

            ax.set_xlabel(f'Normalized {var}', fontsize=20)
            ax.set_ylabel('Density', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=18)  # Increase tick size for all axes

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [f'{int(label.split()[1])}' for label in labels], fontsize='large')

    plt.suptitle(f"Cumulative Distribution of {var}", fontsize=22)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save Cumulative Distribution Plot
    if NH:
        path = utils.make_path("Figures", "GameChoice", f"Wealth_CML_NH")
        plt.savefig(path)
    else:
        path = utils.make_path("Figures", "GameChoice", f"Wealth_CML")
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


def viz_uv(data, variable, bar_name, visualize_wealth = False):
    '''
    Input: 
        data: agent level data
    '''
    # Define the number of steps to visualize
    # Get evenly spaced time steps
    last_timestep =  data['Step'].max()
    time_steps = np.linspace(0, last_timestep, 8)
    time_steps = time_steps.astype(int)

    # Create a figure with 8 subplots
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 0.05])

    # Loop through each step and create a subplot
    for idx, step in enumerate(time_steps):
        # Filter data for the current step
        step_data = data[data['Step'] == step]
        
        if visualize_wealth:
            # Group by UV coordinates and calculate median wealth
            grouped_data = step_data.groupby(['UV'])[variable].median().reset_index(name=variable)
            # Extract wealth values
            values = grouped_data[variable]
            # Normalize median wealth
            values = (values - values.min()) / (values.max() - values.min())
            counts = step_data.groupby('UV').size().reset_index(name='Counts')['Counts']
        else:
            # Group by UV coordinates and risk aversion values and count the number of agents
            grouped_data = step_data.groupby(['UV', variable]).size().reset_index(name='Counts')
            # Extract risk aversion values
            values = grouped_data[variable]
            # Determine bubble sizes based on the number of agents
            counts = grouped_data['Counts']

        bubble_sizes = counts * 10  # Adjust scaling factor as needed
        # Extract UV coordinates
        U = grouped_data['UV'].apply(lambda x: x[0]).tolist()
        V = grouped_data['UV'].apply(lambda x: x[1]).tolist()
        
        # Set colors 
        colors = values
        
        # Determine subplot index
        row = idx // 4  # Determine row index (0 or 1)
        col = idx % 4   # Determine column index (0 to 3)
        ax = fig.add_subplot(gs[row, col])

        ax.scatter(U, V, s=bubble_sizes, c=colors, cmap='viridis', alpha=0.5)  # Scatter plot with bubble sizes and risk aversion or median wealth color mapping
        ax.set_title(f'Step {step+1}', fontsize = 18)  # Set subplot title
        ax.set_xlabel('U', fontsize = 16)  # Set x-axis label
        ax.set_ylabel('V', fontsize = 16)  # Set y-axis label
        ax.grid(True)  # Add grid
        
 
        for u, v in zip(U, V):
            ax.scatter(u, v, color='grey', s=5)  # Little dot

    # Add colorbar
    cax = fig.add_subplot(gs[2, :])
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation='horizontal')
    if visualize_wealth:
        cbar.set_label(f"Normalized {bar_name}", fontsize = 16)

    cbar.set_label(bar_name, fontsize = 16)

    plt.suptitle(f"UV-Space and {bar_name} at Different Time-Steps", fontsize = params.titlesize)
    path = utils.make_path("Figures", "GameChoice", f"Coevolution_UV_{bar_name}")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def viz_coevolution_UV_risk(data):
    viz_uv(data, "Player Risk Aversion", "Risk Aversion", visualize_wealth = False)
    
def viz_UV_Wealth(data):
    viz_uv(data, "Wealth", "Median Recent Wealth", visualize_wealth = True)

def viz_UV_Recent_Wealth(data):
    viz_uv(data, "Recent Wealth", "Recent Wealth", visualize_wealth = True)

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
    
    df_grouped = df.groupby(['Step', 'Round'])[measure].mean().reset_index()

    # Calculate mean and std over simulations
    mean_df = df_grouped.groupby('Step')[measure].mean().reset_index()
    std_df = df_grouped.groupby('Step')[measure].std().reset_index()

    # Plot mean and std
    # Plot mean and shaded area for std
    plt.plot(mean_df['Step'], mean_df[measure], color='m')
    plt.fill_between(mean_df['Step'], mean_df[measure] - std_df[measure], mean_df[measure] + std_df[measure], color='m', alpha=0.3, )
    plt.xlabel('Timestep', fontsize = 16)
    plt.ylabel(measure, fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.title(f"{measure} over Time")
    path = utils.make_path("Figures", "GameChoice", f"{measure}_over_time")
    plt.savefig(path)
    plt.close()


def viz_gini_over_time(data):
    """
    Plot Gini coefficient of wealth and recent wealth over time with 95% confidence interval.

    Parameters:
        data (DataFrame): DataFrame containing the data with columns Step, Round, Wealth, and Recent Wealth.
    """
    # Calculate Gini coefficient for wealth
    gini_wealth = data.groupby(['Step', 'Round'])['Wealth'].apply(lambda x: inequalipy.gini(x.values)).reset_index()
    gini_wealth.columns = ['Step', 'Round', 'Gini_Wealth']

    # Calculate Gini coefficient for recent wealth
    gini_recent_wealth = data.groupby(['Step', 'Round'])['Recent Wealth'].apply(lambda x: inequalipy.gini(x.values)).reset_index()
    gini_recent_wealth.columns = ['Step', 'Round', 'Gini_Recent_Wealth']

    # Merge Gini coefficients
    merged_data = pd.merge(gini_wealth, gini_recent_wealth, on=['Step', 'Round'])

    # Calculate mean and 95% confidence interval for each timestep
    mean_gini_wealth = merged_data.groupby('Step')['Gini_Wealth'].mean()
    sem_gini_wealth = merged_data.groupby('Step')['Gini_Wealth'].sem()
    ci_gini_wealth = 1.96 * sem_gini_wealth  # 1.96 is the z-value for 95% confidence interval

    mean_gini_recent_wealth = merged_data.groupby('Step')['Gini_Recent_Wealth'].mean()
    sem_gini_recent_wealth = merged_data.groupby('Step')['Gini_Recent_Wealth'].sem()
    ci_gini_recent_wealth = 1.96 * sem_gini_recent_wealth  # 1.96 is the z-value for 95% confidence interval

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate y-axis limits
    y_min = min(mean_gini_wealth.min(), mean_gini_recent_wealth.min())
    y_max = max(mean_gini_wealth.max(), mean_gini_recent_wealth.max())

    # Plot Gini coefficient of wealth with 95% confidence interval
    axs[0].plot(mean_gini_wealth.index, mean_gini_wealth, color='blue', label='Mean')
    axs[0].fill_between(mean_gini_wealth.index, mean_gini_wealth - ci_gini_wealth, mean_gini_wealth + ci_gini_wealth, color='blue', alpha=0.3, label='95% CI')
    axs[0].set_xlabel('Timestep', fontsize=20)
    axs[0].set_ylabel('Gini of Wealth', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[0].legend()
    axs[0].set_ylim(y_min, y_max)

    # Plot Gini coefficient of recent wealth with 95% confidence interval
    axs[1].plot(mean_gini_recent_wealth.index, mean_gini_recent_wealth, color='purple', label='Mean')
    axs[1].fill_between(mean_gini_recent_wealth.index, mean_gini_recent_wealth - ci_gini_recent_wealth, mean_gini_recent_wealth + ci_gini_recent_wealth, color='purple', alpha=0.3, label='95% CI')
    axs[1].set_xlabel('Timestep', fontsize=20)
    axs[1].set_ylabel('Gini of Recent Wealth', fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[1].legend()
    axs[1].set_ylim(y_min, y_max)

    

    # Adjust layout and save plot
    plt.suptitle("Gini Coëfficient over Time", fontsize = 22)
    plt.tight_layout()
    path = utils.make_path("Figures", "GameChoice", "Gini_Coefficients_over_Time")
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


def viz_effect_of_risk_and_rationality_on_QRE(df):
    # Define your x and y variables
    x_vars = ['lambda1', 'lambda2', 'etaA', 'etaB']
    y_var = 'avg_qre_result'

    x_names = ["Lambda A", "Lambda B", "Eta A", "Eta B"]
    y_name = "Avg QRE Results"

    # Define colors for each subsequent plot
    colors = ['blue', 'purple', 'red', 'orange']  

    # Create a figure and axis objects
    fig, axes = plt.subplots(1, len(x_vars), figsize=(15, 5))

    # Loop through each pair of x and y variables
    for i, x_var in enumerate(x_vars):
        # Plot scatter plot with specific x and y variables, color, and axes
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=axes[i], color=colors[i])
        # Plot regression line on the same plot
        sns.regplot(x=x_var, y=y_var, data=df, ax=axes[i], scatter=False, color=colors[i])
        # Set x and y labels
        axes[i].set_xlabel(x_names[i], fontsize=params.labelsize)
        axes[i].set_ylabel(y_name, fontsize=params.labelsize)


    # Set title for the entire figure
    fig.suptitle(f'Scatter & Regression Plot of {y_name} vs Risk and Rationality Parameters', fontsize=params.titlesize)

    # Adjust tick font size
    plt.tick_params(axis='both', which='major', labelsize=params.ticksize)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    path = utils.make_path("Figures", "GameChoice", "effect_of_risk_rationality_on_QRE_regression_scatter")
    plt.savefig(path)
    plt.close()



def analyze_wealth_distribution(data):
    # Filter data for the last timestep
    data = data[data['Step'] == data['Step'].max()]
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot wealth distributions
    sns.kdeplot(data=data['Wealth'], color='blue', fill=True, alpha=0.1, linewidth=2, bw_adjust=2, ax=axs[0])
    axs[0].set_xlabel('Wealth', fontsize=params.labelsize)
    axs[0].set_ylabel('Density', fontsize=params.labelsize)
    axs[0].set_title('Wealth Distribution', fontsize=params.titlesize)
    axs[0].tick_params(axis='both', which='major', labelsize=params.ticksize)  

    # Plot recent wealth distributions
    sns.kdeplot(data=data['Recent Wealth'], color='green', fill=True, alpha=0.1, linewidth=2, bw_adjust=2, ax=axs[1])
    axs[1].set_xlabel('Recent Wealth', fontsize=params.labelsize)
    axs[1].set_ylabel('Density', fontsize=params.labelsize)
    axs[1].set_title('Recent Wealth Distribution', fontsize=params.titlesize)
    axs[1].tick_params(axis='both', which='major', labelsize=params.ticksize)  

    plt.tight_layout()

    # Save the figure
    path = utils.make_path("Figures", "GameChoice", "Wealth_Distribution")
    plt.savefig(path)
    plt.close()