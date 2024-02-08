### Vizualize.py
# Contains all functions making vizualizations using simulation data
###


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
    viz_time_series_agent_data_single(df, "Player risk aversion", "Player Risk Aversion", NH = False)


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
    viz_time_series_agent_data_multiple(df, "Player risk aversion", "Player Risk Aversion", "Rationality Distribution")

def viz_time_series_agent_data_pay_off_for_rat_dist(df):
    viz_time_series_agent_data_multiple(df, "Wealth", "Player Wealth", "Rationality Distribution")

def viz_time_series_agent_data_rationality_for_util(df):
    viz_time_series_agent_data_multiple(df, "Player risk aversion", "Player Risk Aversion", "Utility Function")

def viz_time_series_agent_data_pay_off_for_util(df):
    viz_time_series_agent_data_multiple(df, "Wealth", "Player Wealth", "Utility Function")




def plot_wealth_distribution(df, NH = False):

    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create figures with four subplots
    fig_violin, axs_violin = plt.subplots(2, 2, figsize=(12, 8))
    fig_kde, axs_kde = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs_violin array to iterate over subplots
    axs_violin = axs_violin.flatten()
    axs_kde = axs_kde.flatten()

    # Iterate over selected steps and create violin and KDE plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Violin plot
        sns.violinplot(x=subset_data['Step'], y=subset_data['Wealth'], ax=axs_violin[idx], color='m')
        axs_violin[idx].set_title(f'Step {step}')
        axs_violin[idx].set_xlabel('Step')
        axs_violin[idx].set_ylabel('Player Payoff')

        # KDE plot
        sns.kdeplot(subset_data['Wealth'], ax=axs_kde[idx], color='m', fill=True)
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


def plot_cml_wealth(df, NH=False):
    steps_plot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create a single figure for all CML plots
    fig_cml, ax_cml = plt.subplots(figsize=(12, 8))

    # Iterate over selected steps and create cumulative distribution plots
    for step in steps_plot:
        subset_data = df[df['Step'] == step]
        cumulative_distribution_plot(ax_cml, subset_data['Wealth'], step)

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

    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # Iterate over selected steps and create scatter plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Scatter plot with regression line
        sns.regplot(x='Player risk aversion', y='Wealth', data=subset_data, scatter_kws={'alpha': 0.5}, ax=axs[idx])
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


def viz_UV_heatmap(df, NH=False):
    sigma = 16

    stepsPlot = [int(params.n_steps * i / 8) for i in range(1, 9)]

    # Create a 2x4 grid of subplots with equal column widths
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # Initialize empty lists to store U and V values for all steps
    all_U_values = []
    all_V_values = []

    for idx, step in enumerate(stepsPlot):
        dfPlot = df.loc[df['Step'] == step]
        U_step = dfPlot['UV'].apply(lambda x: x[0]).tolist()
        V_step = dfPlot['UV'].apply(lambda x: x[1]).tolist()
        all_U_values.append(U_step)
        all_V_values.append(V_step)

    # Flatten the lists to get values for all steps
    U = [item for sublist in all_U_values for item in sublist]
    V = [item for sublist in all_V_values for item in sublist]

    # Calculate common extent based on the minimum and maximum values of U and V
    common_extent = [np.min(U), np.max(U), np.min(V), np.max(V)]

    for idx, step in enumerate(stepsPlot):
        dfPlot = df.loc[df['Step'] == step]

        U = dfPlot['UV'].apply(lambda x: x[0])
        V = dfPlot['UV'].apply(lambda x: x[1])

        # Ensure that the number of steps is compatible with a 2x4 grid
        ax = axes[idx // 4, idx % 4]  # Get the current subplot

        heatmap, xedges, yedges = np.histogram2d(U, V, bins=50)
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Use common extent for all subplots
        im = ax.imshow(heatmap.T, extent=common_extent, origin='lower', cmap=cm.jet, interpolation='none', aspect='auto') 
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.set_title("Step: {}".format(step))

    # Add a colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)

    # Adjust layout to prevent clipping of titles and colorbar
    plt.tight_layout(rect=[0, 0, 0.91, 1])

    if NH:
        path = utils.make_path("Figures", "GameChoice", "UV_heatmap_NH")
        plt.savefig(path)
    else:
        path = utils.make_path("Figures", "GameChoice", "UV_heatmap")
        plt.savefig(path)
    plt.close()


"""
def viz_UV(UV_space):
    #\\FIXME Most likely some recording of the UV space is incorrect as there is no dynamic change in the UV space 
    
    t=0
    df = pd.DataFrame(UV_space[0][t], columns=['U'])
    df['V'] = UV_space[1][t]

    fig = px.scatter(df, x="U", y="V")
    fig = px.density_contour(df, x="U", y="V", marginal_x="histogram", marginal_y="histogram")
    #fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    fig.show()
"""

def viz_deg_distr(G):

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()



def viz_effect_clustering_coef(name, p, data):
    fig, ax = plt.subplots()
    ax.scatter(p, data, c="green", alpha=0.5, marker='x')
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel(name)
    ax.legend()
    plt.show()


def network_measures_over_timesteps(df):

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


def effect_variable_on_network_measures(df, variable):


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

                    
def vizualize_effect_of_rationality_on_QRE(df):

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Mean Lambda'], df['Std Dev Lambda'], df['QRE Result'], c='blue', marker='o')

    ax.set_xlabel('Mean Lambda')
    ax.set_ylabel('Std Dev Lambda')
    ax.set_zlabel('QRE Result')

    path = utils.make_path("Figures", "GameChoice", "Effect_of_Rationality_on_QRE")
    plt.savefig(path)
    plt.close()



def vizualize_measure_over_time(df, measure):

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

    path = utils.make_path("Figures", "Networks", f"{measure}_over_time")
    plt.savefig(path)
    plt.close()

def vizualize_histogram_over_time(df, measure, bins=10):
    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create a figure with four subplots for plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs array to iterate over subplots
    axs = axs.flatten()

    # Iterate over selected steps and create plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]
        axs[idx].hist(subset_data[measure], edgecolor='black', bins=bins)
        axs[idx].set_xlabel(measure)
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(f'Step {step}')


    # Adjust layout to prevent clipping of labels
    fig.tight_layout()

    path = utils.make_path("Figures", "Networks", f"Histogram_{measure}")
    plt.savefig(path)

    plt.close()
    

def vizualize_Degree_Distr(df, measure, bins=10):
    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create a figure with four subplots for plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs array to iterate over subplots
    axs = axs.flatten()    

    # Iterate over selected steps and create plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]
        
        combined_degree_distribution = []
        for degree_list in subset_data['Degree Distr']:
            combined_degree_distribution.extend(json.loads(degree_list))

        axs[idx].hist(combined_degree_distribution, edgecolor='black', bins=bins)
        axs[idx].set_xlabel(measure)
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(f'Step {step}')


    # Adjust layout to prevent clipping of labels
    fig.tight_layout()

    # Save Plots
    path = utils.make_path("Figures", "Networks", f"Degree_Distr_{measure}")
    plt.savefig(path)
    plt.close()


def plot_speed_network_vs_speed_games(df, measure):
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with colorbar
    scatter = plt.scatter(x=df['Speed_Ratio'], y=df[measure], c=df['rewiring_p'], cmap='viridis', s=100)
    
    plt.title(f'Scatter Plot of {measure}'.replace("_", " ") + " vs Speed Ratio with Colorbar'")
    plt.xlabel('Speed Ratio (e_g / e_n)')
    plt.ylabel(f"{measure}".replace("_", " "))
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Rewiring Probability')

    # Save Plots
    path = utils.make_path("Figures", "UpdatingMechanism", f"Plot of {measure} vs Speed Ratio")
    plt.savefig(path)
    plt.close()

def plot_speed_ratio_time_series(df, values):

    rewiring_probabilities = values[::2]

    for rewiring_p in rewiring_probabilities:

        subset = df[df['rewiring_p'] == rewiring_p]
        df_grouped = subset.groupby(['Step', 'Round'])["Speed_Ratio"].mean().reset_index()

        # Calculate mean and std over simulations
        mean_df = df_grouped.groupby('Step')["Speed_Ratio"].mean().reset_index()
        std_df = df_grouped.groupby('Step')["Speed_Ratio"].std().reset_index()

        # Group the DataFrame by 'rewiring_p' and iterate over groups
        plt.plot(mean_df['Step'], mean_df['Speed_Ratio'], label=f'rewiring_p={rewiring_p}')
        plt.fill_between(mean_df['Step'], mean_df["Speed_Ratio"] - std_df["Speed_Ratio"], mean_df["Speed_Ratio"] + std_df["Speed_Ratio"], alpha=0.3)

    # Customize the plot
    plt.xlabel('Step')
    plt.ylabel('Speed Ratio')
    plt.title('Speed Ratio over Time for Different Rewiring Probabilities')
    plt.legend()
    plt.grid(True)

    # Save Plots
    path = utils.make_path("Figures", "UpdatingMechanism", f"Speed Ratio Time Series")
    plt.savefig(path)
    plt.close()

