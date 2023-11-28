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

import utils

params = utils.get_config()



def viz_time_series_agent_data(df, measure, name_measure):
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
    plt.savefig(f"plots/chap2/{name_measure}__{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")
    # Close the figure to prevent overlap
    plt.close() 


def viz_time_series_agent_data_rationality(df):
    viz_time_series_agent_data(df, "Player risk aversion", "Player Risk Aversion")


def viz_time_series_agent_data_pay_off(df):
    viz_time_series_agent_data(df, "playerPayoff", "Player Wealth")


def viz_wealth_distribution(df):

    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create a figure with four subplots for violin plots
    fig_violin, axs_violin = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs_violin array to iterate over subplots
    axs_violin = axs_violin.flatten()

    # Create a figure with four subplots for KDE plots
    fig_kde, axs_kde = plt.subplots(2, 2, figsize=(12, 8))

    # Flatten the axs_kde array to iterate over subplots
    axs_kde = axs_kde.flatten()

    # Iterate over selected steps and create violin and KDE plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Violin plot
        sns.violinplot(x=subset_data['Step'], y=subset_data['playerPayoff'], ax=axs_violin[idx], color='m')
        axs_violin[idx].set_title(f'Step {step}')
        axs_violin[idx].set_xlabel('Step')
        axs_violin[idx].set_ylabel('Player Payoff')

        # KDE plot
        sns.kdeplot(subset_data['playerPayoff'], ax=axs_kde[idx], color='m', fill=True)
        axs_kde[idx].set_title(f'Step {step}')
        axs_kde[idx].set_xlabel('Player Payoff')
        axs_kde[idx].set_ylabel('Density')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save Plots
    # Save violin plot
    fig_violin.savefig(f"plots/chap2/Wealth_violin_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")
    
    # Save KDE plot
    fig_kde.savefig(f"plots/chap2/Wealth_KDE_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")


def viz_corrrelation(df):

    stepsPlot = [*range(int(params.n_steps / 4), params.n_steps + 1, int(params.n_steps / 4))]

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # Iterate over selected steps and create scatter plots
    for idx, step in enumerate(stepsPlot):
        subset_data = df[df['Step'] == step]

        # Scatter plot with regression line
        sns.regplot(x='Player risk aversion', y='playerPayoff', data=subset_data, scatter_kws={'alpha': 0.5}, ax=axs[idx])
        axs[idx].set_title(f'Scatter Plot at Step {step}')
        axs[idx].set_xlabel('Risk Averseness')
        axs[idx].set_ylabel('Pay-off')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig(f"plots/chap2/Correlation_Wealth_Risk_aversion__{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")
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
    
    plt.savefig(f"plots/chap2/UV_scatter__{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")
    # Close the figure to prevent overlap
    plt.close()  


def viz_UV_heatmap(df):
    sigma = 16
    stepsPlot = [*range(int(params.n_steps / 8), params.n_steps + 1, int(params.n_steps / 8))]  # Adjusted for 2 rows and 4 columns

    # Create a 2x4 grid of subplots with equal column widths
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # Initialize empty lists to store U and V values for all steps
    all_U_values = []
    all_V_values = []

    for idx, step in enumerate(stepsPlot):
        dfPlot = df.loc[df['Step'] == step]

        U = dfPlot['UV'].apply(lambda x: x[0])
        V = dfPlot['UV'].apply(lambda x: x[1])

        # Append U and V values to the lists
        all_U_values.extend(U)
        all_V_values.extend(V)

    # Calculate common extent based on the minimum and maximum values of U and V
    common_extent = [np.min(all_U_values), np.max(all_U_values), np.min(all_V_values), np.max(all_V_values)]

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

    plt.savefig(f"plots/chap2/UV_heatmap_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")
    plt.show()


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



def heatmap(N, rounds, steps):
    '''Compute heatmap for various U and V values for payoffs'''

    U = np.linspace(0,1,10)
    V = np.linspace(0,1,10)
    heat_map = []
    data = []

    for u in U:
        temp = []
        temp2 = []
        for v in V:
            Payoffs = sim.simulateANDWealth(N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,u,v))
            #heatmap[u][v] = gini_new(np.array(Payoffs[-1]))
            temp.append(sim.gini_new(np.array(Payoffs[-1])))
            temp2.append(Payoffs[-1])
        heat_map.append(temp)
        data.append(temp2)

def viz_effect_clustering_coef(name, p, data):
    fig, ax = plt.subplots()
    ax.scatter(p, data, c="green", alpha=0.5, marker='x')
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel(name)
    ax.legend()
    plt.show()


def network_measures_over_timesteps(df):

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
    
    plt.savefig(f"plots/chap1/network_measures_over_time__{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")


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
        plt.savefig(f"plots/chap1/Effect_of_{variable}_on_{measure.replace('M:', '')}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png")

                    


        










