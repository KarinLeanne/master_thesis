'''
OFAT.py
Contains code to perform and plot one factor at a time analysis on the data
'''

import os
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from GamesModel import GamesModel
import utils
from scipy.stats import sem
import scienceplots

plt.style.use(['science', 'ieee'])

def calculate_ci(data):
    '''
    Description: 
    Calculate the mean and confidence interval of the data.
    Inputs:
    - data (array-like): The data for which to calculate the mean and confidence interval.
    Outputs:
    - mean (float): The mean of the data.
    - ci (float): The confidence interval of the data.
    '''
    mean = data.mean()
    ci = 1.96 * sem(data)
    return mean, ci

def plot_ofat_wealth_measures(chapter, data, dependent_var, selected_independent_vars=None, x_labels=None):
    '''
    Description: 
    Plot the mean and confidence interval of wealth measures against independent variables using one-factor-at-a-time (OFAT) analysis.
    Inputs:
    - chapter (str): The name of the chapter or section to save the plot.
    - data (DataFrame): The DataFrame containing the data.
    - dependent_var (str): The dependent variable to plot against.
    - selected_independent_vars (list, optional): List of selected independent variables. Defaults to None.
    - x_labels (list, optional): List of x-axis labels. Defaults to None.
    Outputs:
    - None (plots are saved as image files).
    '''

    if selected_independent_vars is None:
        selected_independent_vars = data['IndependentVariable'].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(selected_independent_vars), figsize=(15, 5), sharey=True)

    if isinstance(axes, np.ndarray):  # Check if axes is a numpy array (i.e., multiple subplots)
        for i, ax in enumerate(axes):
            # Filter data for the current independent variable
            independent_var = selected_independent_vars[i]
            subset = data[data['IndependentVariable'] == independent_var]

            # Group data by the independent variable and calculate mean and CI
            grouped_data = subset.groupby('IndependentValue')[dependent_var].agg(calculate_ci).reset_index()

            # Unpack the calculated values
            mean, ci = zip(*grouped_data[dependent_var])

            # Plot the mean line
            sns.lineplot(x=grouped_data['IndependentValue'], y=mean, ax=ax, label='Mean', color='purple')

            # Fill the confidence interval
            ax.fill_between(grouped_data['IndependentValue'], np.array(mean) - np.array(ci),
                            np.array(mean) + np.array(ci), alpha=0.2, label='95\% CI', color='purple')

            # Set subplot title and labels
            ax.set_title(f'{independent_var.capitalize()} vs {dependent_var}', fontsize=18)
            ax.set_xlabel(x_labels[i] if x_labels else f'{independent_var.capitalize()}', fontsize=16)
            ax.set_ylabel(f'{dependent_var.capitalize()}', fontsize=16)  # Set ylabel for each subplot
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(fontsize=16)

    else:  # Single subplot case
        # Filter data for the current independent variable
        independent_var = selected_independent_vars[0]
        subset = data[data['IndependentVariable'] == independent_var]

        # Group data by the independent variable and calculate mean and CI
        grouped_data = subset.groupby('IndependentValue')[dependent_var].agg(calculate_ci).reset_index()

        # Unpack the calculated values
        mean, ci = zip(*grouped_data[dependent_var])

        # Plot the mean line
        sns.lineplot(x=grouped_data['IndependentValue'], y=mean, ax=axes, label='Mean', color='purple')

        # Fill the confidence interval
        axes.fill_between(grouped_data['IndependentValue'], np.array(mean) - np.array(ci),
                         np.array(mean) + np.array(ci), alpha=0.2, label='95\% CI', color='purple')

        # Set subplot title and labels
        dependent_var = dependent_var.replace("M: ", "")
        axes.set_title(f'{independent_var.capitalize()} vs {dependent_var}', fontsize=18)
        axes.set_xlabel(x_labels[0] if x_labels else f'{independent_var.capitalize()}', fontsize=16)
        axes.set_ylabel(f'{dependent_var.capitalize()}', fontsize=16)  # Set ylabel for the single subplot
        axes.tick_params(axis='both', which='major', labelsize=14)

        # Display legend
        axes.legend(fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    path = utils.make_path("Figures", chapter, f"ofat_{dependent_var}_{selected_independent_vars}")
    plt.savefig(path)
    plt.close()



def plot_network_measures(full_model_data, independent_variable, x_label, chapter):
    '''
    Description: 
    Plot the mean and confidence interval of network measures against an independent variable using one-factor-at-a-time (OFAT) analysis.
    Inputs:
    - full_model_data (DataFrame): The DataFrame containing the full model data.
    - independent_variable (str): The independent variable to plot against.
    - x_label (str): The label for the x-axis.
    - chapter (str): The name of the chapter or section to save the plot.
    Outputs:
    - None (plots are saved as image files).
    '''
    # Filter data for the given independent variable
    data = full_model_data[full_model_data['IndependentVariable'] == independent_variable]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot each network measure
    for i, measure in enumerate(['M: Avg Clustering', 'M: Avg Path Length', 'M: Var of Degree']):
        # Group data by independent variable and run, then calculate mean and standard deviation
        subset = data[data['IndependentVariable'] == independent_variable]

        # Group data by the independent variable and calculate mean and CI
        grouped_data = subset.groupby('IndependentValue')[measure].agg(calculate_ci).reset_index()

        # Unpack the calculated values
        mean, ci = zip(*grouped_data[measure])

        # Plot the mean line
        sns.lineplot(x=grouped_data['IndependentValue'], y=mean, ax=axes[i], label='Mean', color = "m")

        # Fill the confidence interval
        axes[i].fill_between(grouped_data['IndependentValue'], np.array(mean) - np.array(ci),
                         np.array(mean) + np.array(ci), alpha=0.2, label='CI', color = "m")
        measure = measure.replace("M: ", "")
        axes[i].set_title(f'{measure} vs {x_label}', fontsize=18)
        axes[i].set_xlabel(f'{x_label}', fontsize=16)
        axes[i].set_ylabel(measure.split(':')[-1].strip(), fontsize=16)
        axes[i].tick_params(axis='both', which='major', labelsize=14)  
        axes[i].legend(['Mean', '95\% CI'], fontsize=16)


    # Show the plot
    path = utils.make_path("Figures", chapter, f"ofat_{independent_variable}")
    plt.savefig(path)
    plt.close()


def ofat(problem, samples, model_reporters = {}, agent_reporters= {}):
    '''
    Description: 
    Perform one-factor-at-a-time (OFAT) analysis to vary model parameters and generate data.
    Inputs:
    - problem (dict): The problem configuration dictionary containing parameter names and bounds.
    - samples (int): The number of samples to generate.
    - model_reporters (dict, optional): Dictionary of model reporters. Defaults to {}.
    - agent_reporters (dict, optional): Dictionary of agent reporters. Defaults to {}.
    Outputs:
    - full_model_data (DataFrame): The DataFrame containing the full model data.
    - full_agent_data (DataFrame): The DataFrame containing the full agent data.
    '''
    # Prevent mesa's deprecation warnings
    filterwarnings("ignore")

    model = GamesModel  # Remove parentheses here

    # Import parameter configuration from file based on input argument (default: configs.normal)
    params = utils.get_config()
    replicates = params.n_rounds
    max_steps = params.n_steps
    distinct_samples = samples

    full_model_data = pd.DataFrame() 
    full_agent_data = pd.DataFrame()


    for idx, var in enumerate(problem['names']):
        print(f'varying {var} for {model_reporters}{agent_reporters}')

        # Generate the samples
        samples = np.linspace(*problem['bounds'][idx], num=distinct_samples)


        batch = BatchRunner(model,
                            max_steps=max_steps,
                            iterations=replicates,
                            variable_parameters={var: samples},
                            model_reporters = model_reporters,
                            agent_reporters = agent_reporters)

        batch.run_all()
                
        # Save the results
        model_data = batch.get_model_vars_dataframe()
        agent_data = batch.get_agent_vars_dataframe()

        model_data['IndependentVariable'] = var
        model_data['IndependentValue'] = model_data[var]

        agent_data['IndependentVariable'] = var
        agent_data['IndependentValue'] = agent_data[var]

        # Drop the column with current var name
        var = var.strip("[]")
        model_data = model_data.drop(var, axis=1)
        agent_data = agent_data.drop(var, axis=1)

        # Add to data
        full_model_data = pd.concat([full_model_data, model_data])
        full_agent_data = pd.concat([full_agent_data, agent_data])

    return full_model_data, full_agent_data

