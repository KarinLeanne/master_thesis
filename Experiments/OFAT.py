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

def calculate_ci(data):
    mean = data.mean()
    ci = 1.96 * sem(data)
    return mean, ci

def plot_vs_independent(chapter, data, dependent_var):
    independent_vars = data['IndependentVariable'].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(independent_vars), figsize=(15, 5), sharey=True)

    if isinstance(axes, np.ndarray):  # Check if axes is a numpy array (i.e., multiple subplots)
        for i, ax in enumerate(axes):
            # Filter data for the current independent variable
            independent_var = independent_vars[i]
            subset = data[data['IndependentVariable'] == independent_var]

            # Group data by the independent variable and calculate mean and CI
            grouped_data = subset.groupby('IndependentValue')[dependent_var].agg(calculate_ci).reset_index()

            # Unpack the calculated values
            mean, ci = zip(*grouped_data[dependent_var])

            # Plot the mean line
            sns.lineplot(x=grouped_data['IndependentValue'], y=mean, ax=ax, label='Mean')

            # Fill the confidence interval
            ax.fill_between(grouped_data['IndependentValue'], np.array(mean) - np.array(ci),
                            np.array(mean) + np.array(ci), alpha=0.2, label='CI')

            # Set subplot title and labels
            ax.set_title(f'{independent_var.capitalize()} vs {dependent_var}')
            ax.set_xlabel(f'{independent_var.capitalize()}')
            ax.set_ylabel(f'{dependent_var.capitalize()}')  # Set ylabel for each subplot

            # Display legend in the last subplot
            if i == len(independent_vars) - 1:
                ax.legend()

    else:  # Single subplot case
        # Filter data for the current independent variable
        independent_var = independent_vars[0]
        subset = data[data['IndependentVariable'] == independent_var]

        # Group data by the independent variable and calculate mean and CI
        grouped_data = subset.groupby('IndependentValue')[dependent_var].agg(calculate_ci).reset_index()

        # Unpack the calculated values
        mean, ci = zip(*grouped_data[dependent_var])

        # Plot the mean line
        sns.lineplot(x=grouped_data['IndependentValue'], y=mean, ax=axes, label='Mean')

        # Fill the confidence interval
        axes.fill_between(grouped_data['IndependentValue'], np.array(mean) - np.array(ci),
                         np.array(mean) + np.array(ci), alpha=0.2, label='CI')

        # Set subplot title and labels
        dependent_var = dependent_var.replace("M: ", "")
        axes.set_title(f'{independent_var.capitalize()} vs {dependent_var}')
        axes.set_xlabel(f'{independent_var.capitalize()}')
        axes.set_ylabel(f'{dependent_var.capitalize()}')  # Set ylabel for the single subplot

        # Display legend
        axes.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    path = utils.make_path("Figures", chapter, f"ofat_{dependent_var}")
    plt.savefig(path)
    plt.close()

def ofat(problem, samples, model_reporters = {}, agent_reporters= {}):
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

