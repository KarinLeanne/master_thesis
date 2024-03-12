import os
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from GamesModel import GamesModel
from tabulate import tabulate
import utils
from scipy.stats import sem

def calculate_ci(data):
    mean = data.mean()
    ci = 1.96 * sem(data)
    return mean, ci

def plot_vs_independent(data, dependent_var):
    independent_vars = data['IndependentVariable'].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(independent_vars), figsize=(15, 5), sharey=True)

    for i, independent_var in enumerate(independent_vars):
        # Filter data for the current independent variable
        subset = data[data['IndependentVariable'] == independent_var]

        # Group data by the independent variable and calculate mean and CI
        grouped_data = subset.groupby('IndenpendentValue')[dependent_var].agg(calculate_ci).reset_index()

        # Unpack the calculated values
        mean, ci = zip(*grouped_data[dependent_var])

        # Plot the mean line
        sns.lineplot(x=grouped_data['IndenpendentValue'], y=mean, ax=axes[i], label='Mean')

        # Fill the confidence interval
        axes[i].fill_between(grouped_data['IndenpendentValue'], np.array(mean) - np.array(ci),
                             np.array(mean) + np.array(ci), alpha=0.2, label='CI')

        # Set subplot title and labels
        axes[i].set_title(f'{independent_var.capitalize()} vs {dependent_var}')
        axes[i].set_xlabel(f'{independent_var.capitalize()}')

        # Display legend in the last subplot
        if i == len(independent_vars) - 1:
            axes[i].legend()

    # Set common ylabel
    axes[0].set_ylabel(f'{dependent_var.capitalize()}')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def save_data_to_excel(data, dependent_variable, chapter, level='model'):
    # Save data to Excel file
    path = utils.make_path("Data", chapter, f"{chapter}_{level}_{dependent_variable}.xlsx")
    data.to_excel(path, index=False)

def ofat(chapter, dependent_variable, model_reporters = {}, agent_reporters= {}, level='model'):
    # Prevent mesa's deprecation warnings
    filterwarnings("ignore")

    path = utils.make_path("Data", "Networks", chapter)

    if os.path.isfile(os.path.join(path, f"{chapter}_{level}_{dependent_variable}.xlsx")):
        # Read data from Excel file if it exists
        data = pd.read_excel(os.path.join(path, f"{chapter}_{level}_{dependent_variable}.xlsx"))
    else:
        model = GamesModel  # Remove parentheses here

        # Import parameter configuration from file based on input argument (default: configs.normal)
        params = utils.get_config()
        replicates = params.n_rounds
        max_steps = params.n_steps
        distinct_samples = 4
        problem = params.problem

        data = pd.DataFrame()  # Initialize an empty DataFrame
        # ...

        for idx, var in enumerate(problem['names']):
            print(f'varying {var}')

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
            if level == 'model':
                current_data = batch.get_model_vars_dataframe()
            elif level == 'agent':
                current_data = batch.get_agent_vars_dataframe()

            current_data['IndependentVariable'] = var
            current_data['IndenpendentValue'] = current_data[var]


            # Drop the column with current var name
            var = var.strip("[]")
            current_data = current_data.drop(var, axis=1)

            # Add to data
            data = pd.concat([data, current_data])

        # Plot the results
        return data

