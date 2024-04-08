'''
GSA.py
Contains code to perform and plot the global sensitivity analysis
'''

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunner
from IPython.display import clear_output
from matplotlib import pyplot as plt
from itertools import combinations
from warnings import filterwarnings
import seaborn as sns
import os
from tabulate import tabulate
import scienceplots

import utils
from Simulate import simulate
from GamesModel import GamesModel

params = utils.get_config()
filterwarnings("ignore")
plt.style.use(['science', 'ieee'])


def plot_index(s, params, i, title=''):
    """
    Description: creates a plot for Sobol sensitivity analysis that shows the contributions
                 of each parameter to the global sensitivity.
    Inputs:
        - s: dictionary of dictionaries that hold
             the values for a set of parameters
        - params: the parameters taken from s
        - i: string that indicates what order the sensitivity is.
        - title: title for the plot
    """

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']

    l = len(indices)
    plt.yticks(range(l), params, fontsize = 20)

    
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o', markersize=14, capsize=10, color = "purple")
    plt.axvline(0, c='k', linestyle='--', color = "purple")

    plt.xlabel('Sensitivity Index', fontsize=24)
    plt.ylabel('Parameters', fontsize=24)
    
    plt.tick_params(axis='x', which='major', labelsize=18)  
    plt.title(title, fontsize=30)

    # Safe Figure
    path = utils.make_path("Figures", "Sobol", f"{title}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_global(Si, problem, title=''):
    '''
    Description: plots the first and total order sensitivity of parameters
    Inputs:
        - Si: sensitivity
        - problem: dictionary of parameters to perform sensitivity analysis on
        - title: title for the plot
    '''

    # First order
    plot_index(Si, problem['names'], '1', f'First order sensitivity - {title}')
    
    # Total order
    plot_index(Si, problem['names'], 'T', f'Total order sensitivity - {title}')
    

def sensitivity_analysis_for_variable(problem, data, output_variable, data_column):
    # Set calc_second_order_sampling based on the actual option used during sampling
    calc_second_order_sampling = True

    # Perform Sobol analysis with the correct calc_second_order option
    Si = sobol.analyze(problem, data[data_column].values, calc_second_order=calc_second_order_sampling)

    # Print sensitivity indices
    print(f"{output_variable} - First-order indices:", Si['S1'])
    print(f"{output_variable} - Total-order indices:", Si['ST'])
    plot_global(Si, problem, title=output_variable)

def global_sensitivity_analysis():
    path_agent = utils.make_path("Data", "Sobol", "Sobol_agent")
    path_model = utils.make_path("Data", "Sobol", "Sobol_model")

    # Define parameter ranges
    n_samples = 6 
    problem = {
        'num_vars': 4,
        'names': ['rewiring_p', 'alpha', 'beta', 'rat'],
        'bounds': [[0, 1], [0, 1], [0, 1], [0, 5]]
        }

    if (os.path.isfile(path_agent) and os.path.isfile(path_model)):
        full_data_agent = pd.read_csv(path_agent)
        full_data_model = pd.read_csv(path_model)
    else:

        # Generate Sobol samples
        param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

        # Initialize DataFrame with NaN values
        full_data_model = pd.DataFrame()
        full_data_agent = pd.DataFrame()

        # Define model reporters
        model_reporters = {
            "M: Mean Degree": lambda m: m.get_mean_degree(),
            "M: Var of Degree": lambda m: m.get_variance_degree(),
            "M: Avg Clustering": lambda m: m.get_clustering_coef(),
            "M: Avg Path Length": lambda m: m.get_average_path_length(),
            "Gini Coefficient": lambda m: m.get_gini_coef()
        }

        agent_reporters = {
            "Wealth": "wealth", 
            "Player Risk Aversion": "eta",
            "Recent Wealth": "recent_wealth"
        }

        batch = BatchRunner(GamesModel, 
                            max_steps=params.n_steps,
                            variable_parameters={name: [] for name in problem['names']},
                            agent_reporters=agent_reporters,
                            model_reporters=model_reporters)
        # Initialize run count
        run_count = 0  

        # Run the model for each set of parameters
        for values in param_values:
            prmvalues = list(values)
            variable_parameters = {name: val for name, val in zip(problem['names'], prmvalues)}

            # Run model
            batch.run_iteration(variable_parameters, tuple(values), run_count)

            # Increment run_count
            run_count += 1

            # Get model and agent data
            model_data = batch.get_model_vars_dataframe()
            agent_data = batch.get_agent_vars_dataframe()

            # Append parameter values to model and agent data
            model_data['rewiring_p'] = prmvalues[0]
            model_data['alpha'] = prmvalues[1]
            model_data['beta'] = prmvalues[2]
            model_data['rat'] = prmvalues[3]

            agent_data['rewiring_p'] = prmvalues[0]
            agent_data['alpha'] = prmvalues[1]
            agent_data['beta'] = prmvalues[2]
            agent_data['rat'] = prmvalues[3]


            # Concatenate model and agent data with full data
            full_data_model = pd.concat([full_data_model, model_data], ignore_index=True)
            full_data_agent = pd.concat([full_data_agent, agent_data], ignore_index=True)

            clear_output()
            print(f'running... ({run_count / len(param_values) * 100:.2f}%)', end='\r', flush=True)

        # Save data
        full_data_agent.to_csv(path_agent, index=False)
        full_data_model.to_csv(path_model, index=False)
            

    # Perform Sobol analysis for Wealth
    sensitivity_analysis_for_variable(problem, full_data_agent, "Wealth", "Wealth")

    # Perform Sobol analysis for Wealth
    sensitivity_analysis_for_variable(problem, full_data_agent, "Recent Wealth", "Recent Wealth")

    # Perform Sobol analysis for eta
    sensitivity_analysis_for_variable(problem, full_data_agent, "Risk Aversion", "Player Risk Aversion")

    # Perform Sobol analysis for Gini Coefficient
    sensitivity_analysis_for_variable(problem, full_data_model,"Gini Coefficient", "Gini Coefficient")