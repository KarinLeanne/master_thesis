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

import utils
from Simulate import simulate
from GamesModel import GamesModel

params = utils.get_config()
filterwarnings("ignore")

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

    plt.title(title, fontsize=16)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params, fontsize=12)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o', markersize=8, capsize=5)
    plt.axvline(0, c='k', linestyle='--')

    plt.xlabel('Sensitivity Index', fontsize=14)
    plt.ylabel('Parameters', fontsize=14)

    # Safe Figure
    path = utils.make_path("Figures", "Sobol", f"{title}")
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
    

def global_sensitivity_analysis():
    path = utils.make_path("Data", "Sobol", "Sobol")

    # Define parameter ranges
    problem = {
        'num_vars': 3,
        'names': ['rewiring_p', 'alpha', 'beta'],
        'bounds': [[0.1, 1], [0.1, 1], [0, 1]]
        }

    if os.path.isfile(path):
        data = pd.read_excel(path)
    else:
        # Generate Sobol samples
        n_samples = 4
        param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

        # Initialize DataFrame with NaN values
        data_columns = ['alpha', 'beta', 'rewiring_p', 'Run', 'Wealth', 'eta', "Gini_Coefficient"]
        data = pd.DataFrame(index=range(params.n_steps * len(param_values)), columns=data_columns)

        # Add this line to properly initialize 'Gini Coefficient' column
        data['Gini_Coefficient'] = np.nan



        batch = BatchRunner(GamesModel, 
                    max_steps=params.n_steps,
                    variable_parameters={name: [] for name in problem['names']},
                    agent_reporters={"Wealth": "wealth", "Player risk aversion": "eta"},
                    model_reporters={"Gini Coefficient": GamesModel.get_gini_coef})


        # Run the model for each set of parameters
        count = 0
        for _ in range(params.n_steps):
            for values in param_values:
                prmvalues = list(values)
                variable_parameters = {name: val for name, val in zip(problem['names'], prmvalues)}

                # Run model
                batch.run_iteration(variable_parameters, tuple(values), count)

                # Save results directly to data DataFrame
                data.iloc[count, 0:3] = prmvalues
                data.iloc[count, 3:8] = count, batch.get_agent_vars_dataframe()['Wealth'].iloc[count], batch.get_agent_vars_dataframe()['Player risk aversion'].iloc[count], batch.get_model_vars_dataframe()['Gini Coefficient'].iloc[count]

                count += 1
                clear_output()
                print(f'running... ({count / (len(param_values) * (params.n_steps)) * 100:.2f}%)', end='\r', flush=True)


        # Save data
        data.to_excel(path, index=False)
            
    
    # Perform Sobol analysis for Wealth
    Si_wealth = sobol.analyze(problem, data['Wealth'].values, calc_second_order=True)

    # Print sensitivity indices for Wealth
    print("Wealth - First-order indices:", Si_wealth['S1'])
    print("Wealth - Total-order indices:", Si_wealth['ST'])
    plot_global(Si_wealth, problem, title='Wealth')

    # Perform Sobol analysis for eta
    Si_eta = sobol.analyze(problem, data['eta'].values, calc_second_order=True)

    # Print sensitivity indices for eta
    print("eta - First-order indices:", Si_eta['S1'])
    print("eta - Total-order indices:", Si_eta['ST'])
    plot_global(Si_eta, problem, title='Risk Aversion')

    # Perform Sobol analysis for Gini Coefficient
    Si_gini = sobol.analyze(problem, data['Gini_Coefficient'].values, calc_second_order=True)

    # Print sensitivity indices for Gini Coefficient
    print("Gini Coefficient - First-order indices:", Si_gini['S1'])
    print("Gini Coefficient - Total-order indices:", Si_gini['ST'])
    plot_global(Si_gini, problem, title='Gini Coefficient')