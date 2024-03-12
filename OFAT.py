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

def plot_average_with_ci(data, x_var, y_var, chapter, level='model'):
    # ... (unchanged code for plotting)

    # Save Figure
    path = utils.make_path("Figures", chapter, f"{chapter}_{level}_{x_var}_{y_var}")
    plt.savefig(path)
    plt.close()

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

            #print("_______________________________________________")
            # Display the DataFrame after dropping the column
            #print(tabulate(current_data, headers='keys', tablefmt='psql'))
            #print("_______________________________________________")

            # Append the current data to the overall data DataFrame
            data = pd.concat([data, current_data])

        # ...
            
        print(tabulate(data.head(), headers='keys', tablefmt='psql'))

        # Filter the DataFrame to keep only rows where 'IndependentVariable' matches 'IndependentValue'
        #filtered_data = data[data['IndependentVariable'] == data['IndependentValue']]

        # Melt the filtered DataFrame
        #melted_data = pd.melt(filtered_data, id_vars=['IndependentVariable', 'IndependentValue'], var_name='Iteration', value_name='Value')

        #print(tabulate(melted_data.head(), headers='keys', tablefmt='psql'))

        # Save the combined data for the dependent variable
        save_data_to_excel(data, dependent_variable, chapter, level)

        # Plot the results
        plot_average_with_ci(data, 'Parameter', dependent_variable, chapter, level=level)

# Example usage for Gini Coefficient (model-level)
gini_reporters = {"Gini Coefficient": lambda m: m.get_gini_coef()}
ofat('Networks', 'Gini Coefficient', model_reporters = gini_reporters, level='model')

# Example usage for Wealth (agent-level)
wealth_reporters = {"Wealth": "wealth"}
ofat('Networks', 'Wealth', agent_reporters = wealth_reporters, level='agent')

# Example usage for Player risk aversion (agent-level)
risk_reporters = {"Player risk aversion": "eta"}
ofat('Networks', 'Player risk aversion', agent_reporters = risk_reporters, level='agent')
