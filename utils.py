from importlib import import_module
import pandas as pd
import os
import ast

def get_config():
    '''
    description: retrieves hyperparameters from config file (uses configs/normal.py if none given)
    outputs:
        - hyperparameters form config file
    '''
    return import_module('config')


def make_path(type, chapter, name):
    '''
    description: 
    inputs:
        
    outputs:
        
    '''
    # Make folder if it does not yet exist
    os.makedirs(f"{type}/{chapter}", exist_ok=True) 
    params = get_config()
    # Get path
    if type == "Data":
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.csv"
    else:
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png"

    return result_path

def string_to_list(data):
    if isinstance(data.values[0], str):
        data = data.apply(ast.literal_eval).tolist()
    return data


def xlsx_to_csv(input_folder, output_folder):
    """
    Convert all XLSX files in the input folder to CSV files and save them in the output folder.

    Input:
    - input_folder (str): Path to the folder containing XLSX files.
    - output_folder (str): Path to the folder where CSV files will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all XLSX files in the input folder
    files = [file for file in os.listdir(input_folder) if file.endswith('.xlsx')]

    # Iterate over each XLSX file and convert it to CSV
    for file in files:
        # Construct the input and output file paths
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.xlsx', '.csv'))

        # Read the XLSX file and write it to CSV
        df = pd.read_excel(input_path)
        df.to_csv(output_path, index=False)
