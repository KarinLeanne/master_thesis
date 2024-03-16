from importlib import import_module
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
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.xlsx"
    else:
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png"

    return result_path

def string_to_list(data):
    if isinstance(data.values[0], str):
        data = data.apply(ast.literal_eval).tolist()
    return data