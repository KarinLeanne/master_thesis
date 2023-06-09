from importlib import import_module

def get_config():
    '''
    description: retrieves hyperparameters from config file (uses configs/normal.py if none given)
    outputs:
        - hyperparameters form config file
    '''
    return import_module('config')