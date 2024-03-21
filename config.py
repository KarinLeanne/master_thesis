### config.py
# A configuration file with hyperparameters.
###

n_steps = 560             # Must be a multiple of 8
n_agents = 500
n_rounds = 3

rewiring_p = 0.6           # Has to be between 0 and 1
alpha = 0.5                # alpha is the homophilic parameter
beta = 0.1                 # beta is the homophilic parameter

rat = 0.1

networks = [("HK", 4, 0.3), ("RR", 4, 1), ("WS", 4, 0.2)]
default_network = ("WS", 4, 0.2)

distinct_samples = 4 
problem = {
        'num_vars': 4,
        'names': ['rewiring_p', 'alpha', 'beta', 'rat'],
        'bounds': [[0.1, 1], [0.1, 1], [0, 1], [0,5]]
        }

