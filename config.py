### config.py
# A configuration file with hyperparameters.
###

n_steps = 10              # Must be a multiple of 8
n_agents = 10
n_rounds = 2

rewiring_p = 0.6           # Has to be between 0 and 1
alpha = 0.5                # alpha is the homophilic parameter
beta = 0.1                 # beta is the homophilic parameter


networks = [("HK", 4, 1), ("RR", 4, 1), ("WS", 4, 0.3)]
default_network = ("WS", 4, 0.3)

distinct_samples = 4 
problem = {
        'num_vars': 3,
        'names': ['rewiring_p', 'alpha', 'beta'],
        'bounds': [[0.1, 1], [0.1, 1], [0, 1]]
        }

