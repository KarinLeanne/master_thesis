### config.py
# A configuration file with hyperparameters.
###

n_steps = 400              # Must be a multiple of 8
n_agents = 500
n_rounds = 5

rewiring_p = 0.6           # Has to be between 0 and 1
alpha = 0.5                # alpha is the homophilic parameter
beta = 0.1                 # beta is the homophilic parameter


networks = [("HK", 4, 1), ("RR", 4, 1), ("WS", 4, 1)]
default_network = ("WS", 4, 1)

