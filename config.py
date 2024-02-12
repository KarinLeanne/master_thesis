### config.py
# A configuration file with hyperparameters.
###

n_steps = 120              # Must be a multiple of 8
n_agents = 100
n_rounds = 3

rewiring_p = 0.6           # Has to be between 0 and 1
alpha = 0.5                # alpha is the homophilic parameter
beta = 0.1                 # beta is the homophilic parameter


networks = [("HK", 4, 1), ("RR", 4, 1), ("WS", 4, 0.3)]
default_network = ("WS", 4, 0.3)

