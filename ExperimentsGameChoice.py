import numpy as np
import pandas as pd

import utils
import Simulate as sim
import Vizualize as viz

params = utils.get_config()

def baselineExperiments():
    _, agent_data = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = ('RR',4,1), rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,1,2))
    viz.viz_UV_heatmap(agent_data)
    viz.viz_time_series_agent_data_rationality(agent_data)
    viz.viz_time_series_agent_data_pay_off(agent_data)
    viz.viz_wealth_distribution(agent_data)
    viz.viz_corrrelation(agent_data)
