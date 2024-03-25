import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
import ast


import utils
import Simulate as sim
import Game 
import Experiments.vizGameChoice as viz
import Experiments.OFAT as OFAT

params = utils.get_config()

def baselineExperiments():
    '''
    Description: calculates how much a certain characteristic should change when agent
                     is subjected to a stimulus
    Inputs:
        - characteristics: name of the characteristic to modify
    Outputs:
        - modification to characteristics
        '''
    path_agent = utils.make_path("Data", "GameChoice", "Agent_Baseline")
    path_model = utils.make_path("Data", "GameChoice", "Model_Baseline")
    if os.path.isfile(path_agent) and os.path.isfile(path_model):
        df_agent = pd.read_csv(path_agent)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
        df_model = pd.read_csv(path_model)
    else:
        df_model, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,None,None,False))
        df_agent.to_csv(path_agent, index=False)
        df_model.to_csv(path_model, index=False)

    viz.viz_UV_heatmap(df_agent, df_model)
    viz.viz_UV_heatmap(df_agent[df_agent['Step'] < 40], df_model, only_start=True)
    viz.viz_time_series_agent_data_rationality_single(df_agent)
    viz.viz_time_series_agent_data_pay_off_single(df_agent)
    viz.viz_wealth_distribution(df_agent)
    viz.viz_cml_wealth(df_agent)
    viz.viz_corrrelation(df_agent)

def baselineExperiments_NH():
    "The baseline experiment but with only the non-harmonious (NH) games included"
    path_agent = utils.make_path("Data", "GameChoice", "Agent_Baseline_NH")
    path_model = utils.make_path("Data", "GameChoice", "Model_Baseline_NH")
    if os.path.isfile(path_agent) and os.path.isfile(path_model):
        df_agent = pd.read_csv(path_agent)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
        df_model = pd.read_csv(path_model)
    else:
        df_model, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,None,None,True))
        df_agent.to_csv(path_agent, index = False)
        df_model.to_csv(path_model, index=False)

    viz.viz_UV_heatmap(df_agent, df_model)
    viz.viz_UV_heatmap(df_agent[df_agent['Step'] < 40], df_model, True, True)
    viz.viz_time_series_agent_data_rationality_single(df_agent, True)
    viz.viz_time_series_agent_data_pay_off_single(df_agent, True)
    viz.viz_wealth_distribution(df_agent, True)
    viz.viz_cml_wealth(df_agent, True)
    viz.viz_corrrelation(df_agent, True)


def effect_of_risk_distribution_on_wealth():
    path = utils.make_path("Data", "GameChoice", "Influence_risk_distribution")
    if os.path.isfile(path):
        df_agent = pd.read_csv(path)
    else:
        distributions = ["uniform", "log_normal", "gamma", "exponential"]
        df_agent = pd.DataFrame()
        for distribution in distributions:
            _, df_agent_partial = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, risk_distribution = distribution)
            df_agent_partial['Rationality Distribution'] = distribution
            # Add partial data to the network that needsa to contain the full data
            df_agent = pd.concat([df_agent, df_agent_partial])


        df_agent.to_csv(path, index=False)
        
    viz.viz_time_series_agent_data_rationality_for_rat_dist(df_agent)
    viz.viz_time_series_agent_data_pay_off_for_rat_dist(df_agent)        

def effect_of_utility_function_on_wealth():
    path = utils.make_path("Data", "GameChoice", "Influence_Utility_Function_on_Wealth")
    if os.path.isfile(path):
        df_agent = pd.read_csv(path)
    else:
        utility_functions = ['isoelastic', 'linex']
        df_agent = pd.DataFrame()
        for utility_function in utility_functions:
            _, df_agent_partial = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, utility_function = utility_function)
            df_agent_partial['Utility Function'] = utility_function
            # Add partial data to the network that needsa to contain the full data
            df_agent = pd.concat([df_agent, df_agent_partial])


        df_agent.to_csv(path, index=False)
        
    viz.viz_time_series_agent_data_rationality_for_util(df_agent)
    viz.viz_time_series_agent_data_pay_off_for_util(df_agent)     


def calculate_nash_equilibrium(payoff_matrix):
    # Find the best responses for each player
    best_responses_player1 = np.argmax(payoff_matrix[0], axis=1)
    best_responses_player2 = np.argmax(payoff_matrix[1], axis=1)

    # Check for Nash Equilibrium
    nash_equilibria = np.argwhere(best_responses_player1 == best_responses_player2)

    return nash_equilibria

def effect_of_rationality_on_QRE():
    path = utils.make_path("Data", "GameChoice", "Influence_Rationality_on_QRE")
    
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        num_trials = 100
        results = []

        for _ in range(num_trials):
            uvpay = np.random.rand(2) * 2
            game = Game.Game((uvpay[0], uvpay[1]))

            # Vary lambda for both players independently
            lambda1 = np.random.uniform(0, 5)
            lambda2 = np.random.uniform(0, 5)

            # Call QRE function with the initialized lambdas
            qre_result = game.getQreChance(rationalityA=lambda1, rationalityB=lambda2, etaA=np.random.rand()*2, etaB=np.random.rand()*2, utility_function="isoelastic")

            # Append the results for analysis
            results.append({
                'lambda1': lambda1,
                'lambda2': lambda2,
                'qre_result': qre_result[0]  # Assuming you're interested in the QRE result for player 1
            })

        # Extract data for plotting
        lambda1_values = np.array([result['lambda1'] for result in results])
        lambda2_values = np.array([result['lambda2'] for result in results])
        qre_results = np.array([result['qre_result'] for result in results])


        df = pd.DataFrame({'Lambda1': lambda1_values, 'Lambda2': lambda2_values, 'QRE Result': qre_results})
        df.to_csv(path, index=False)
    
    viz.viz_effect_of_rationality_on_QRE(df)


def game_wealth_rationality_correlation():
    path = utils.make_path("Data", "GameChoice", "Game_Wealth_Risk_Correlation")
    if os.path.isfile(path):
        df_agent = pd.read_csv(path)
    else:
        df_agent = pd.DataFrame()
        _ , df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)

        df_agent.to_csv(path, index=False)

    
    viz.vizualize_game_correlation(df_agent, 'Player Risk Aversion')
    viz.vizualize_game_correlation(df_agent, 'Wealth')


def track_num_games_in_pop():
    path = utils.make_path("Data", "GameChoice", "Games_in_Population")
    if os.path.isfile(path):
        df_model = pd.read_csv(path)
        df_model['Unique Games'] = df_model['Unique Games'].apply(ast.literal_eval)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)

        df_model.to_csv(path, index=False)

    df_model['Num Unique Games'] = df_model['Unique Games'].apply(lambda x: len(x))
    viz.viz_measure_over_time(df_model, 'Num Unique Games')

def gini_over_time():
    path = utils.make_path("Data", "GameChoice", "Gini_Time_series")
    if os.path.isfile(path):
        df_model = pd.read_csv(path)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)
        df_model.to_csv(path, index=False)

    viz.viz_measure_over_time(df_model, "Gini Coefficient")


def run_default_data():
    # Only obtain data if data does not already exist
    path_agent = utils.make_path("Data", "GameChoice", "Default_Sim_Agent")
    path_model = utils.make_path("Data", "GameChoice", "Default_Sim_Model")

    if os.path.isfile(path_agent) and os.path.isfile(path_model):
        df_agent = pd.read_csv(path_agent)
        df_model = pd.read_csv(path_model)
    else:
        df_model, df_agent = sim.simulate(N = params.n_agents, rewiring_p = params.rewiring_p, alpha=params.alpha, beta=params.beta, network=params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False)
        df_agent.to_csv(path_agent, index=False)
        df_model.to_csv(path_model, index=False)

    viz.viz_histogram_over_time(df_agent, "Games played", bins=40)
    viz.viz_Degree_Distr(df_model, "Degree Distr", bins=10)

def run_ofat_GC():
    path_ofat_model = utils.make_path("Data", "GameChoice", "df_ofat_model")
    path_ofat_agent = utils.make_path("Data", "GameChoice", "df_ofat_agent")

    if (os.path.isfile(path_ofat_agent) and os.path.isfile(path_ofat_model)):
        df_ofat_agent = pd.read_csv(path_ofat_agent)
        df_ofat_model = pd.read_csv(path_ofat_model)
    else:
        # Define model reporters
        model_reporters = {
            "Gini Coefficient": lambda m: m.get_gini_coef()
            }

        agent_reporters = {
            "Wealth": "wealth", 
            "Player Risk Aversion": "eta",
            "Recent Wealth": "recent_wealth"
            }
        
        # Define the problem
        distinct_samples = 6 
        problem = {
        'num_vars': 1,
        'names': ['rat'],
        'bounds': [[0, 5]]
        }

        # Run OFAT for all reporters
        df_ofat_model, df_ofat_agent = OFAT.ofat(problem, distinct_samples, model_reporters=model_reporters, agent_reporters=agent_reporters)

        # Save the results to respective paths
        df_ofat_model.to_csv(path_ofat_model, index=False)
        df_ofat_agent.to_csv(path_ofat_agent, index=False)

    # Vizualize the ofat
    OFAT.plot_vs_independent('GameChoice', df_ofat_model, "Gini Coefficient")
    OFAT.plot_vs_independent('GameChoice', df_ofat_agent, "Wealth")
    OFAT.plot_vs_independent('GameChoice', df_ofat_agent, "Player Risk Aversion")
    OFAT.plot_vs_independent('GameChoice', df_ofat_agent, "Recent Wealth")







            







    




