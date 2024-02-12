import numpy as np
import pandas as pd
import os
from scipy.stats import entropy


import utils
import Simulate as sim
import Vizualize as viz
import Game 
import ast

params = utils.get_config()

def baselineExperiments():
    path = utils.make_path("Data", "GameChoice", "Baseline")
    if os.path.isfile(path):
        df_agent = pd.read_excel(path)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
    else:
        _, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False)
        df_agent.to_excel(path, index=False)

    viz.viz_UV_heatmap(df_agent)
    viz.viz_time_series_agent_data_rationality_single(df_agent)
    viz.viz_time_series_agent_data_pay_off_single(df_agent)
    viz.plot_wealth_distribution(df_agent)
    viz.plot_cml_wealth(df_agent)
    viz.viz_corrrelation(df_agent)

def baselineExperiments_NH():
    "The baseline experiment but with only the non-harmonious (NH) games included"
    path = utils.make_path("Data", "GameChoice", "Baseline_NH")
    if os.path.isfile(path):
        df_agent = pd.read_excel(path)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
    else:
        _, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,None,None,True))
        df_agent.to_excel(path, index = False)

    viz.viz_UV_heatmap(df_agent, True)
    viz.viz_time_series_agent_data_rationality_single(df_agent, True)
    viz.viz_time_series_agent_data_pay_off_single(df_agent, True)
    viz.plot_wealth_distribution(df_agent, True)
    viz.plot_cml_wealth(df_agent, True)
    viz.viz_corrrelation(df_agent, True)


def effect_of_risk_distribution_on_wealth():
    path = utils.make_path("Data", "GameChoice", "Influence_risk_distribution")
    if os.path.isfile(path):
        df_agent = pd.read_excel(path)
    else:
        distributions = ["uniform", "log_normal", "gamma", "exponential"]
        df_agent = pd.DataFrame()
        for distribution in distributions:
            _, df_agent_partial = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, risk_distribution = distribution)
            df_agent_partial['Rationality Distribution'] = distribution
            # Add partial data to the network that needsa to contain the full data
            df_agent = pd.concat([df_agent, df_agent_partial])


        df_agent.to_excel(path, index=False)
        
    viz.viz_time_series_agent_data_rationality_for_rat_dist(df_agent)
    viz.viz_time_series_agent_data_pay_off_for_rat_dist(df_agent)        

def effect_of_utility_function_on_wealth():
    path = utils.make_path("Data", "GameChoice", "Influence_Utility_Function_on_Wealth")
    if os.path.isfile(path):
        df_agent = pd.read_excel(path)
    else:
        utility_functions = ['isoelastic', 'linex']
        df_agent = pd.DataFrame()
        for utility_function in utility_functions:
            _, df_agent_partial = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, utility_function = utility_function)
            df_agent_partial['Utility Function'] = utility_function
            # Add partial data to the network that needsa to contain the full data
            df_agent = pd.concat([df_agent, df_agent_partial])


        df_agent.to_excel(path, index=False)
        
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
        df = pd.read_excel(path)
    else:
        mean_range = [0, 5]
        std_dev_range = [0.2, 0.8]
        num_trials = 50

        results = []

        for _ in range(num_trials):
            uvpay = np.random.rand(2) * 2
            game = Game.Game((uvpay[0], uvpay[1]))

            # Vary mean and standard deviation for lambda initialization
            mean_lambda = np.random.uniform(mean_range[0], mean_range[1])
            std_dev_lambda = np.random.uniform(std_dev_range[0], std_dev_range[1])

            # Initialize lambdas using a lognormal distribution with varying mean and standard deviation
            lambda1 = np.random.lognormal(mean_lambda, std_dev_lambda)
            lambda2 = np.random.lognormal(mean_lambda, std_dev_lambda)

            # Call your QRE function with the initialized lambdas
            qre_result = game.getQreChance(rationalityA=lambda1, rationalityB=lambda2, etaA = np.random.rand()*2, etaB = np.random.rand()*2, utility_function= "isoelastic")

            # Append the results for analysis
            results.append({
                'mean_lambda': mean_lambda,
                'std_dev_lambda': std_dev_lambda,
                'qre_result': qre_result[0]  # Assuming you're interested in the QRE result for player 1
            })

        # Extract data for plotting
        mean_lambda_values = np.array([result['mean_lambda'] for result in results])
        std_dev_lambda_values = np.array([result['std_dev_lambda'] for result in results])
        qre_results = np.array([result['qre_result'] for result in results])


        # Create a DataFrame for easier manipulation
        df = pd.DataFrame({'Mean Lambda': mean_lambda_values, 'Std Dev Lambda': std_dev_lambda_values, 'QRE Result': qre_results})
        df.to_excel(path, index=False)
    viz.vizualize_effect_of_rationality_on_QRE(df)

def game_wealth_rationality_correlation():
    path = utils.make_path("Data", "GameChoice", "Game_Wealth_Risk_Correlation")
    if os.path.isfile(path):
        df_agent = pd.read_excel(path)
    else:
        df_agent = pd.DataFrame()
        _ , df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)

        df_agent.to_excel(path, index=False)

    
    viz.vizualize_game_correlation(df_agent, 'Player risk aversion')
    viz.vizualize_game_correlation(df_agent, 'Wealth')


def track_num_games_in_pop():
    path = utils.make_path("Data", "GameChoice", "Games_in_Population")
    if os.path.isfile(path):
        df_model = pd.read_excel(path)
        df_model['Unique Games'] = df_model['Unique Games'].apply(ast.literal_eval)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)

        df_model.to_excel(path, index=False)

    df_model['Num Unique Games'] = df_model['Unique Games'].apply(lambda x: len(x))
    viz.vizualize_measure_over_time(df_model, 'Num Unique Games')

def gini_over_time():
    path = utils.make_path("Data", "GameChoice", "Gini_Time_series")
    if os.path.isfile(path):
        df_model = pd.read_excel(path)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)
        df_model.to_excel(path, index=False)

    viz.vizualize_measure_over_time(df_model, "Gini Coefficient")



            







    



