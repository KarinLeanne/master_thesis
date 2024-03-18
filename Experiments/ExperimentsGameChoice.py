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
        df_agent = pd.read_excel(path_agent)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
        df_model = pd.read_excel(path_model)
    else:
        df_model, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,None,None,False))
        df_agent.to_excel(path_agent, index=False)
        df_model.to_excel(path_model, index=False)

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
        df_agent = pd.read_excel(path_agent)
        df_agent['UV'] = df_agent['UV'].apply(ast.literal_eval)
        df_model = pd.read_excel(path_model)
    else:
        df_model, df_agent = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, netRat = 0.1, partScaleFree = 1, alwaysOwn = False, UV=(True,None,None,True))
        df_agent.to_excel(path_agent, index = False)
        df_model.to_excel(path_model, index=False)

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
        df.to_excel(path, index=False)
    
    viz.viz_effect_of_rationality_on_QRE(df)


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
    viz.viz_measure_over_time(df_model, 'Num Unique Games')

def gini_over_time():
    path = utils.make_path("Data", "GameChoice", "Gini_Time_series")
    if os.path.isfile(path):
        df_model = pd.read_excel(path)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)
        df_model.to_excel(path, index=False)

    viz.viz_measure_over_time(df_model, "Gini Coefficient")

def run_ofat_GC():
    path_gini = utils.make_path("Data", "GameChoice", "ofat_gini")
    path_wealth = utils.make_path("Data", "GameChoice", "ofat_wealth")
    path_risk_aversion = utils.make_path("Data", "GameChoice", "ofat_risk_aversion")
    path_recent_wealth = utils.make_path("Data", "GameChoice", "ofat_recent_wealth")

    # For Gini Coefficient (model-level)
    if os.path.isfile(path_gini):
        df_ofat_gini = pd.read_excel(path_gini)
    else:
        gini_reporters = {"Gini Coefficient": lambda m: m.get_gini_coef()}
        df_ofat_gini = OFAT.ofat(model_reporters = gini_reporters, level='model')
        df_ofat_gini.to_excel(path_gini, index=False)

    # For Wealth (agent-level)
    if os.path.isfile(path_wealth):
        df_ofat_wealth = pd.read_excel(path_wealth)
    else:
        wealth_reporters = {"Wealth": "wealth"}
        df_ofat_wealth = OFAT.ofat(agent_reporters = wealth_reporters, level='agent')
        df_ofat_wealth.to_excel(path_wealth, index=False)

    # For Player risk aversion (agent-level)
    if os.path.isfile(path_risk_aversion):
        df_ofat_risk_aversion = pd.read_excel(path_risk_aversion)
    else:
        risk_reporters = {"Player risk aversion": "eta"}
        df_ofat_risk_aversion = OFAT.ofat(agent_reporters = risk_reporters, level='agent')
        df_ofat_risk_aversion.to_excel(path_risk_aversion, index=False)

    # For recent wealth (agent-level)
    if os.path.isfile(path_recent_wealth):
        df_ofat_recent_wealth = pd.read_excel(path_recent_wealth)
    else:
        recent_wealth_reporters = {"Recent Wealth": "recent_wealth"}
        df_ofat_recent_wealth = OFAT.ofat(agent_reporters = recent_wealth_reporters, level='agent')
        df_ofat_recent_wealth.to_excel(path_recent_wealth, index=False)


    # Vizualize the ofat
    OFAT.plot_vs_independent('GameChoice', df_ofat_gini, "Gini Coefficient")
    OFAT.plot_vs_independent('GameChoice', df_ofat_wealth, "Wealth")
    OFAT.plot_vs_independent('GameChoice', df_ofat_risk_aversion, "Player risk aversion")
    OFAT.plot_vs_independent('GameChoice', df_ofat_recent_wealth, "Recent Wealth")



            







    




