import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
import ast
import matplotlib.pyplot as plt
import seaborn as sns



import utils
import Simulate as sim
import Game 
import Experiments.vizGameChoice as viz
import Experiments.OFAT as OFAT
import Experiments.StatisticsGameChoice as sgc

params = utils.get_config()

def baselineExperiments():
    '''
    Description: 
    Run baseline experiments and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
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

    viz.viz_coevolution_UV_risk(df_agent)
    viz.viz_UV_Wealth(df_agent)
    viz.viz_UV_Recent_Wealth(df_agent)

    viz.viz_cml_wealth(df_agent)
    viz.viz_cml_wealth(df_agent)

    viz.viz_gini_over_time(df_agent)


    viz.viz_UV_heatmap(df_agent, df_model)
    viz.viz_UV_heatmap(df_agent[df_agent['Step'] < 40], df_model, only_start=True)
    viz.viz_time_series_agent_data_rationality_single(df_agent)
    viz.viz_time_series_agent_data_pay_off_single(df_agent)
    viz.viz_time_series_agent_data_recent_wealth_single(df_agent)
    viz.viz_wealth_distribution(df_agent)
    viz.viz_cml_wealth(df_agent)
    viz.viz_corrrelation(df_agent)

    sgc.coevolution_risk_UV(df_agent, "Player Risk Aversion")
    sgc.coevolution_risk_UV(df_agent, "Wealth")
    sgc.coevolution_risk_UV(df_agent, "Recent Wealth")
    sgc.distributions_statistics(df_agent)




def baselineExperiments_NH():
    '''
    Description: 
    Run baseline experiments with only non-harmonious (NH) games included and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
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
    viz.viz_time_series_agent_data_recent_wealth_single(df_agent, True)
    viz.viz_wealth_distribution(df_agent, True)
    viz.viz_cml_wealth(df_agent, True)
    viz.viz_corrrelation(df_agent, True)


def effect_of_risk_distribution_on_wealth():
    '''
    Description: 
    Investigate the effect of risk distribution on wealth and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
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
    '''
    Description: 
    Investigate the effect of utility function on wealth and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    path = utils.make_path("Data", "GameChoice", "Influence_Utility_Function_on_Wealth")
    if os.path.isfile(path):
        df_agent = pd.read_csv(path)
    else:
        utility_functions = ['isoelastic', 'linex']
        df_agent = pd.DataFrame()
        for utility_function in utility_functions:
            _, df_agent_partial = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network = params.default_network, rounds = params.n_rounds, steps = params.n_steps, utility_function = utility_function)
            df_agent_partial['Utility Function'] = utility_function
            # Add partial data to the network that needs to contain the full data
            df_agent = pd.concat([df_agent, df_agent_partial])


        df_agent.to_csv(path, index=False)
        
    viz.viz_time_series_agent_data_rationality_for_util(df_agent)
    viz.viz_time_series_agent_data_pay_off_for_util(df_agent)     


def calculate_nash_equilibrium(payoff_matrix):
    '''
    Description: 
    Calculate the Nash equilibrium for a given payoff matrix.
    Inputs:
        payoff_matrix: The payoff matrix of the game.
    Outputs:
        nash_equilibria: A list of tuples representing the Nash equilibrium strategies.
    '''
    # Find the best responses for each player
    best_responses_player1 = np.argmax(payoff_matrix[0], axis=1)
    best_responses_player2 = np.argmax(payoff_matrix[1], axis=1)

    # Check for Nash Equilibrium
    nash_equilibria = np.argwhere(best_responses_player1 == best_responses_player2)

    return nash_equilibria

def effect_of_risk_and_rationality_on_QRE():
    '''
    Description: 
    Investigate the effect of risk and rationality on Quantal Response Equilibrium (QRE) and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    path = utils.make_path("Data", "GameChoice", "Influence_Risk_and_Rationality_on_QRE")
    
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        num_trials = 100
        lambda_values = [0, 1, 2, 3, 4,5,6,7,8]  # Specify the lambda values to test
        eta_values = [0, 1, 2, 3, 4,5,6,7,8]  # Specify the eta values to test
        results = []

        for lambda1 in lambda_values:
            for lambda2 in lambda_values:
                for etaA in eta_values:
                    for etaB in eta_values:
                        trial_results = []

                        for _ in range(num_trials):
                            uvpay = np.random.rand(2) * 2
                            game = Game.Game((uvpay[0], uvpay[1]))

                            # Call QRE function with the initialized parameters
                            qre_result = game.getQreChance(rationalityA=lambda1, rationalityB=lambda2,
                                                            etaA=etaA, etaB=etaB, utility_function="isoelastic")

                            # Append the results for analysis
                            trial_results.append(qre_result[0])  # Assuming you're interested in the QRE result for player 1

                        # Compute average QRE result for the current parameter combination
                        avg_qre_result = np.mean(trial_results)

                        # Append the average result for the current parameter combination
                        results.append({
                            'lambda1': lambda1,
                            'lambda2': lambda2,
                            'etaA': etaA,
                            'etaB': etaB,
                            'avg_qre_result': avg_qre_result
                        })

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
    
    viz.viz_effect_of_risk_and_rationality_on_QRE(df)



def game_wealth_rationality_correlation():
    '''
    Description: 
    Analyze the correlation between game wealth and player risk aversion and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
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
    '''
    Description: 
    Track the number of unique games in the population over time and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
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
    '''
    Description: 
    Calculate the Gini coefficient over time and visualize the results.
    Inputs:
        None
    Outputs:
        None (plots are saved as image files).
    '''
    path = utils.make_path("Data", "GameChoice", "Gini_Time_series")
    if os.path.isfile(path):
        df_model = pd.read_csv(path)
    else:
        df_model = pd.DataFrame()
        df_model, _ = sim.simulate(params.n_agents, params.rewiring_p, params.alpha, params.beta, network =params.default_network, rounds = params.n_rounds, steps = params.n_steps)
        df_model.to_csv(path, index=False)

    viz.viz_measure_over_time(df_model, "Gini Coefficient")


def run_ofat_GC():
    '''
    Description: 
    Run One-Factor-At-a-Time (OFAT) analysis for game choice simulation data and visualize the results.
    Inputs:
        None
    Outputs:
        None 
    '''
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








            







    




