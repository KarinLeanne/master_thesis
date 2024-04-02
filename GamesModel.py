'''
GamesModel.py
Specifies the GamesModel class which is a subclass of a mesa.Model. This 
represents the environment of the model and handles all global functionality 
like environment variables, data collection, initializing agents, and calling 
the agents' step functions each step.
'''

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import inequalipy   
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import GameAgent as ga
import utils

params = utils.get_config()

class GamesModel(Model):
    '''
    Description: The model that will simulate economic games in a network.'
    Inputs:
        - N: The number of agents in a model
        - rewiring_p: The probability that an agent rewires a connection for each timestep
        - alpha: alpha is the homophilic parameter 
        - beta: beta controls homophily together with alpha
        - network: A truple sepcifying the type of network
        - netRat: A parameter of agent rationality
        - partScaleFree: 
        - AlwaysOwn: Boolean, if true always choose the own strategy
        - UV: A truple specifing whether the UV space should be generated randomly as well as the default values for U and V

    Functions:
        - get_mean_degree(): Gets the mean degree of the model network
        - get_variance_degree(): Gets the variance of the degree in the model network
        - get_clustering_coefficient(): Gets the clustering coefficent of the model network
        - step(): updates model environment and takes a step for each agent

    '''

    def __init__(self,
                 N = params.n_agents, 
                 rewiring_p = params.rewiring_p, 
                 alpha = params.alpha, 
                 beta = params.beta, 
                 rat = params.rat,
                 network = params.default_network, 
                 partScaleFree = 0, 
                 alwaysOwn = False, 
                 UV = (True, None, None, False), 
                 risk_distribution = "uniform", 
                 utility_function = "isoelastic"):
        '''
        Description: Initializes the model with given parameters.
        Inputs:
            - N: The number of agents in a model
            - rewiring_p: The probability that an agent rewires a connection for each timestep
            - alpha: alpha is the homophilic parameter 
            - beta: beta controls homophily together with alpha
            - rat: A parameter of agent rationality
            - network: A truple specifying the type of network
            - partScaleFree: 
            - alwaysOwn: Boolean, if true always choose the own strategy
            - UV: A truple specifying whether the UV space should be generated randomly as well as the default values for U and V
            - risk_distribution: Type of risk distribution
            - utility_function: Type of utility function
        '''
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.netRat = rat
        self.ratFunct = lambda f : f**2
        self.alwaysSafe = alwaysOwn
        self.utility_function = utility_function
        self.NH = UV[3]
        self.running = True
        self.games = [] 

        # The amount of times the games are updated (i.e the UV space) 
        self.e_g = 0

        # The amount of times the network is updated
        self.e_n = 0
        
        # Generate the network.
        if network[0] == 'RR':
            if network[1]%2:
                network[1] -= 1 
            self.graph = nx.random_regular_graph(network[1], N)
        if network[0] == 'WS':
            self.graph = nx.watts_strogatz_graph(N, network[1], network[2])
        if network[0] == 'HK':
            self.graph = nx.powerlaw_cluster_graph(N, int(network[1]/2), network[2])


        #save mean degree of network
        self.initial_mean_degree = self.get_mean_degree()


        # Create Games using stratified sampling
        if self.NH:
            space = (0, 2, 0, 2)
            blocked_area = (0, 1, 0, 1)
            games = self.stratified_sampling(N, space, blocked_area)
        else:
            space = (0, 2, 0, 2)  # Assuming UV space ranges from 0 to 2
            games = self.stratified_sampling(N, space)

        # Create agents.
        self.agents = np.array([])
        for idx, node in enumerate(self.graph):
            agent = ga.GameAgent(node, self, rewiring_p, alpha, beta,UV=UV, uvpay = games[idx], risk_aversion_distribution = risk_distribution)
            self.agents = np.append(self.agents, agent)
            self.schedule.add(agent)

        # Collect model timestep data.
        self.datacollector = DataCollector(
            model_reporters={"M: Mean Degree" : self.get_mean_degree, "M: Var of Degree" : self.get_variance_degree, "M: Avg Clustering" : self.get_clustering_coef, "M: Avg Path Length" : self.get_average_path_length, "Gini Coefficient": self.get_gini_coef,
                             "Unique Games": self.get_unique_games, "Degree Distr": self.get_degree_distribution, "e_n": "e_n", "e_g": "e_g", "Game data": self.get_game_data},
            agent_reporters={"Wealth": "wealth","Player Risk Aversion": "eta", "UV": "game.UV", "Games played": "games_played", "Recent Wealth": "recent_wealth"}
        )


    def stratified_sampling(self, n_agents, space_range, blocked_area=None):
        '''
        Description: Perform stratified sampling to create games.
        Inputs:
            - n_agents: The number of agents
            - space_range: The range of UV space
            - blocked_area: Area to be excluded from sampling
        Outputs:
            - Sampled UV values
        '''
        # Extract space range boundaries
        x_min, x_max, y_min, y_max = space_range

        # Increase the number of samples along each dimension
        num_samples = int(np.sqrt(n_agents)) * 3

        # Generate stratified samples for x and y coordinates
        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = np.linspace(y_min, y_max, num_samples)

        # Generate all possible combinations of x and y coordinates
        xy_combinations = [(x, y) for x in x_samples for y in y_samples]

        # Remove samples in the blocked area if specified
        if blocked_area:
            x_blocked_min, x_blocked_max, y_blocked_min, y_blocked_max = blocked_area
            xy_combinations = [xy for xy in xy_combinations if not (x_blocked_min <= xy[0] < x_blocked_max and
                                                                    y_blocked_min <= xy[1] < y_blocked_max)]

        # Randomly shuffle the list of combinations
        np.random.shuffle(xy_combinations)

        # Select the first n_agents combinations as the sampled games
        sampled_uv = xy_combinations[:n_agents]

        return sampled_uv
    
    def get_mean_degree(self):
        '''
        Description: Gets the mean degree of the model network
        Output: Mean degree network
        '''
        total_degree = sum([x[1] for x in self.graph.degree()])
        return (total_degree / self.graph.number_of_nodes())
    
    
    def get_degree_distribution(self):
        '''
        Description: Gets the degree distribution of the model network
        Output: Degree distribution of the model network
        '''
        return [x[1] for x in self.graph.degree()]
    
    
    def get_variance_degree(self):
        '''
        Description: Gets the variance of the degree in the model network
        Output: Variance of the degree in the model network
        '''
        degree_list = [x[1] for x in self.graph.degree()]
        mean = self.get_mean_degree()
        return sum((i - mean) ** 2 for i in degree_list) / len(degree_list)
    
    def get_clustering_coef(self):
        '''
        Description: Gets the clustering coefficent of the model network
        Output: Clustering coefficent network 
        '''
        return nx.average_clustering(self.graph)
    
    def get_average_path_length(self):
        '''
        Description: Gets the average path length of the model network
        Output: Average path length of the model network
        '''
        return nx.average_shortest_path_length(self.graph)
    
    def get_gini_coef(self):
        '''
        Description: Gets the Gini coefficient of the wealth distribution among agents
        Output: Gini coefficient of the wealth distribution
        '''
        wealth = np.array([agent.wealth for agent in self.agents])
        return inequalipy.gini(wealth)


    def get_unique_games(self):
        '''
        Description: Gets the unique UV space configurations of games played by agents
        Output: Unique UV space configurations
        '''
        return list(set([agent.game.UV for agent in self.agents]))
    
    def get_ratio_updating_speed(self):
        '''
        Description: Gets the ratio of UV space updating speed to network updating speed
        Output: Ratio of UV space updating speed to network updating speed
        '''
        if self.e_n == 0 or  self.e_g == 0:
            return 0
        return self.e_g / self.e_n
    
    def get_game_data(self):
        '''
        Description: Gets the data of all games played in the model
        Output: Data of all games played
        '''
        game_data = []
        for game in self.games:
            game_data.append([game.name, game.play_count, game.total_payoff, game.UV])
        return game_data

    def step(self):
        '''
        Description: updates model environment and takes a step for each agent
        '''
        self.schedule.step()
        self.datacollector.collect(self)

