### GamesModel.py
# Specifies the GamesModel class which is a subclass of a mesa.Model. This 
# represents the environment of the model and handles all global functionality 
# like environment variables, data collection, initializing agents, and calling 
# the agents' step functions each step.
###


from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import inequalipy   
import networkx as nx
import numpy as np


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
                 network = params.default_network, 
                 netRat = 0.1, 
                 partScaleFree = 0, 
                 alwaysOwn = False, 
                 UV = (True, None, None, False), 
                 risk_distribution = "uniform", 
                 utility_function = "isoelastic"):
        
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.netRat = netRat
        self.ratFunct = lambda f : f**2
        self.alwaysSafe = alwaysOwn
        self.utility_function = utility_function
        self.NH = UV[3]
        self.running = True

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


        #//FIXME: this should be  a fixed seed network'
        #save mean degree of network
        self.initial_mean_degree = self.get_mean_degree()

        # Create agents.
        self.agents = np.array([])
        for node in self.graph:
            agent = ga.GameAgent(node, self, rewiring_p, alpha, beta, risk_distribution)
            self.agents = np.append(self.agents, agent)
            self.schedule.add(agent)

        # Collect model timestep data.
        self.datacollector = DataCollector(
            #model_reporters={"Mean Degree" : self.get_mean_degree, "Var of Degree" : self.get_variance_degree, "Avg Clustering" : self.get_clustering_coef, "Game Distribution" : "game_list"},
            model_reporters={"M: Mean Degree" : self.get_mean_degree, "M: Var of Degree" : self.get_variance_degree, "M: Avg Clustering" : self.get_clustering_coef, "Gini Coefficient": self.get_gini_coef,
                             "Unique Games": self.get_unique_games, "Degree Distr": self.get_degree_distribution, "e_n": "e_n", "e_g": "e_g"},
            agent_reporters={"Wealth": "wealth","Player risk aversion": "eta", "UV": "game.UV", "Games played": "games_played", "Neighbours": "nNeighbors"}
        )

    
    def get_mean_degree(self):
        '''
        Description: Gets the mean degree of the model network
        Output: Mean degree network
        '''
        total_degree = sum([x[1] for x in self.graph.degree()])
        return (total_degree / self.graph.number_of_nodes())
    
    def get_degree_distribution(self):
        return [x[1] for x in self.graph.degree()]
    
    
    def get_variance_degree(self):
        '''
        Description: Gets the variance of the degree in the model network
        Output: variance of the degree in the model network
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
    
    
    def get_gini_coef(self):
        wealth = np.array([agent.wealth for agent in self.agents])
        return inequalipy.gini(wealth)


    def get_unique_games(self):
        return list(set([agent.game.UV for agent in self.agents]))
    
    def get_ratio_updating_speed(self):
        if self.e_n == 0 or  self.e_g == 0:
            return 0
        return self.e_g / self.e_n


    def step(self):
        '''
        Description: updates model environment and takes a step for each agent
        '''
        self.schedule.step()
        self.datacollector.collect(self)

        
