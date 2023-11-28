### GamesModel.py
# Specifies the GamesModel class which is a subclass of a mesa.Model. This 
# represents the environment of the model and handles all global functionality 
# like environment variables, data collection, initializing agents, and calling 
# the agents' step functions each step.
###


from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np

import GameAgent as ga


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

    def __init__(self, N, rewiring_p, alpha, beta, network, netRat = 0.1, partScaleFree = 0, alwaysOwn = False, UV = (True, None, None)):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.netRat = netRat
        self.ratFunct = lambda f : f**2
        self.alwaysSafe = alwaysOwn
        

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
        for node in self.graph:
            agent = ga.GameAgent(node, self, rewiring_p, alpha, beta)
            self.schedule.add(agent)

        # Collect model timestep data.
        self.datacollector = DataCollector(
            #model_reporters={"Mean Degree" : self.get_mean_degree, "Var of Degree" : self.get_variance_degree, "Avg Clustering" : self.get_clustering_coef, "Game Distribution" : "game_list"},
            model_reporters={"M: Mean Degree" : self.get_mean_degree, "M: Var of Degree" : self.get_variance_degree, "M: Avg Clustering" : self.get_clustering_coef},
            agent_reporters={"playerPayoff": "totPayoff","Player risk aversion": "eta", "UV": "game.UV"}
        )

    
    def get_mean_degree(self):
        '''
        Description: Gets the mean degree of the model network
        Output: Mean degree network
        '''
        total_degree = sum([x[1] for x in self.graph.degree()])
        return (total_degree / self.graph.number_of_nodes())
    
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

    def step(self):
        '''
        Description: updates model environment and takes a step for each agent
        '''
        self.schedule.step()
        self.datacollector.collect(self)
      
