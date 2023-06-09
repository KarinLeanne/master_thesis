from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np

import GameAgent as ga


class GamesModel(Model):
    'The model that will simulate economic games in a network.'

    def __init__(self, N, netRat = 0.1, partScaleFree = 0, alwaysSafe = False, UV = (True, None, None), network = ('RR', 4, 2)):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.netRat = netRat
        self.ratFunct = lambda f : f**2
        self.alwaysSafe = alwaysSafe

        # Generate the network.
        if network[0] == 'RR':
            if network[1]%2:
                network[1] -= 1 
            self.graph = nx.random_regular_graph(network[1], N)
        if network[0] == 'WS':
            self.graph = nx.watts_strogatz_graph(N, network[1], network[2])
        if network[0] == 'HK':
            self.graph = nx.powerlaw_cluster_graph(N,int(network[1]/2), network[2])

        #//FIXME: this should be  a fixed seed network


        if UV[0]:
            self.uvpay = np.random.RandomState().rand(N,2)*3 - 1
            game_l = [None] * N
            for i in range(N):
                game_l[i] = [[1,self.uvpay[i][0]],[self.uvpay[i][1],0]]
            self.game_list = game_l
        if not UV[0]:
            game_l = [None] * N
            for i in range(N):
                game_l[i] = [[1,UV[1]],[UV[2],0]]
            self.game_list = game_l
            

        # Create agents.
        for node in self.graph:
            agent = ga.GameAgent(node, self)
            self.schedule.add(agent)

        # Collect model timestep data.
        self.datacollector = DataCollector(
            #model_reporters={"Mean Degree" : self.get_mean_degree, "Var of Degree" : self.get_variance_degree, "Avg Clustering" : self.get_clustering_coef, "Game Distribution" : "game_list"},
            model_reporters={"Mean Degree" : self.get_mean_degree, "Var of Degree" : self.get_variance_degree, "Avg Clustering" : self.get_clustering_coef},
            agent_reporters={"playerPayoff": "totPayoff","Player risk aversion": "eta"}
        )

    
    def get_mean_degree(self):
        total_degree = sum([x[1] for x in self.graph.degree()])
        return (total_degree / self.graph.number_of_nodes())
    
    def get_variance_degree(self):
        degree_list = [x[1] for x in self.graph.degree()]
        mean = self.get_mean_degree()
        return sum((i - mean) ** 2 for i in degree_list) / len(degree_list)
    
    def get_clustering_coef(self):
        return nx.average_clustering(self.graph)
    
    def get_avg_path_length(self):
        return nx.average_shortest_path_length(self.graph)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        #print(self.game_list)
        #print(list(dict(self.graph.degree()).values()))
        #self.graph = self.update_network()
