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
            model_reporters={"Degree Distribution": self.compute_degdis, "Game Distribution" : "game_list"},
            agent_reporters={"playerPayoff": "totPayoff","player risk aversion": "eta"}
        )

    def compute_degdis(model):
        deg_array = list(dict(model.graph.degree()).values())

        return (deg_array)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        #print(self.game_list)
        #print(list(dict(self.graph.degree()).values()))
        #self.graph = self.update_network()
