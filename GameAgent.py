import random as rand
from random import random, randint
import networkx as nx
import numpy as np
from math import exp
import scipy as sp
from mesa import Agent

import Game

P0 = UP = LEFT = RISK = 0
P1 = DOWN = RIGHT = SAFE = 1


class GameAgent(Agent):
    'The agent that will play economic games.'

    def __init__(self, id, model, rewiring_p, alpha, beta):
        super().__init__(id, model)
        self.neighbors = list(model.graph.neighbors(id))
        self.rationality = model.netRat * model.ratFunct(len(self.neighbors)) # rationality is the rationality of the agent
        self.eta = np.random.rand()*2      # eta is the risk aversion parameter
        self.eta_base = self.eta        # eta_base is the default risk aversion parameter
        self.alpha = alpha              # alpha is the homophilic parameter
        self.beta = beta                # beta controls homophily together with alpha
        self.p_rewiring = rewiring_p
        self.totPayoff = 0              # totPayoff is the (starting) total payoff 

        self.model = model
        self.paidoff = [model.game_list[self.unique_id]]

        self.neighChoice = list(model.graph.neighbors(id))
        self.edges = list(model.graph.edges)

        self.full_graph = model.graph
        self.nNeigbors = len(self.neighChoice)
        self.posiVals = [15, 6]

        self.added_edges = [0,0]
        self.removed_edges = [0, 0]


    def get_rewiring_prob(self, neighbors, alpha, beta):
        payoff_diff = []
        for neighbor in neighbors:
            second_neigh_pay = self.model.schedule.agents[neighbor]
            payoff_diff.append(np.abs(self.totPayoff - second_neigh_pay.totPayoff))
        pay_diff = np.array(payoff_diff)
        
        # limit pay_diff to 600 such that exp(600) does not overflow
        limit = 600
        pay_diff[(alpha*(pay_diff-beta)) > limit] = limit

        # Use softmax to get probabilities
        softmax = lambda x :  np.exp(x)/sum(np.exp(x))
        P_con = softmax(alpha*(pay_diff-beta))

        return P_con

    def get_first_order_neighbours(self):
        subgraph0 = nx.ego_graph(self.model.graph, self.unique_id ,radius=0)
        subgraph1 = nx.ego_graph(self.model.graph, self.unique_id ,radius=1)
        subgraph1.remove_nodes_from(subgraph0.nodes())
        return list(subgraph1.nodes())

    def get_second_order_neighbours(self):
        subgraph0 = nx.ego_graph(self.model.graph, self.unique_id ,radius=0)
        subgraph1 = nx.ego_graph(self.model.graph, self.unique_id ,radius=1)
        subgraph2 = nx.ego_graph(self.model.graph, self.unique_id ,radius=2)
        subgraph2.remove_nodes_from(subgraph0.nodes())
        subgraph2.remove_nodes_from(subgraph1.nodes())
        return list(subgraph2.nodes())
    

    def rewire(self, alpha, beta, rewiring_p, random_rewiring = 0.1):

        removedEdge = False
        addedEdge = False
        first_order_neighbours =  self.get_first_order_neighbours()

        if np.random.uniform() < rewiring_p:

            #Remove an edge
            if len(first_order_neighbours) > 1:        
                P_con = self.get_rewiring_prob(first_order_neighbours, alpha, beta)

                # Make choice from first-order neighbours based on probability
                remove_neighbor = np.random.choice(first_order_neighbours, p=P_con)
                self.model.graph.remove_edge(self.unique_id, remove_neighbor)
                removedEdge = True


            second_order_neighbours = self.get_second_order_neighbours()

            #Add an edge
            second_order_neighbours = self.get_second_order_neighbours()
            if len(second_order_neighbours) > 0:
                P_con = self.get_rewiring_prob(second_order_neighbours, alpha, beta)
                # Make choice from second-order neighbours based on probability
                add_neighbor = np.random.choice(second_order_neighbours, p=P_con)
                self.model.graph.add_edge(self.unique_id, add_neighbor)
                addedEdge = True

            if addedEdge and not removedEdge:
                # Remove a random edge in the network
                edges = list(self.model.graph.edges())
                d_edge = self.random.choice(edges)
                self.model.graph.remove_edge(d_edge[0], d_edge[1])


            if not addedEdge and removedEdge:
                # Pick a randomn nide
                first_node = np.random.choice(self.model.graph.nodes())
                all_nodes = list(self.model.graph.nodes())
                neighbours = list(self.model.graph.neighbors(first_node)) + [first_node]
                # Remove the first node and all its neighbours from the candidates
                possible_nodes = [x for x in all_nodes if x not in neighbours] 
                second_node = np.random.choice(list(possible_nodes))
                self.model.graph.add_edge(first_node, second_node)

            
            if removedEdge:
                self.removed_edges[0] += 1
            else:
                self.removed_edges[1] += 1
                

            if addedEdge:
                self.added_edges[0] += 1
            else:
                self.removed_edges[1] += 1
        

    def getPlayerChance0(self, other_agent, game):
        '''This returns the strategy of the player based on his own,
           and the other agent rationality.'''

        chanceOf0 ,chance2 = game.GetQreChance(0, self.rationality, other_agent.rationality)

        return(chanceOf0, chance2)


    def playerRiskGameChance(self, riskGameMean, safeGameMean):
        '''This takes the game means and combines it with rationality to get the
           chance of choosing the risky game.'''
           #\\FIXME: Division by zero error.
        if self.model.alwaysSafe == True:
            return 0

        try:
            risk = exp(self.rationality * riskGameMean)
            safe = exp(self.rationality * safeGameMean)

            return risk / (risk + safe)


        except OverflowError:
            return 1 if riskGameMean > safeGameMean else 0


    def playerStrat(self, other_agent):
        'Returns the preffered game type and the strategy.'
        
        # Calculating the own game values.
        gameRisk = Game.Game(self.paidoff[0], self.paidoff[0])
        riskChance0, chance2 = self.getPlayerChance0(other_agent, gameRisk)
        riskGameMean = gameRisk.getUtilityMean(0, chance2, riskChance0, self.eta)

        # Calculating the safe game values.
        gameSafe = Game.Game(other_agent.paidoff[0], other_agent.paidoff[0])
        safeChance0, chance2= self.getPlayerChance0(other_agent, gameSafe)
        safeGameMean = gameSafe.getUtilityMean(0, chance2, safeChance0, self.eta)

        # Making a choice in between them with rationality in mind.
        if random() < self.playerRiskGameChance(riskGameMean, safeGameMean):
            return (RISK, riskChance0, riskGameMean)
        else:
            return (SAFE, safeChance0, riskGameMean)


    def step(self):
        '''Advances the agent one time step in the model.'''
        # If the node does not have neighbours, it can be skipped.
        if self.nNeigbors == 0:
            return

        # A neighbor is chosen to play a game with.
        neighId = self.random.choice(self.neighChoice)

        other_agent = self.model.schedule.agents[neighId]

        # The player choices are made.
        _, P0chance0, ownGameMean = self.playerStrat(other_agent)
        P1game, P1chance0, otherGameMean = other_agent.playerStrat(self)
        
        # The game played is depending on the risk aversion of the other player.
        if P1game:
            game = Game.Game(self.paidoff[0], self.paidoff[0])
        if not P1game:
            game = Game.Game(other_agent.paidoff[0], other_agent.paidoff[0])

        P0choice = 0 if random() < P0chance0 else 1
        P1choice = 0 if random() < P1chance0 else 1

        # The game is played.
        (payoff0, payoff1) = game.playGame(P0choice, P1choice)

        # Both players get their respective payoffs.
        self.totPayoff += payoff0
        other_agent.totPayoff += payoff1

        #player adjust their game depending on earnings 
        #//FIXME: replicator dynamics for game adoption and risk preference with probability proportional to payoff!
        if self.totPayoff < other_agent.totPayoff and self.paidoff[0] <= other_agent.paidoff[0]:
            self.eta = (other_agent.eta+self.eta)/2

        if (ownGameMean < payoff0) and (self.totPayoff < other_agent.totPayoff):
            self.model.game_list[self.unique_id] = self.model.game_list[other_agent.unique_id]
            self.paidoff = [self.model.game_list[other_agent.unique_id]]
            self.eta = other_agent.eta
            #self.eta = self.eta_base

        #random mutation of risk averion eta
        if rand.random() < 0.01:
            self.eta = rand.random()*2

        
        self.rewire(self.alpha, self.beta, self.p_rewiring)