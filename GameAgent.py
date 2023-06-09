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
        self.eta = rand.random()*2      # eta is the risk aversion parameter
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


    def rewire(self, alpha, beta, rewiring_p):

        # Add all neighbours to a list
        neighbors = list(self.model.graph.neighbors(self.unique_id))

        # Only rewire if there are neighbours and Try to rewire with certain probability
        if neighbors and np.random.uniform() < rewiring_p:

            # Only possible to rewire to second-order neighbours that are not also first-order neighbours
            subgraph0 = nx.ego_graph(self.model.graph, self.unique_id ,radius=0)
            subgraph1 = nx.ego_graph(self.model.graph, self.unique_id ,radius=1)
            subgraph2 = nx.ego_graph(self.model.graph, self.unique_id ,radius=2)

            subgraph1.remove_nodes_from(subgraph0.nodes())
            first_order_neighbors = list(subgraph1.nodes())

            subgraph2.remove_nodes_from(subgraph0.nodes())
            subgraph2.remove_nodes_from(subgraph1.nodes())
            second_order_neighbors = list(subgraph2.nodes())


            # rewiring prob of each second order neighbour is proportional to total payoff difference (homophily)
            if second_order_neighbors:
                P_con = self.get_rewiring_prob(second_order_neighbors, alpha, beta)

                # Make choice from second-order neighbours based on probability
                add_neighbor = np.random.choice(second_order_neighbors, p=P_con)
                self.model.graph.add_edge(self.unique_id, add_neighbor)

        # Try to remove a connection with certain prob + Only remove a neighbour if there is more then one neighbour
        if np.random.uniform() < rewiring_p and len(first_order_neighbors) > 1:        


            P_con = self.get_rewiring_prob(first_order_neighbors, alpha, beta)

            # Make choice from first-order neighbours based on probability
            remove_neighbor = np.random.choice(first_order_neighbors, p=P_con)
            self.model.graph.remove_edge(self.unique_id, remove_neighbor)


              

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