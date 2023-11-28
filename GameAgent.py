### GameAgent.py
# Speficies the GameAgent class which is a subclass of a mesa.Agent. This 
# represents the agents of the model and handles characteristics and interaction
###


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

    def __init__(self, id, model, rewiring_p, alpha, beta, UV = (True, None, None)):
        super().__init__(id, model)
        self.neighbors = list(model.graph.neighbors(id))
        self.rationality = model.netRat * model.ratFunct(len(self.neighbors)) # rationality is the rationality of the agent
        self.eta = np.random.rand()*2      # eta is the risk aversion parameter
        self.eta_base = self.eta        # eta_base is the default risk aversion parameter
        self.alpha = alpha              # alpha is the homophilic parameter
        self.beta = beta                # beta controls homophily together with alpha
        self.rewiring_p = rewiring_p
        self.totPayoff = 0              # totPayoff is the (starting) total payoff 

        self.model = model
        

        self.neighChoice = list(model.graph.neighbors(id))
        self.edges = list(model.graph.edges)

        self.full_graph = model.graph
        self.nNeigbors = len(self.neighChoice)
        self.posiVals = [15, 6]

        self.added_edges = [0,0]
        self.removed_edges = [0, 0]

        # Each agent has a game
        if UV[0]:
            uvpay = np.random.RandomState().rand(2)*3 - 1
            #print(uvpay)
            self.game = Game.Game((uvpay[0], uvpay[1]))
        if not UV[0]:
            self.game = Game.Game((UV[1],UV[2]))
        


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

            # Remove an edge
            if len(first_order_neighbours) > 1:        
                P_con = self.get_rewiring_prob(first_order_neighbours, alpha, beta)

                # Make choice from first-order neighbours based on probability
                remove_neighbor = np.random.choice(first_order_neighbours, p=P_con)
                self.model.graph.remove_edge(self.unique_id, remove_neighbor)
                removedEdge = True


            second_order_neighbours = self.get_second_order_neighbours()

            # Add an edge
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
                # Pick a randomn node
                first_node = np.random.choice(self.model.graph.nodes())
                all_nodes = list(self.model.graph.nodes())
                neighbours = list(self.model.graph.neighbors(first_node)) + [first_node]
                # Remove the first node and all its neighbours from the candidates
                possible_nodes = [x for x in all_nodes if x not in neighbours] 
                second_node = np.random.choice(list(possible_nodes))
                self.model.graph.add_edge(first_node, second_node)
        

    def getPlayerStrategyProbs(self, other_agent):
        '''
        Description: This returns the probability for each game and for each player that strategy 0 is chosen
        based on the rationality of both agents
        Input:
            - other_agent: The agent chosen to play a game with
        Output: 
            - g0_P0_Prob_S0: The probability of player 0 choosing strategy 0 in its own game (G0)
            - g0_P1_Prob_S0: The probability of player 1 choosing strategy 0 in the others game (G0)
            - g1_P0_Prob_S0: The probability of player 0 choosing strategy 0 in the others game (G1)
            - g1_P1_Prob_S0: The probability of player 0 choosing strategy 0 in its own game (G1)

        '''
    
        p0_g0_Prob_S0 , p1_g0_Prob_S0  = self.game.getQreChance(0, self.rationality, other_agent.rationality)
        p0_g1_Prob_S0 , p1_g1_Prob_S0  = other_agent.game.getQreChance(0, self.rationality, other_agent.rationality)
        return(p0_g0_Prob_S0 , p1_g0_Prob_S0,  p0_g1_Prob_S0 , p1_g1_Prob_S0)
    

    def getGameChooserProb(chooser, chooser_mean, not_chooser_mean):
        '''This takes the game means and combines it with rationality to get the
           probability that the agent who gets to choose chooses his own game'''
                
        if  chooser.model.alwaysSafe == True:
            return 1 if chooser_mean > not_chooser_mean else 0
        
        rational_utility_chooser = exp(chooser.rationality * chooser_mean)
        rational_utility_notchooser  = exp(chooser.rationality * not_chooser_mean)

        return rational_utility_chooser / (rational_utility_notchooser  + rational_utility_chooser)

    
    def chooseGame(chooser, notChooser, chooser_gChooser_Prob_S0 , notchooser_gchooser_Prob_S0,  chooser_gNotChooser_Prob_S0 , notchooser_gNotChooser_Prob_S0):
        '''
        Description This returns the game that is going to be played.
        '''


        # Calculating the mean utility of the games for the chooser
        chooser_gChooser_UtilityMean = chooser.game.getUtilityMean(0, chooser_gChooser_Prob_S0, notchooser_gchooser_Prob_S0, chooser.eta)
        chooser_gNotchooser_UtilityMean =  notChooser.game.getUtilityMean(1, chooser_gNotChooser_Prob_S0, notchooser_gNotChooser_Prob_S0, chooser.eta)

        # Making a choice between the games with rationality in mind.
        if random() < chooser.getGameChooserProb(chooser_gChooser_UtilityMean, chooser_gNotchooser_UtilityMean):
            return (chooser.game, chooser_gChooser_Prob_S0, notchooser_gchooser_Prob_S0)
        else:
            return (notChooser.game, chooser_gNotChooser_Prob_S0, notchooser_gNotChooser_Prob_S0)



    def step(self):
        '''Advances the agent one time step in the model.'''

        # If the node does not have neighbours, it can be skipped.
        # Should be connected?
        if self.nNeigbors == 0:
            return

        # A neighbor is chosen to play a game with.
        neighId = self.random.choice(self.neighChoice)

        other_agent = self.model.schedule.agents[neighId]


        # Compute strategy for both players
        p0_g0_Prob_S0 , p1_g0_Prob_S0,  p0_g1_Prob_S0 , p1_g1_Prob_S0 = self.getPlayerStrategyProbs(other_agent)


        # Most risk averse player chooses game
        if self.eta > other_agent.eta:
            game, p0_Prob_s0 , p1_Prob_s0 = self.chooseGame(other_agent, p0_g0_Prob_S0 , p1_g0_Prob_S0,  p0_g1_Prob_S0 , p1_g1_Prob_S0)
        else:
            game, p1_Prob_s0 , p0_Prob_s0 = other_agent.chooseGame(self, p1_g1_Prob_S0 , p0_g1_Prob_S0,  p1_g0_Prob_S0 , p0_g0_Prob_S0)

        # Choose strategy game for both players
        P0_strategy = 0 if random() < p0_Prob_s0 else 1
        P1_strategy = 0 if random() < p1_Prob_s0 else 1

        # The game is played.
        (payoff0, payoff1) = game.playGame(P0_strategy, P1_strategy)

        # Both players get their respective payoffs.
        self.totPayoff += payoff0
        other_agent.totPayoff += payoff1

        #player adjust their game depending on earnings 
        #//FIXME: replicator dynamics for game adoption and risk preference with probability proportional to payoff!
        if self.totPayoff < other_agent.totPayoff and self.game.UV <= other_agent.game.UV:
            self.eta = (other_agent.eta+self.eta)/2

        # Change game and eta if other game seems more useful
        ownGameMean = self.game.getUtilityMean(0, p1_Prob_s0, p0_Prob_s0, self.eta)
        if (ownGameMean < payoff0) and (self.totPayoff < other_agent.totPayoff):
            self.game = other_agent.game
            self.eta = other_agent.eta

        #random mutation of risk averion eta
        if rand.random() < 0.01:
            self.eta = rand.random()*2

        
        self.rewire(self.alpha, self.beta, self.rewiring_p)