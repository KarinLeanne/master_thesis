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

    def __init__(self, id, model, rewiring_p, alpha, beta, UV = (True, None, None, False), risk_aversion_distribution =  "uniform", rationality = (2, 0.5)):
        super().__init__(id, model)
        self.id = id
        self.neighbors = list(model.graph.neighbors(id))
        self.rationality = sp.stats.halfnorm.rvs()
        
        
        self.alpha = alpha              # alpha is the homophilic parameter
        self.beta = beta                # beta controls homophily together with alpha
        self.rewiring_p = rewiring_p
        self.wealth = 0              # wealth is the (starting) total payoff 

        self.model = model
        

        self.neighChoice = list(model.graph.neighbors(id))
        self.nNeighbors = len(self.neighChoice)
        #self.edges = list(model.graph.edges)
        #self.full_graph = model.graph
        self.posiVals = [15, 6]

        #self.added_edges = [0,0]
        #self.removed_edges = [0, 0]
        self.games_played = 0

        # Each agent has a risk aversion parameter
        if risk_aversion_distribution == "uniform":
            self.eta = np.random.rand()*2      
        elif risk_aversion_distribution == "log_normal":
            self.eta = np.random.lognormal()*2
        elif risk_aversion_distribution == "gamma":
            self.eta = np.random.gamma()*2
        elif risk_aversion_distribution == "exponential":
            self.eta = np.random.exponential()*2
       
        
        # eta_base is the default risk aversion parameter
        self.eta_base = self.eta        

        # Each agent has a game
        if UV[0] and not UV[3]:
            uvpay = np.random.RandomState().rand(2) * 2
            self.game = Game.Game((uvpay[0], uvpay[1]))
        elif UV[0] and UV[3]:
            while True:
                uvpay = np.random.RandomState().rand(2) * 2
                if uvpay[0] >= 1 or uvpay[1] >= 1:
                    self.game = Game.Game((uvpay[0], uvpay[1]))
                    break
        if not UV[0]:
            self.game = Game.Game((UV[1],UV[2]))
        
    """
    
    def get_rewiring_prob(self, neighbors, alpha, beta, connect = False):

        payoff_diff = [np.abs(self.wealth - self.model.agents[neighbor].wealth) for neighbor in neighbors]
        pay_diff = np.array(payoff_diff)
        
        # limit pay_diff to 600 such that exp(600) does not overflow
        limit = 600
        pay_diff[(alpha*(pay_diff-beta)) > limit] = limit

        # Use softmax to get probabilities
        softmax = lambda x :  np.exp(x)/sum(np.exp(x))
        if connect:
            P_con = softmax(alpha * (pay_diff - beta))
        else:
            P_con = softmax(-alpha * (pay_diff - beta))

        return P_con
        """
    
    
    
    def get_rewiring_prob(self, neighbors, alpha, beta, connect=False):

        
        payoff_diff = [np.abs(self.wealth - self.model.agents[neighbor].wealth) for neighbor in neighbors]
        pay_diff = np.array(payoff_diff)

        # Limit pay_diff to 600 to prevent overflow
        limit = 600
        pay_diff[(alpha * (pay_diff - beta)) > limit] = limit

        # Use the social attachement equation to get probabilities
        epsilon = 1e-12
        if connect:
            P_con = 1 / (1 + np.power((1/beta) * pay_diff, alpha))
        else:
            P_con = 1 / (1 + np.power((1/beta) * np.maximum(pay_diff,epsilon), -alpha))

        # Check if the sum of probabilities is zero
        if np.sum(P_con) == 0:
            # Assign equal probabilities if the sum is zero
            P_con = np.ones_like(P_con) / len(P_con)

        # Normalize probabilities to ensure they sum to 1
        P_con /= np.sum(P_con)
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
            self.model.e_n += 1

            # Remove an edge
            if len(first_order_neighbours) > 1:        
                P_con = self.get_rewiring_prob(first_order_neighbours, alpha, beta, connect=False)
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
            - g1_P1_Prob_S0: The probability of player 1 choosing strategy 0 in its own game (G1)

        '''
    
        p0_g0_Prob_S0 , p1_g0_Prob_S0 = self.game.getQreChance(self.rationality, other_agent.rationality, self.eta, other_agent.eta, self.model.utility_function)
        p1_g1_Prob_S0, p0_g1_Prob_S0  = other_agent.game.getQreChance(other_agent.rationality, self.rationality, other_agent.eta, self.eta, self.model.utility_function)
        return(p0_g0_Prob_S0 , p1_g0_Prob_S0,  p0_g1_Prob_S0 , p1_g1_Prob_S0)
    
    """
    
    def softmax(self, values, temperature=1.0):
        exp_values = np.exp(values / temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    
    
    def getGameChooserProb(chooser, chooser_game_mean, not_chooser_game_mean):
        '''
        probability of choosing own game
        '''
        if chooser.model.alwaysSafe == True:
            return 1 if chooser_game_mean > not_chooser_game_mean else 0

        else:
            values = np.array([chooser_game_mean, not_chooser_game_mean])
            return chooser.softmax(values)[0]
    """        

    
    def chooseGame(chooser, notChooser, chooser_gChooser_Prob_S0 , notchooser_gchooser_Prob_S0,  chooser_gNotChooser_Prob_S0 , notchooser_gNotChooser_Prob_S0):
        '''
        Description This returns the game that is going to be played.
        '''


        # Calculating the mean utility of the games for the chooser
        chooser_gChooser_UtilityMean = chooser.game.getUtilityMean(0, chooser_gChooser_Prob_S0, notchooser_gchooser_Prob_S0, chooser.eta, chooser.model.utility_function)
        chooser_gNotchooser_UtilityMean =  notChooser.game.getUtilityMean(1, chooser_gNotChooser_Prob_S0, notchooser_gNotChooser_Prob_S0, chooser.eta, chooser.model.utility_function)


        # Probability of choosing a game is proportional to the ratio of the means
        p_gChooser = chooser_gChooser_UtilityMean / (chooser_gChooser_UtilityMean + chooser_gNotchooser_UtilityMean)

        # Making a choice between the games
        if random() < p_gChooser:
            return (chooser.game, chooser_gChooser_Prob_S0, notchooser_gchooser_Prob_S0)
        else:
            return (notChooser.game, chooser_gNotChooser_Prob_S0, notchooser_gNotChooser_Prob_S0)



    def step(self):
        '''Advances the agent one time step in the model.'''

        # If the node does not have neighbours, it can be skipped.
        # Should be connected?
        if self.nNeighbors == 0:
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
        self.wealth += payoff0
        other_agent.wealth += payoff1

        # Add that they played one game
        self.games_played += 1
        other_agent.games_played += 1

        #player adjust their game depending on earnings 
        #//FIXME: replicator dynamics for game adoption and risk preference with probability proportional to payoff!

        #What does this mean??
        if self.wealth < other_agent.wealth and self.game.UV == other_agent.game.UV:
            self.eta = (other_agent.eta+self.eta)/2


        # Change game and eta if other game seems more useful
        # Should be both ways?? As then its more useful to play more games otherwise less so?
        ownGameMean = self.game.getUtilityMean(0, p1_Prob_s0, p0_Prob_s0, self.eta, self.model.utility_function)


        mutated = False
        adapted = False

        # Throw utility function on pay off
        if self.model.utility_function == "isoelastic":
            ownPayoff = self.game.isoelastic_utility(self.eta, payoff0)
        else:
            ownPayoff = self.game.linex_utility(self.eta, payoff0)


        if (ownGameMean < ownPayoff) and (self.wealth < other_agent.wealth):
            self.model.e_g += 1
            self.game = other_agent.game
            self.eta = other_agent.eta
            adapted = True


        #random mutation of risk averion eta
        if rand.random() < 1/(self.model.num_agents)**2:
            self.eta = rand.random()*2

         
        #random mutation of game
        # Use 1/N^2    
        if rand.random() < 1/(self.model.num_agents)**2:
            mutated = True
            if self.model.NH:
                while True:
                    uvpay = np.random.RandomState().rand(2) * 2
                    if uvpay[0] > 1 and uvpay[1] > 1:
                        self.game = Game.Game((uvpay[0], uvpay[1]))
                        break
            else:
                uvpay = np.random.RandomState().rand(2) * 2
                self.game = Game.Game((uvpay[0], uvpay[1]))

        if (mutated or adapted):
            self.model.e_g += 1

        
        self.rewire(self.alpha, self.beta, self.rewiring_p)
        
        # Update information about neighbours 
        self.neighChoice = list(self.model.graph.neighbors(self.id))
        self.nNeighbors = len(self.neighChoice)
        