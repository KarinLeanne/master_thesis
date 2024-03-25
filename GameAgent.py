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
import copy


import Game

P0 = UP = LEFT = RISK = 0
P1 = DOWN = RIGHT = SAFE = 1


class GameAgent(Agent):

    'The agent that will play economic games.'

    def __init__(self, id, model, rewiring_p = 0, alpha = 0, beta = 0, rat = 0, uvpay = (0,0), UV = (True, None, None, False), risk_aversion_distribution =  "uniform"):
        super().__init__(id, model)
        self.id = id
        self.rationality = rat
        
        
        self.alpha = alpha              # alpha is the homophilic parameter
        self.beta = beta                # beta controls homophily together with alpha
        self.rewiring_p = rewiring_p
        self.wealth = 0              # wealth is the (starting) total payoff 
        self.payoff_list = [0] * 5
        self.recent_wealth = 0

        self.model = model

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
            self.eta = np.random.gamma(shape=1)*2
        elif risk_aversion_distribution == "exponential":
            self.eta = np.random.exponential()*2
       
        
        # eta_base is the default risk aversion parameter
        self.eta_base = self.eta        

        # Each agent has a game
        if UV[0]:
            self.game = Game.Game((uvpay[0], uvpay[1]))
            self.model.games.append(self.game)
        if not UV[0]:
            self.game = Game.Game((UV[1],UV[2]))
            self.model.games.append(self.game)
        
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
    
    def weighted_sum(self, pay_off_list):
        if len(pay_off_list) != 5:
            raise ValueError("Input list must contain exactly 5 numbers")

        weights = [0.2 * (i + 1) for i in range(5)]
        total_weighted_sum = sum(num * weight for num, weight in zip(pay_off_list, weights))
        
        return total_weighted_sum
    
    def fifo_shift_payoff(self, payoff_list, payoff):
        if len(payoff_list) == 0:
            return payoff_list  # Return empty list if payoff_list is empty
        
        # Shift elements in the payoff_list towards the beginning
        for i in range(len(payoff_list) - 1):
            payoff_list[i] = payoff_list[i + 1]
        
        # Add the new payoff at the end
        payoff_list[-1] = payoff
        
        return payoff_list
        

    def get_rewiring_prob(self, neighbors, alpha, beta, connect=False):

        
        payoff_diff = [np.abs(self.wealth - self.model.agents[neighbor].wealth) for neighbor in neighbors]
        pay_diff = np.array(payoff_diff)

        # Limit pay_diff to 600 to prevent overflow
        limit = 600
        pay_diff[(alpha * (pay_diff - beta)) > limit] = limit

        # Use the social attachement equation to get probabilities
        epsilon = 1e-12
        P_con = 1 / (1 + np.power((1/beta) * np.maximum(pay_diff,epsilon), -alpha))

        # Check if the sum of probabilities is zero
        if np.sum(P_con) == 0:
            # Assign equal probabilities if the sum is zero
            P_con = np.ones_like(P_con) / len(P_con)

        # Normalize probabilities to ensure they sum to 1
        P_con /= np.sum(P_con)
        return P_con
    

    def get_non_neighbors(self):
        node = self.id
        # Get all nodes in the graph
        all_nodes = list(self.model.graph.nodes())
        # Remove node A from the list
        all_nodes.remove(node)
        # Get first-order neighbors of node A
        first_order_neighbors = list(self.model.graph.neighbors(node))
        # Remove first-order neighbors from the list
        non_neighbors = [node for node in all_nodes if node not in first_order_neighbors]
        return non_neighbors

    def get_second_order_neighbors(self):
        node = self.id
        # Get the first-order neighbors of node B
        first_order_neighbors = set(self.model.graph.neighbors(node))
        # Initialize a set to store second-order neighbors
        second_order_neighbors = set()
        # Iterate over the first-order neighbors
        for neighbor in first_order_neighbors:
            # Get the neighbors of the current neighbor excluding node B and its first-order neighbors
            second_order_neighbors.update(set(self.model.graph.neighbors(neighbor)) - first_order_neighbors - {node})
        return list(second_order_neighbors)
    
    def get_valid_neighbors(self):
        valid_neighbors = []
        node = self.id
        # Get all neighbors of the node
        neighbors = list(self.model.graph.neighbors(node))
        # Iterate through each neighbor
        for neighbor in neighbors:
            # Make a copy of the graph
            graph_copy = copy.deepcopy(self.model.graph)
            graph_copy.remove_edge(node, neighbor)
            # Check if removing the connection with 'node' will not disconnect the network
            if nx.is_connected(graph_copy):
                valid_neighbors.append(neighbor)
        return valid_neighbors
    

    def rewire(self, alpha, beta, rewiring_p):
        
        # Randomly determine if rewiring probability threshold is met
        if np.random.uniform() < rewiring_p:

            # Only rewire edge if it can be done without disconnecting the network
            candidates_removal = self.get_valid_neighbors()
            if len(candidates_removal) > 1:
                self.model.e_n += 1
                # Calculate probablties of removal
                P_con = self.get_rewiring_prob(candidates_removal, alpha, beta, connect=False)
                # Make choice from first-order neighbours based on probability
                removed_neighbor = np.random.choice(candidates_removal, p=P_con)
                self.model.graph.remove_edge(self.id, removed_neighbor)

                # Add an edge if the agent has second order neighbours
                candidates_connection = self.get_second_order_neighbors()
                if len(candidates_connection) > 0:
                    P_con = self.get_rewiring_prob(candidates_connection, alpha, beta)
                    # Make choice from second-order neighbours based on probability
                    new_neighbor = np.random.choice(candidates_connection, p=P_con)
                    self.model.graph.add_edge(self.id, new_neighbor)

                # Else make a connection with a random node
                else:
                    candidates_connection = self.get_non_neighbors()
                    new_neighbor = np.random.choice(candidates_connection)
                    self.model.graph.add_edge(self.id, new_neighbor)



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
        if self.model.graph.degree(self.id) == 0:       
            return

        # A neighbor is chosen to play a game with.
        neighId = self.random.choice(list(self.model.graph.neighbors(self.id)))

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
        self.payoff_list = self.fifo_shift_payoff(self.payoff_list, payoff0)
        self.recent_wealth = self.weighted_sum(self.payoff_list)

        other_agent.wealth += payoff1
        other_agent.pay_off_list = other_agent.fifo_shift_payoff(other_agent.payoff_list, payoff1)
        other_agent.recent_wealth = other_agent.weighted_sum(other_agent.payoff_list)

        # Add that they played one game
        self.games_played += 1
        other_agent.games_played += 1

        #What does this mean??
        if self.wealth < other_agent.wealth and self.game.UV == other_agent.game.UV:
            self.eta = (other_agent.eta+self.eta)/2

        # Change game and eta if other game seems more useful
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
                        self.model.games.append(self.game)
                        break
            else:
                uvpay = np.random.RandomState().rand(2) * 2
                self.game = Game.Game((uvpay[0], uvpay[1]))
                self.model.games.append(self.game)

        if (mutated or adapted):
            self.model.e_g += 1

        self.rewire(self.alpha, self.beta, self.rewiring_p)

        