### Game.py
# Specifies the Game class. This represent the games agents can play with one another.
# And computes characteristics of the game
###

import numpy as np
from scipy.optimize import least_squares
from math import exp


class Game:
    def __init__(self, UV = (3, 5)):
        
        '''If no payoff matrix is given, the prisoners dilemma is chosen.
        Can also create a game from the payoffs of two players.'''
        self.UV = UV
        
    def getPayoffMatrix(self):
        '''
        Added rescaling
        does move utilities:(
        '''
        payoff_matrix = [[(1, 1), (self.UV[0], self.UV[1])], [(self.UV[1], self.UV[0]), (0,0)]]

        # Normalize the payoff matrix
        norm_factor = 1 + self.UV[0] + self.UV[1]
        return [[element / norm_factor for element in row] for row in payoff_matrix]
    
    def playGame(self, choiceP0, choiceP1):
        'Simulates a game with the player choices.'
        payoffs = self.getPayoffMatrix()
        return payoffs[choiceP0][choiceP1]


    def getPlayerCells(self, player):
        '''Returns the cells in a choice by choice order.
        c21 means player A second strat, player B first strat'''

        c11 = self.getPayoffMatrix()[0][0][player]
        c21 = self.getPayoffMatrix()[0][1][player]
        c12 = self.getPayoffMatrix()[1][0][player]
        c22 = self.getPayoffMatrix()[1][1][player]

        return (c11, c21, c12, c22)

    
    def equations(self, vars, pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22, lambA, lambB):
        '''
        QRE eq
        '''
        pc1, pc2 = vars

        eq1 = np.exp(lambA * (pc2 * pA_u11 + (1 - pc2) * pA_u12)) / (
            np.exp(lambA * (pc2 * pA_u11 + (1 - pc2) * pA_u12)) + np.exp(lambA * (pc2 * pA_u21 + (1 - pc2) * pA_u22))
        ) - pc1

        eq2 = np.exp(lambB * (pc1 * pB_u11 + (1 - pc1) * pB_u21)) / (
            np.exp(lambB * (pc1 * pB_u11 + (1 - pc1) * pB_u21)) + np.exp(lambB * (pc1 * pB_u12 + (1 - pc1) * pB_u22))
        ) - pc2

        return [eq1, eq2]
    
    def proportional_scaling_with_range(self, numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        range_val = max_val - min_val
        scaled_values = [(x - min_val) / range_val for x in numbers]
        return scaled_values


    def getQreChance(self, rationalityA, rationalityB, etaA, etaB, utility_function):
        '''Returns the chance of the given player choosing option 0 in the game, 
        using quantal response equilibrium. Numerical solution of the equations and 
        in case of error, a random number is returned.'''

        'The mean value for the game utility is returned using strategy chance0.'
        if utility_function == 'isoelastic':
            utility_function = self.isoelastic_utility
        else:
            utility_function = self.linex_utility

        # Get the utilities for player A
        pA_c11, pA_c21, pA_c12, pA_c22 = self.getPlayerCells(0)
        pA_u11 = utility_function(etaA, pA_c11)
        pA_u21 = utility_function(etaA, pA_c21)
        pA_u12 = utility_function(etaA, pA_c12)
        pA_u22 = utility_function(etaA, pA_c22)

        # Get the utilities for player B
        pB_c11, pB_c21, pB_c12, pB_c22 = self.getPlayerCells(1)
        pB_u11 = utility_function(etaB, pB_c11)
        pB_u21 = utility_function(etaB, pB_c21)
        pB_u12 = utility_function(etaB, pB_c12)
        pB_u22 = utility_function(etaB, pB_c22)

        pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22 =  self.proportional_scaling_with_range([pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22])

        lambA, lambB = rationalityA, rationalityB

        try:
            x, y = least_squares(
            self.equations, (0.5, 0.5), args=(pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22, lambA, lambB)).x
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            if pA_c21 < 0 and pA_c12 < 0:
                return 0, 0
            else:
                return np.random.uniform(0, 1), np.random.uniform(0, 1)
        return x, y


    def isoelastic_utility(self, eta, c):
        'The implemented utility function is the isoelastic utility function.'
        if c == 0:
            return 0
        elif eta == 1:
            return np.log(c)
        else:
            return c**(1 - eta) / (1 - eta)

        
    def linex_utility(self, eta, c):
        'The implemented utility function is the linex utility function.'
        return (1 / eta) * (np.exp(-eta * c) - eta * c - 1)
            

    def getUtilityMean(self, player, chance2, chance0, eta, utility_function):
        'The mean value for the game utility is returned using strategy chance0.'
        if utility_function == 'isoelastic':
            utility_function = self.isoelastic_utility
        else:
            utility_function = self.linex_utility

        (firstCell, secondCell, thirdCell, fourthCell) = self.getPlayerCells(player)
        Pstrat0 = chance0 * (utility_function(eta, firstCell)*chance2 + (1-chance2)*utility_function(eta, secondCell))
        Pstrat1 = (1 - chance0) * (utility_function(eta, thirdCell)*chance2 + (1-chance2)*utility_function(eta, fourthCell))

        return Pstrat0 + Pstrat1
    




