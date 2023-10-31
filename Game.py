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
        
    def getPayoffmatrix(self):
        return [[(1, 1), (self.UV[0], self.UV[1])], [(self.UV[1], self.UV[0]), (0,0)]]
    
    def playGame(self, choiceP0, choiceP1):
        'Simulates a game with the player choices.'
        payoffs = self.getPayoffmatrix()
        return payoffs[choiceP0][choiceP1]


    def getPlayerCells(self, player):
        'Returns the cells in a choice by choice order.'

        firstCell = self.getPayoffmatrix()[0][0][player]
        secondCell = self.getPayoffmatrix()[0][1][player]
        thirdCell = self.getPayoffmatrix()[1][0][player]
        fourthCell = self.getPayoffmatrix()[1][1][player]

        return (firstCell, secondCell, thirdCell, fourthCell)


    def GetQreChance(self, player, rationality1, rationality2):
        '''Returns the chance of the given player choosing option 0 in the game, 
        using quantal response equilibrium. Numerical solution of the equations and 
        in case of error, a random number is returned.'''

        (firstCell, secondCell, thirdCell, fourthCell) = self.getPlayerCells(player)

        print(firstCell, secondCell, thirdCell, fourthCell)
        print(rationality1, rationality2)

        u11 = firstCell
        u12 = secondCell
        u21 = thirdCell
        u22 = fourthCell

        lamb1 = rationality1
        lamb2 = rationality2

        def equations(vars):

            pc1, pc2 = vars
            eq1 = (exp(lamb1*(pc2*u11+(1-pc2)*u12)))/(exp(lamb1*(pc2*u11+(1-pc2)*u12))+exp(lamb1*(pc2*u21+(1-pc2)*u22) )) - pc1 
            #print("eq1", eq1)
            eq2 = (exp(lamb2*(pc1*u11+(1-pc1)*u21)))/(exp(lamb2*(pc1*u11+(1-pc1)*u21))+exp(lamb2*(pc1*u12+(1-pc1)*u22)))  - pc2 
            #print("eq2", eq2)
            return [eq1, eq2]
        
    
        try:
            #x, y =  fsolve(equations, (0.5, 0.5))
            x, y = least_squares(equations, (0,5, 0,5), bounds = ((0, 1), (0, 1)))
        except:
            #if u12 and u21 are negative return 0 for y and x
            if u12 < 0 and u21 < 0:
                return (0,0)
            else:
                return (np.random.uniform(0,1), np.random.uniform(0,1))
        return (x,y)


    def getUtilityMean(self, player, chance2, chance0,  eta):
        'The mean value for the game utility is returned using strategy chance0.'

        (firstCell, secondCell, thirdCell, fourthCell) = self.getPlayerCells(player)
        Pstrat0 = chance0 * (self.utilityFunct(eta, firstCell)*chance2 + (1-chance2)*self.utilityFunct(eta, secondCell))
        Pstrat1 = (1 - chance0) * (self.utilityFunct(eta, thirdCell)*chance2 + (1-chance2)*self.utilityFunct(eta, fourthCell))

        return Pstrat0 + Pstrat1


    def utilityFunct(self, eta, c):
        'The implemented utility function is the isoelastic utility function.'

        if eta != 0:
            return (1- np.exp(-eta*c))/(eta)
        else:
            return c
        

