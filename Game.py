import numpy as np
from scipy.optimize import least_squares
from math import exp

P0 = UP = LEFT = RISK = 0
P1 = DOWN = RIGHT = SAFE = 1

class Game:
    def __init__(self, payoff0 = None, payoff1 = None,
                                       payoffs = [[(1, 3),(0, 5)],
                                                  [(5, 0),(1, 1)]]):
        '''If no payoff matrix is given, the prisoners dilemma is chosen.
        Can also create a game from the payoffs of two players.'''

        if payoff0 != None:
            payoffs = [list(zip(payoff0[LEFT], payoff1[LEFT])) ,
                       list(zip(payoff0[RIGHT], payoff1[RIGHT]))]

        self.payoffs = payoffs


    def playGame(self, choiceP0, choiceP1):
        'Simulates a game with the player choices.'

        return self.payoffs[choiceP0][choiceP1]


    def getPlayerCells(self, player):
        'Returns the cells in a choice by choice order.'
        a = b = 0

        # Depending on the player, the cell order changes.
        if player == P0:
            b = 1
        else:
            a = 1

        firstCell = self.payoffs[UP][LEFT][player]
        secondCell = self.payoffs[UP + a][LEFT + b][player]
        thirdCell = self.payoffs[DOWN - a][RIGHT - b][player]
        fourthCell = self.payoffs[DOWN][RIGHT][player]

        return (firstCell, secondCell, thirdCell, fourthCell)



    def GetQreChance(self, player, rationality1, rationality2):
        '''Returns the chance of the given player choosing option 0 in the game, 
        using quantal response equilibrium. Numerical solution of the equations and 
        in case of error, a random number is returned.'''

        (firstCell, secondCell, thirdCell, fourthCell) = self.getPlayerCells(player)

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
        

    def Game_to_UV(payoff):
        'Returns normalized payoffs for the game.'
        u11= 1
        u12= (payoff[1][2]-payoff[2][2])/(payoff[1][1]-payoff[2][2])
        u21= (payoff[2][1]-payoff[1][1])/(payoff[1][1]-payoff[2][2])
        u22= 0

        norm_payoffs = [[u11, u12], [u21, u22]]

        return(norm_payoffs)