import numpy as np
import itertools

import GamesModel as gm

def gini_new(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    array = array.flatten() #all values are treated equally, arrays must be 1d
    #make array into floats
    array = array.astype(float)
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    #array = np.sum(array, 0.0000001, dtype=np.float)
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements

    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 



def simulateANDWealth(N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None), network=('RR',4,1)):
    '''This function runs the model for given parameters and returns wealth and risk-aversion development'''
    avWealth = []
    avRisk = []
    for j in range(rounds):
        mod = gm.GamesModel(N, netRat, partScaleFree, alwaysSafe,UV)
        # Step through the simulation.
        for n in range(steps):
            mod.step()
        Payoff = mod.datacollector.get_agent_vars_dataframe() 
        temp, temp2, rtemp, rtemp2  = [], [], [], []
        for i in range(steps):
            temp = list(itertools.chain(Payoff["playerPayoff"].loc[i, :].values.tolist()))
            rtemp = list(itertools.chain(Payoff["player risk aversion"].loc[i, : ].values.tolist()))
            temp2.append(temp)
            rtemp2.append(rtemp)
            if len(avWealth):
                avWealth[i].extend(temp)
                avRisk[i].extend(rtemp)
        if not len(avWealth):
            avWealth = temp2
            avRisk = rtemp2
    avWealth.pop(0)

    return(avWealth, avRisk)


def simulateANDPayoff(N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None), network=('RR',4,1)):
    '''This function runs the model for given parameters and returns payoff development'''
    avPayoff = []
    #avRat = []
    for j in range(rounds):
        #print('round', j)
        mod = gm.GamesModel(N, netRat, partScaleFree, alwaysSafe,UV)
        # Step through the simulation.
        for n in range(steps):
            mod.step()
        Payoff = mod.datacollector.get_agent_vars_dataframe()
        temp, temp2 = [], []
        memory = [0] * N

        for i in range(steps):
            temp = list(itertools.chain(Payoff["playerPayoff"].loc[i, : ].values.tolist())) 
            temp2.append(list(np.subtract(temp, memory)))
            memory = temp
            if len(avPayoff):
                avPayoff[i].extend(temp2[i])
               
        if not len(avPayoff):
            avPayoff = temp2
        temp2=[]
        memory=[0] * N
    avPayoff.pop(0)
  
    return(avPayoff)

def simulateANDdegree(N, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None), network=('RR',4,1)):
    '''This function runs the model for given parameters and returns degree distribution and UV space population'''
    avDegree = []
    UV_space = [[],[]]
    for j in range(rounds):
        mod = gm.GamesModel(N, netRat, partScaleFree, alwaysSafe, UV)
        for n in range(steps):
            mod.step()

        Data = mod.datacollector.get_model_vars_dataframe()
        degdis = Data.iloc[:, 0:1]
        games = Data.iloc[:, 1:2]
        temp = []
        gtemp = []
        
        for i in range(steps):
            temp.append(degdis.loc[i, :].values.tolist()[0])
            gtemp.append(games.loc[i, :].values.tolist()[0])
            if len(avDegree):
                avDegree[i].extend(temp[i])

        if not len(avDegree):
            avDegree = temp

    U_space, V_space = [], []
    temp3, temp4 = [], []
    UV_coors_temp = []

    for k in range(len(gtemp)):
        for l in range(len(gtemp[k])):
            temp3.append( gtemp[k][l][0][1])
            temp4.append( gtemp[k][l][1][0])
            uvcoor= [gtemp[k][l][0][1], gtemp[k][l][1][0]]
            UV_coors_temp.append(uvcoor)
        
        U_space.append(temp3)
        V_space.append(temp4)
        temp3, temp4 = [], []

    
        if len(UV_space[0]):
            UV_space[0][k].extend(U_space[k])
            UV_space[1][k].extend(V_space[k])
        
    if not len(UV_space[0]):
        UV_space[0] = U_space
        UV_space[1] = V_space



    return (avDegree, UV_space)
