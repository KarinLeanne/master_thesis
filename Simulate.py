import numpy as np
import pandas as pd
import itertools
from IPython.display import display
import GamesModel as gm
from tabulate import tabulate

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

def simulate(N, p_rewiring, alpha, beta, network, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None)):

    #agent_data = pd.DataFrame()
    network_data = pd.DataFrame(columns=['Round'])
    agent_data = pd.DataFrame(columns=['Round'])

    for round in range(rounds):
        #print("round", round)
        
        # Keep track of how edges are made

        Track_edges = [[0,0], [0,0]]

        model = gm.GamesModel(N, p_rewiring, alpha, beta, network, netRat, partScaleFree, alwaysSafe, UV)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        network_data = pd.concat([network_data, model.datacollector.get_model_vars_dataframe()])
        network_data['Round'] = network_data['Round'].fillna(round)

    network_data.index.name = 'Step'
    return network_data, agent_data

def simulateANDdegree(N,  p_rewiring, alpha, beta, network, rounds, steps, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,None,None)):
    '''This function runs the model for given parameters and returns degree distribution and UV space population'''
    network_data = pd.DataFrame(columns=['Round'])

    avDegree = []
    UV_space = [[],[]]
    for round in range(rounds):
        model = gm.GamesModel(N, p_rewiring, alpha, beta, network, netRat, partScaleFree, alwaysSafe, UV)
        for _ in range(steps):
            model.step()

        network_data = pd.concat([network_data, model.datacollector.get_model_vars_dataframe()])
        network_data['Round'] = network_data['Round'].fillna(round)    
        print(tabulate(network_data.head(2), headers = 'keys', tablefmt = 'psql'))



        """

        Data = model.datacollector.get_model_vars_dataframe()
        #degdis = Data.iloc[:, 0:1]
        games = Data.iloc[:, 1:2]
        #temp = []
        gtemp = []

        print(games)
        
        for i in range(steps):
            #temp.append(degdis.loc[i, :].values.tolist()[0])
            gtemp.append(games.loc[i, :].values.tolist()[0])
            #if len(avDegree):
            #    avDegree[i].extend(temp[i])

        #if not len(avDegree):
        #    avDegree = temp

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



    return (avDegree, UV_space)"""


