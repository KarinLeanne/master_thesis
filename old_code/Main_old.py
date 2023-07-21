import Simulate_old as sim
import Vizualize as viz

N_STEPS = 10
N_AGENTS = 20
N_ROUNDS = 1

def main():


    Payoffs = sim.simulateANDPayoff(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('HK',4,1))
    Degrees, UV_space = sim.simulateANDdegree(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(True,1,2))
    Wealth, RiskAv = sim.simulateANDWealth(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2))

    viz.vizualize_UV(UV_space)

    '''Show last time step of differen network types and output variables'''
    #collect Payoffs, Degrees and Wealth for HK, RR and WS networks
    Payoffs = []
    Degrees = []
    Wealth = []
    for i in range(3):
        if i == 0:
            Payoffs.append(sim.simulateANDPayoff(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('HK',4,1))[-1])
            Degrees.append(sim.simulateANDdegree(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('HK',4,1))[-1])
            Wealth.append(sim.simulateANDWealth(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('HK',4,1))[-1])
        if i == 1:
            Payoffs.append(sim.simulateANDPayoff(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('RR',4,1))[-1])
            Degrees.append(sim.simulateANDdegree(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('RR',4,1))[-1])
            Wealth.append(sim.simulateANDWealth(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('RR',4,1))[-1])
        if i == 2:
            Payoffs.append(sim.simulateANDPayoff(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('WS',4,1))[-1])
            Degrees.append(sim.simulateANDdegree(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('WS',4,1))[-1])
            Wealth.append(sim.simulateANDWealth(N=N_AGENTS, rounds = N_ROUNDS, steps = N_STEPS, netRat = 0.1, partScaleFree = 1, alwaysSafe = False, UV=(False,1,2), network=('WS',4,1))[-1])




if __name__ == "__main__":
    main()
