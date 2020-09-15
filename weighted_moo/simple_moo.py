from pulp import *
from graphs import data_graph_functions
import networkx as nx
import matplotlib.pyplot as plt

G = nx.ladder_graph(10)
# nx.draw_spring(G, node_size=200, width=1.5, with_labels=True)
# plt.show()

def solve_lp(G1, T_init, num_fires, num_ffs, savable_proportion, start_nodes):
    #Set constants
    V = G1.number_of_nodes()

    # Determine number of firefighters, initial fires, and start fire(s)
    p = num_ffs
    f = num_fires

    # Set savable proportion (0 if only saving 1 node) and initial T
    s = savable_proportion
    T = T_init

    # Refresh graph to original state
    G = G1.copy()

    # Problem Formulation
    nodes = list(G.nodes)  # list of nodes
    time = list(range(0, T + 1))  # Time from 0 to T
    time1 = list(range(1, T + 1))  # Time from 1 to T

    neighbor_dict = {}
    for x in nodes:
        neighbor_dict[x] = [n for n in G.neighbors(x)] # get neighbors by calling dictionary

    start_nbrs_all = []
    for x in start_nodes:
        start_nbrs_all.append(neighbor_dict[x]) # get neighbors of all start nodes
    start_nbrs_all = list(set([item for sublist in start_nbrs_all for item in sublist])) # ensure no duplicates in list

    # Instantiate problem
    prob = LpProblem("MAXSAVE_tFIRE_Problem", LpMinimize)

    # Independent Variables
    b = pulp.LpVariable.dicts("B",
                              ((x, t) for x in nodes for t in time),
                              lowBound=0, upBound=1, cat='Binary')  # Inclusive bound, burning vars by node and time
    d = pulp.LpVariable.dicts("D",
                              ((x, t) for x in nodes for t in time),
                              lowBound=0, upBound=1, cat='Binary') # Inclusive bound, defended vars by node and time

    # Objective
    prob += pulp.lpSum(b[(x, T)] - b[(x, T - 1)] for x in nodes), 'Obj for time {}'.format(T)

    # Constraints (spontaneous combustion if infeasible)
    for t in time1:
        for x in nodes:
            nbrs = neighbor_dict[x]
            for y in nbrs:
                prob += - b[(x, t)] - d[(x, t)] + b[(y, t - 1)] <= 0  #

            prob += -pulp.lpSum(b[(y, t - 1)] for y in nbrs) + b[(x, t)] - b[(x, t - 1)] <= 0
            prob += b[(x, t)] + d[(x, t)] <= 1
            prob += b[(x, t - 1)] - b[(x, t)] <= 0
            prob += d[(x, t - 1)] - d[(x, t)] <= 0

        if t < T:
            prob += pulp.lpSum(d[(x, t)] - d[(x, t - 1)] for x in nodes) == p  # save exactly p ==
        else:
            prob += pulp.lpSum(d[(x, T)] - d[(x, T - 1)] for x in nodes) <= p  # except in final time step

    if s > 0:  # Savable proportion case
        prob += -(1 - (1 / V) * pulp.lpSum((b[(x, T)] + d[(x, T)]) for x in nodes)) <= -s  # save at least s proportion
    else:  # Savable amount (1) case
        prob += -(V - pulp.lpSum(b[(x, T)] + d[(x, T)] for x in nodes)) <= -1  # save at least 1

    for x in nodes:
        if x in start_nodes:
            prob += (b[(x, 0)]) == 1
        else:
            prob += (b[(x, 0)]) == 0
        prob += (d[(x, 0)]) == 0

    for x in nodes:
        if x not in start_nbrs_all and x not in start_nodes:
            prob += (b[(x, 1)]) == 0

    prob.writeLP("MAXSAVE_tFIRE_Problem.lp")
    # Solving
    prob.solve()

    status = LpStatus[prob.status]
    obj = value(prob.objective)
    vars = prob.variables()

    return prob, status, obj, vars

print(solve_lp(G, 2, 1, 1, 0, [1,2]))
#
#
# prob = LpProblem("MAXSAVE_tFIRE_Problem",LpMinimize)
#
# LpVariable("example", 0, 1, LpInteger)