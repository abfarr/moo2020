from pulp import *
from graphs import data_graph_functions as dgf
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import pandas as pd

def solve_lp(G1, T_init, num_ffs, start_nodes, alpha):

    # Determine number of firefighters
    p = num_ffs

    # Initial T
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
    prob += lpSum(b[(x, T)]*alpha + (b[(x, T)]-b[(x, T-1)])*(1-alpha)), 'Obj for T={}, alpha={}'.format(T, alpha) # MAXSAVE * alpha + TFIRE * (1-alpha)
    # prob += pulp.lpSum(b[(x, T)] - b[(x, T - 1)] for x in nodes), 'Obj for time {}'.format(T)

    # Constraints (spontaneous combustion if infeasible)
    for t in time1:
        for x in nodes:
            nbrs = neighbor_dict[x]
            for y in nbrs:
                prob += - b[(x, t)] - d[(x, t)] + b[(y, t - 1)] <= 0  # neighbors of burning node must burn or be defended in next time step

            prob += -lpSum(b[(y, t - 1)] for y in nbrs) + b[(x, t)] - b[(x, t - 1)] <= 0 # prevent spontaneous combustion
            prob += b[(x, t)] + d[(x, t)] <= 1 # can't be burnt and defended
            prob += b[(x, t - 1)] - b[(x, t)] <= 0 # burnt stays burnt
            prob += d[(x, t - 1)] - d[(x, t)] <= 0 # defended stays defended

        prob += pulp.lpSum(d[(x, T)] - d[(x, T - 1)] for x in nodes) <= p # ADD: use no more than the allotted number of firefighters

    for x in nodes:
        if x in start_nodes:
            prob += (b[(x, 0)]) == 1 # init burn
        else:
            prob += (b[(x, 0)]) == 0 # init unburnt
        prob += (d[(x, 0)]) == 0 # init defended

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


def find_optimal_T(G1, f, p, T, T_upper, alpha, plot):
    G = G1.copy()
    size = nx.number_of_nodes(G)

    start_nodes = []
    for n in G.nodes:
        if G.nodes[n]['start_nodes'] == 1:
            start_nodes.append(n)

    optimal_T = False
    while optimal_T != True and T <= T_upper:

        prob, status, obj, vars = solve_lp(G, T_init=T, num_ffs=p, start_nodes=start_nodes, alpha=alpha)
        #print('Status: {} \tObjective: {}\tStart Nodes: {}'.format(status,obj, start_nodes))


        # Get all burnt throughout entire process. Use this to verify containment since obj == 0 does not apply with MAXSAVE
        burnt = []
        for v in vars:
            if v.name.startswith('B') and v.varValue > 0:
                node_b = int(v.name[v.name.find("(") + 1:v.name.find(",")])
                time_b = int(v.name[v.name.find(",") + 2:v.name.find(")")])
                burnt.append((node_b, time_b))

        last_burnt = []
        second_last_burnt = []
        max_time = max(burnt, key=lambda x: x[1])[1]
        for tup in burnt:
            if tup[1] == max_time:
                last_burnt.append(tup)
            elif tup[1] == max_time - 1:
                second_last_burnt.append(tup)
        if len(list(set(last_burnt))) - len(list(set(second_last_burnt))) == 0:
            contained=True
        else:
            contained=False

        if status == 'Optimal' and contained == True:
            optimal_T = True
        elif status == 'Optimal' and contained == False:
            optimal_T = False
            T = T + 1
        elif status == 'Infeasible':
            optimal_T = False

        # if optimal_T == True:
        #     print('Time: {}\t\tOptimal: {}'.format(T, optimal_T))
        # else:
        #     print('Time: {}\t\tOptimal: {}'.format(T-1, optimal_T))

    if optimal_T == True:
        print('Optimal T Found: T = {}'.format(T-1))

        # Get all defended nodes throughout entire process.
        defended = []
        for v in vars:
            if v.name.startswith('D') and v.varValue > 0:
                node_d = int(v.name[v.name.find("(") + 1:v.name.find(",")])
                time_d = int(v.name[v.name.find(",") + 2:v.name.find(")")])
                defended.append((node_d, time_d))

        # Find earliest time step at which a node burns.
        b_first = {}
        for n in set([b[0] for b in burnt]):
            b_times = []
            for tup in burnt:
                if tup[0] == n:
                    b_times.append(tup[1])
            for tup in burnt:
                if tup[0] == n and tup[1] == min(b_times):
                    b_first[tup[0]] = min(b_times)
        b_first = dict(sorted(b_first.items(), key=operator.itemgetter(1)))

        # Find earliest time step at which a node is defended.
        d_first = {}
        for n in set([d[0] for d in defended]):
            d_times = []
            for tup in defended:
                if tup[0] == n:
                    d_times.append(tup[1])
            for tup in defended:
                if tup[0] == n and tup[1] == min(d_times):
                    d_first[tup[0]] = min(d_times)
        d_first = dict(sorted(d_first.items(), key=operator.itemgetter(1)))

        containment = True
        tot_burnt = len(b_first)
        tot_defended = len(d_first)
        tot_saved = size - tot_burnt
        tot_saved_indirect = size - tot_burnt - tot_defended

        if plot == True:
            for n in list(G.nodes):
                G.nodes[n]['b/n/d'] = 0
            for n in b_first.keys():
                G.nodes[n]['b/n/d'] = -1
            for n in d_first.keys():
                G.nodes[n]['b/n/d'] = 1

            dgf.get_fig(graph=G,
                        time=T,
                        graph_type_proper = 'Subgraph of Grid',
                        algorithm_proper = 'Simple Weighted MOO LP',
                        f=f,
                        k=p,
                        title_ONOFF='ON')
            #nx.draw(G1, pos=nx.get_node_attributes(G1, 'pos'))
            nx.draw_networkx_labels(G, pos=nx.get_node_attributes(G, 'pos'), font_color='w')
            plt.show()
    else:
        containment = False
        tot_burnt = 0
        tot_defended = 0
        tot_saved = 0
        tot_saved_indirect = 0
        print('NO OPTIMAL SOLUTION FOR T_MAX = {}'.format(T-1))

    return T-1, tot_burnt, tot_defended, tot_saved, tot_saved_indirect, containment


# G = nx.ladder_graph(10)
# nx.draw_spring(G, node_size=200, width=1.5, with_labels=True)
# plt.show()

# print(solve_lp(G,
#                T_init=2,
#                num_ffs=1,
#                start_nodes=[1,2],
#                alpha=0.5))

T_max = 5
G = dgf.load_graph(file_loc='../data/inf-power/inf-power.mtx')
subgraphs = dgf.xsubgraphs(G, x=5, k=T_max+2, seed=42)

g = subgraphs[0]
# start_nodes = []
# for n in g.nodes:
#     if g.nodes[n]['start_nodes'] == 1:
#         start_nodes.append(n)
# print(solve_lp(g,
#                T_init=1,
#                num_ffs=1,
#                start_nodes=start_nodes,
#                alpha=0.5))

alpha_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []
G_num = 0
for g in subgraphs:
    for a in alpha_vals:
        T, tot_burnt, tot_defended, tot_saved, tot_saved_indirect, containment = \
            find_optimal_T(g,
                       f=1,
                       p=1,
                       T=1,
                       T_upper=T_max,
                       alpha=a,
                       plot=False)
        ans_val = [G_num, a, T, tot_burnt, tot_defended, tot_saved, tot_saved_indirect, containment]
        ans_key = ['G Num', 'Alpha', 'T', 'Burnt', 'Defended',
                   'Saved', 'Saved (Indirect)',
                   'Contained']
        ans_dict = {ans_key[i]:ans_val[i] for i in range(len(ans_key))}
        results.append(ans_dict)
    G_num+=1

df = pd.DataFrame(results)
print(df)

# dgf.plot_T0(g)