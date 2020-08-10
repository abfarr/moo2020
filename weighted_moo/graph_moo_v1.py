import autograd.numpy as anp # wrapper for numpy to help with differentiation... can we just use np?
from pymoo.model.problem import Problem

class GraphMoo(Problem):
    def __init__(self, G, T, start_nodes, alpha=0.1):
        # Store necessary constants
        self.G = G
        self.T = T
        self.start_nodes = start_nodes
        self.alpha = alpha

        # Make dictionary of neighbors for later access
        self.nbrs_dict = {}
        for n in G.nodes:
            self.nbrs_dict[n] = [nbr for nbr in G[n]]
        print(self.n_nbrs_dict)

        # total variables = nodes in the graph * T (for each time step) * 2 (for burnt and defended variables)
        tot_var = G.size() * T * 2
        x_lower_bound = 0 * anp.ones(tot_var)
        x_upper_bound = 1 * anp.ones(tot_var)
        # 5 constraints for each variable at each time step, 3 constraints for each node at timestep 0
        tot_constraints = tot_var * 5 * T + (G.size()) * 3

        # super() allows us to directly call functions that are in the pymoo Problem class
        super().__init__(n_var=tot_var,
                         n_obj=2,
                         n_constr=tot_constraints,
                         xl=x_lower_bound,
                         xu=x_upper_bound,
                         type_var=anp.int)
    # B is array
    def _evaluate(self, B, D, out, *args, **kwargs):
        # B has rows x and columns T so [:,-1] is B_x_T for and [:,-2] is B_x_T-1 for all x
        f1 = sum(B[:,-1])
        f2 = sum(B[:,-1]-B[:,-2])

        constraints = [] #anp.zeros(self.n_constr)
        ## Constraints must be in form so that they are greater than or equal to 0
        for i, (bx_row, dx_row) in enumerate(zip(B,D)): # i corresponds to node x
            for ic, (bx_t, dx_t) in enumerate(zip(bx_row, dx_row)): # ic corresponds to time step t
                if ic >= 1: # these constraints only apply for t>=1 and upt to T
                    for y in self.nbrs_dict[i]:
                        g0 = bx_t + dx_t - B[y,ic-1] # neighbor of burning node must be burnt or defended in next time step
                        constraints.append(g0)
                    g1 = sum(B[y,1:ic-1]) - bx_t-bx_row[ic-1] # prevent spontaneous combustion
                    g2 = 1-bx_t-dx_t # node can't be burnt and defended
                    g3 = bx_t - bx_row[ic-1] # burnt node stays burnt
                    g4 = dx_t - dx_row[ic-1] # defended node stays defended
                    constraints.append(g1)
                    constraints.append(g2)
                    constraints.append(g3)
                    constraints.append(g4)


        #### ALL CONSTRAINTS ABOVE HAVE BEEN ADDED... NEXT STEP: FIGURE OUT HOW TO ADD THESE
        for node in range(0, B.shape[0]):
            if node in self.start_nodes:
                B[node,0] = 1
            else:
                B[node,0] = 0
            D[node,0] = 0


        out["F"] = anp.column_stack([f1,f2])
        out["G"] = anp.column_stack(constraints)

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

from pymoo.optimize import minimize
import networkx as nx

res = minimize(WeightedMoo(nx.path_graph(10), T=3, start_nodes=[2]),
               algorithm,
               ('n_gen', 40),
               seed=1,
               verbose=True)

print(res)


### CONSTRAINT TESTING
# B = anp.array([['x0_t0','x0_t1','x0_t2','x0_t3','x0_t4'],
#               ['x1_t0','x1_t1','x1_t2','x1_t3','x1_t4'],
#               ['x2_t0','x2_t1','x2_t2','x2_t3','x2_t4']])
# D = anp.array([['dx0_t0','dx0_t1','dx0_t2','dx0_t3','dx0_t4'],
#               ['dx1_t0','dx1_t1','dx1_t2','dx1_t3','dx1_t4'],
#               ['dx2_t0','dx2_t1','dx2_t2','dx2_t3','dx2_t4']])
#
# nbrs_dict = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8]}
# ## Constraints must be in form so that they are greater than or equal to 0
# for i, (bx_row, dx_row) in enumerate(zip(B, D)):  # i corresponds to node x
#     for ic, (bx_t, dx_t) in enumerate(zip(bx_row, dx_row)):  # ic corresponds to time step t
#         if ic >= 1:  # these constraints only apply for t>=1 and upt to T
#             for y in nbrs_dict[i]:
#                 print('{}+{}-{}'.format(bx_t, dx_t, B[y, ic - 1])) # neighbor of burning node must be burnt or defended in next time step
#             # Can't sum strings
#             g1 = sum(B[y, 1:ic - 1]) - bx_t - bx_row[ic - 1]  # prevent spontaneous combustion
#             print('1-{}-1{}'.format(bx_t,dx_t)) # node can't be burnt and defended
#             print('{}-{}'.format(bx_t, bx_row[ic-1])) # burnt node stays burnt
#             print('{}-{}'.format(dx_t, dx_row[ic - 1])) # defended node stays defended
