from jmetal.core.problem import BinaryProblem, Problem
from jmetal.core.solution import BinarySolution, Solution

# jmetal assumes minimization

class WeightedMOO(BinaryProblem):
    def __init__(self, G, T):
        """ Weighted MOO (effectively single objective)
        :type G: networkx graph
        :param G: Networkx graph of interest
        :type T: int
        :param T: Max time step for the problem
        """
        super(WeightedMOO, self).__init__(reference_front=None)
        self.G = G
        self.T = T

        self.number_of_objectives = 1
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['MaxSave w/ t-FIRE']

        self.number_of_variables = G.size()*T*2
        # 5 constraints for each variable at each timestep, 3 constraints for each node at timestep 0
        self.number_of_constraints = (self.number_of_variables)*5*T + (G.size())*3

    ############# HOW TF DO WE DO THIS?#########
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # g = self.eval_g(solution)
        # h = self.eval_h(solution.variables[0], g)
        obj = 1000000

        for index, var in enumerate(solution.variables[0]):
            obj += var

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = self.number_of_objectives

        return solution
