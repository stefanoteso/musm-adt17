import gurobipy as gurobi

from .problem import Problem

class Synthetic(Problem):
    def __init__(self, domain_sizes=4, **kwargs):
        try:
            # Assumes a list of integers
            len(domain_sizes)
            self.domain_sizes = domain_sizes
        except TypeError:
            # Assumes an integer
            self.domain_sizes = [domain_sizes] * domain_sizes
        super().__init__(sum(self.domain_sizes))

    def _add_constraints(self, model, x):
        base = 0
        for domain_size in self.domain_sizes:
            x_domain = [x[z] for z in range(base, base + domain_size)]
            model.addConstr(gurobi.quicksum(x_domain) == 1)
            base += domain_size
