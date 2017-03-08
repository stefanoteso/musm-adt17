import gurobipy as gurobi

from .problem import Problem

class PC(Problem):
    def __init__(self, **kwargs):
        raise NotImplementedError()
        super().__init__('', 76)

    def _add_constraints(self, model, x):
        raise NotImplementedError()
