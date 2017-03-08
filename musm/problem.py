import numpy as np
import itertools as it
import gurobipy as gurobi
from gurobipy import GRB as G
from textwrap import dedent

from . import get_logger, freeze


_LOG = get_logger('adt17')


def dot(x, z):
    return gurobi.quicksum([x[i] * z[i] for i in range(len(x))])


def bilinear(x, A, z):
    return dot(x, [dot(a[i], z) for i in range(len(x))])


class Problem(object):
    def __init__(self, template, num_attributes):
        self.num_attributes = num_attributes

    def infer(self, w, transform=None):
        assert not hasattr(self, 'cost_matrix')

        transformed_w = w
        if transform is not None:
            a, b = transform
            transformed_w = a * w + b
            assert (transformed_w >= 0).all()

        _LOG.debug(dedent('''
                INFERENCE
                w = {}
                transformed w = {}
            ''').format(w, transformed_w))

        model = gurobi.Model('inference')

        model.params.Threads = 1
        model.params.Seed = 0
        model.params.OutputFlag = 0

        x = [model.addVar(vtype=G.BINARY) for z in range(self.num_attributes)]

        model.modelSense = G.MAXIMIZE
        model.setObjective(dot(w, x))
        self._add_constraints(model, x)
        model.optimize()

        x = np.array([x[z].x for z in range(self.num_attributes)])

        _LOG.debug('inferred {}'.format(x))

        return x

    def select_query(self, dataset, set_size, alpha, transform=None):
        assert not hasattr(self, 'cost_matrix')

        _LOG.debug(dedent('''\
                SELECTING QUERY SET k={} alpha={}
                dataset =
                {}
            ''').format(set_size, alpha, dataset))

        w_min = np.zeros(self.num_attributes)
        w_max = np.ones(self.num_attributes)
        if transform is not None:
            a, b = transform
            w_min = a * w_min + b
            w_max = a * w_max + b
            assert (w_min >= 0).all() and (w_max >= 0).all()
        w_top = w_max.max()

        model = gurobi.Model('queryselection')
        model.params.Threads = 1
        model.params.Seed = 0
        model.params.OutputFlag = 0

        x = {(i, z): model.addVar(vtype=G.BINARY)
             for i in range(set_size) for z in range(self.num_attributes)}
        w = {(i, z): model.addVar(lb=0, vtype=G.CONTINUOUS)
             for i in range(set_size) for z in range(self.num_attributes)}
        p = {(i, j, z): model.addVar(lb=0, vtype=G.CONTINUOUS)
             for i, j, z in it.product(range(set_size), range(set_size), range(self.num_attributes))}
        slacks = {(i, s): model.addVar(lb=0, vtype=G.CONTINUOUS)
                  for i in range(set_size) for s in range(len(dataset))}
        margin = model.addVar(vtype=G.CONTINUOUS)

        p_diag = [p[i,i,z] for i in range(set_size) for z in range(self.num_attributes)]

        # eq 4
        model.modelSense = G.MAXIMIZE
        model.setObjective(margin
                           - alpha[0] * gurobi.quicksum(slacks.values())
                           - alpha[1] * gurobi.quicksum(w.values())
                           + alpha[2] * gurobi.quicksum(p_diag))

        # eq 5
        for i in range(set_size):
            for s, delta in enumerate(dataset):
                udiff = gurobi.quicksum([w[i,z] * delta[z] for z in range(self.num_attributes)])
                model.addConstr(udiff >= margin - slacks[i,s])

        # eq 6
        for i, j in it.product(range(set_size), repeat=2):
            if i != j:
                udiff = gurobi.quicksum([p[i,i,z] - p[i,j,z] for z in range(self.num_attributes)])
                model.addConstr(udiff >= margin)

        # eq 7
        for i, z in it.product(range(set_size), range(self.num_attributes)):
            model.addConstr(p[i,i,z] <= w_top * x[i,z])
            model.addConstr(p[i,i,z] <= w[i,z])

        # eq 8
        for i, j in it.product(range(set_size), repeat=2):
            if i != j:
                for z in range(self.num_attributes):
                    model.addConstr(p[i,j,z] >= (w[i,z] - 2 * w_top * (1 - x[j,z])))

        # eq 9a
        for i in range(set_size):
            for z in range(self.num_attributes):
                model.addConstr(w[i,z] >= w_min[z])
                model.addConstr(w[i,z] <= w_max[z])

        # work around unbounded problems
        if set_size == 1 and len(dataset) == 0:
            model.addConstr(margin == 0)

        # add hard constraints
        for k in range(set_size):
            self._add_constraints(model, [x[k,z] for z in range(self.num_attributes)])

        model.optimize()

        output_w = np.zeros((set_size, self.num_attributes))
        output_x = np.zeros((set_size, self.num_attributes))
        for i, z in it.product(range(set_size), range(self.num_attributes)):
            output_w[i,z] = w[i,z].x
            output_x[i,z] = x[i,z].x
        w, x = output_w, output_x

        _LOG.debug(dedent('''\
                selected
                w =
                {}
                x =
                {}
            ''').format(w, x))

        return w, x
