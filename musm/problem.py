import numpy as np
import itertools as it
import gurobipy as gurobi
from gurobipy import GRB as G
from textwrap import dedent

from . import get_logger, freeze, subdict


_LOG = get_logger('adt17')


_STATUS = {
    1: 'LOADED',
    2: 'OPTIMAL',
    3: 'INFEASIBLE',
    4: 'INF_OR_UNBD',
    5: 'UNBOUNDED',
    6: 'CUTOFF',
    7: 'ITERATION_LIMIT',
    8: 'NODE_LIMIT',
    9: 'TIME_LIMIT',
    10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED',
    12: 'NUMERIC',
    13: 'SUBOPTIMAL',
}


def dict2array(d):
    indices = np.array(list(d.keys()))
    if not len(indices):
        return None
    ndim = len(indices[0])
    shape = [indices[:, dim].max() + 1 for dim in range(ndim)]
    array = np.zeros(shape)
    for index in map(tuple, indices):
        array[index] = d[index].x
    return array


class Problem(object):
    def __init__(self, num_attributes, num_threads=0):
        self.num_attributes = num_attributes
        self.num_threads = num_threads

    def infer(self, w, transform=(1, 0)):
        """Computes a highest-utility configuration w.r.t. the given weights.

        Parameters
        ----------
        w : ndarray of shape (num_attributes,)
            A weight vector.
        transform : tuple of (float, 1D ndarray)
            The transformation parameters (a, b).

        Returns
        -------
        x : ndarray of shape (num_attributes,)
            An optimal configuration.
        """

        a, b = transform
        transformed_w = a * w + b
        assert (transformed_w >= 0).all()

        _LOG.debug(dedent('''
                INFERENCE
                w = {}
                transformed w = {}
            ''').format(w, transformed_w))

        model = gurobi.Model('inference')

        model.params.OutputFlag = 0
        model.params.Threads = self.num_threads
        model.params.Seed = 0

        x = [model.addVar(vtype=G.BINARY) for z in range(self.num_attributes)]

        model.modelSense = G.MAXIMIZE
        model.setObjective(gurobi.quicksum([w[i] * x[i] for i in range(self.num_attributes)]))
        self._add_constraints(model, x)
        model.optimize()

        x = np.array([x[z].x for z in range(self.num_attributes)])

        _LOG.debug('inferred {}'.format(x))

        return x

    def select_query(self, dataset, set_size, alpha, transform=(1, 0)):
        """Solves the set-wise max-margin problem for a given set size.

        Parameters
        ----------
        dataset : ndarray of shape (num_examples, num_attributes)
            Array of collected feedback of the form $x^+ - x^-$.
        set_size : int
            Size of the query set.
        alpha : tuple of float
            Hyperparameters
        transform : tuple of (float, 1D ndarray)
            The transformation parameters (a, b).

        Returns
        -------
        w : ndarray of shape (set_size, num_attributes)
            Optimal weight vectors.
        x : ndarray of shape (set_size, num_attributes)
            Optimal configurations.
        """

        w_min = np.zeros(self.num_attributes)
        w_max = np.ones(self.num_attributes)

        a, b = transform
        w_min = a * w_min + b
        w_max = a * w_max + b
        assert (w_min >= 0).all() and (w_max >= 0).all()

        w_top = w_max.max()

        _LOG.debug(dedent('''\
                SELECTING QUERY SET k={set_size} alpha={alpha}
                dataset =
                {dataset}
                transform = {transform}
                w_min = {w_min}
                w_max = {w_max}
                w_top = {w_top}
            ''').format(**subdict(locals(), nokeys=['self'])))

        model = gurobi.Model('queryselection')

        model.params.OutputFlag = 0
        model.params.Threads = self.num_threads
        model.params.Seed = 0

        x = {(i, z): model.addVar(vtype=G.BINARY, name='x_{}_{}'.format(i, z))
             for i in range(set_size) for z in range(self.num_attributes)}
        w = {(i, z): model.addVar(lb=0, vtype=G.CONTINUOUS, name='w_{}_{}'.format(i, z))
             for i in range(set_size) for z in range(self.num_attributes)}
        p = {(i, j, z): model.addVar(lb=0, vtype=G.CONTINUOUS, name='p_{}_{}_{}'.format(i, j, z))
             for i, j, z in it.product(range(set_size), range(set_size), range(self.num_attributes))}
        slacks = {(i, s): model.addVar(lb=0, vtype=G.CONTINUOUS, name='slack_{}_{}'.format(i, s))
                  for i in range(set_size) for s in range(len(dataset))}
        margin = model.addVar(vtype=G.CONTINUOUS, name='margin')

        p_diag = [p[i,i,z] for i in range(set_size) for z in range(self.num_attributes)]
        obj_slacks = 0
        if len(slacks) > 0:
            obj_slacks = gurobi.quicksum(slacks.values())

        # eq 4
        model.modelSense = G.MAXIMIZE
        model.setObjective(margin
                           - alpha[0] * obj_slacks
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
            model.addConstr(p[i,i,z] <= (w_top * x[i,z]))
        for i, z in it.product(range(set_size), range(self.num_attributes)):
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
        apply_workaround = set_size == 1 and len(dataset) == 0
        if apply_workaround:
            model.addConstr(margin == 0)

        # add hard constraints
        for i in range(set_size):
            self._add_constraints(model, [x[i,z] for z in range(self.num_attributes)])

        try:
            model.optimize()
            model.objVal
        except gurobi.GurobiError:
            status = _STATUS[model.status]
            msg = dedent('''\
                    unsatisfiable, reason: {status}

                    set_size = {set_size}
                    alpha = {alpha}
                    dataset =
                    {dataset}
                    transform = {transform}
                ''').format(**locals())
            model.write('failed.lp')
            raise RuntimeError(msg)

        x = dict2array(x)
        w = dict2array(w)
        p = dict2array(p)
        slacks = dict2array(slacks)
        margin = margin.x

        _LOG.debug(dedent('''\
                SELECTED QUERY SET:
                utilities
                w =
                {w}
                x =
                {x}
                p =
                {p}
                slacks =
                {slacks}
                margin = {margin}
            ''').format(**locals()))

        if not apply_workaround and (w == 0).all():
            _LOG.warning('all-zero weights are bad')

        return w, x
