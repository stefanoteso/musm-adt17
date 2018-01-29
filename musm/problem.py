import numpy as np
import itertools as it
import gurobipy as gurobi
from gurobipy import GRB as G
from textwrap import dedent
from math import*
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


def dot(x, z):
    grutil = gurobi.quicksum([x[i] * z[i] for i in range(len(x))])
    return grutil



def bilinear(x, A, z):
    return dot(x, [dot(a[i], z) for i in range(len(x))])

def L1_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))
def Ep_distance(x,y):
    return sum(a-b for a, b in zip(x, y))


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
#Implement groupwise query selection

    #Compute x_star

    def benchmark(self, W, omega_star):


        W = np.squeeze(np.asarray(W))
        omega_star = np.squeeze(np.asarray(omega_star))
        ws_star = np.dot(W, omega_star, out=None)


        model = gurobi.Model('inference')
        model.params.Threads = self.num_threads
        model.params.Seed = 0
        model.params.OutputFlag = 0

        x_star = [model.addVar(vtype=G.BINARY) for z in range(self.num_attributes)]

        model.modelSense = G.MAXIMIZE
        model.update()

        model.setObjective(dot(ws_star,x_star))
        self._add_constraints(model, x_star)
        model.optimize()



        x_star = np.array([x_star[z].x for z in range(self.num_attributes)])

        print("This is True X =", x_star)

        _LOG.debug('inferred {}'.format(x_star))

        return x_star


    def infer_query(self, W ,omega):
        LAMBDA = 0.5
        M = 1000000

        # XXX w_star is supposed to be a matrix of shape (num_attributes,
        # num_users), each column encodes the preferences of one user; omega
        # a vector of shape (num_users,), each element encodes the importance
        # of one user. Is this vvv correct in this case?

        #print ("Shape of aggregate_utility =", W.shape, " = (num_attributes, num_users)")
        #print ("Shape of omega =", omega.shape," = (num_users,)")
        #omega = omega[:,None]
        W = np.squeeze(np.asarray(W))
        omega = np.squeeze(np.asarray(omega))
        ws_star = np.dot(W, omega, out=None)

        model = gurobi.Model('inference')
        model.params.Threads = self.num_threads
        model.params.Seed = 0
        model.params.OutputFlag = 0

        # XXX
        # x1 = model.addVar(vtype=G.BINARY)
        # x2 = model.addVar(vtype=G.BINARY)
        # d = model.addVar(vtype=G.INTEGER, lb=-1, ub=1)
        # a = model.addVar(vtype=G.INTEGER, lb=0, ub=1)
        # model.modelSense = G.MAXIMIZE
        # model.update()
        # model.setObjective(x1 + 2 * x2 + 100 * a)
        # model.addConstr(d == x1 - x2)
        # model.addGenConstrAbs(a, d)
        # model.write('problem.lp')
        # model.optimize()
        # print(model.objVal)
        # print('x1 =', x1.x)
        # print('x2 =', x2.x)
        # print('diff =', d.x)
        # print('|diff| =', a.x)
        # quit()


        x1 = [model.addVar(vtype=G.BINARY, name='x1'+str(z)) for z in range(self.num_attributes)]
        x2 = [model.addVar(vtype=G.BINARY, name='x2'+str(z)) for z in range(self.num_attributes)]

        f1 = dot(ws_star, x1)
        f2 = dot(ws_star, x2)

        diff = [model.addVar(vtype=G.INTEGER, lb=-1, ub=1, name='diff'+str(z))
                for z in range(self.num_attributes)]
        absdiff = [model.addVar(vtype=G.INTEGER, lb=0, ub=1, name='absdiff'+str(z))
                for z in range(self.num_attributes)]

        model.modelSense = G.MAXIMIZE
        model.update()

        # objective fun page 3 of notes
        if np.all(ws_star) == 0:
            model.setObjective(LAMBDA * (dot(ws_star, x1) + dot(ws_star, x2)) + \
                           (1 - LAMBDA) * gurobi.quicksum(absdiff))
        else:

            model.setObjective(
                LAMBDA * 1 / np.sum(np.abs(ws_star)) * (f1 + f2) + \
                (1 - LAMBDA) * 1 / self.num_attributes * gurobi.quicksum(absdiff))

        for z in range(self.num_attributes):
            # diff[z] == x1[z] - x2[z]
            model.addConstr(diff[z] == x1[z] - x2[z])
            # absdiff[z] == abs{diff[z]}
            model.addGenConstrAbs(absdiff[z], diff[z])

        self._add_constraints(model, x1)
        self._add_constraints(model, x2)
        model.write('problem.lp')
        model.optimize()

        if model.status == G.Status.OPTIMAL:
            print('Optimal objective: %g' % model.objVal)
        elif model.status == G.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            exit(0)
        elif model.status == G.Status.INFEASIBLE:
            print('Model is infeasible')
            exit(0)
        elif model.status == G.Status.UNBOUNDED:
            print('Model is unbounded')
            exit(0)
        else:
            print('Optimization ended with status %d' % model.status)
            exit(0)

        x1 = np.array([x1[z].x for z in range(self.num_attributes)])
        x2 = np.array([x2[z].x for z in range(self.num_attributes)])

        _LOG.debug('inferred {} {}'.format(x1, x2))

        print ("This is X1", x1)
        print ("This is X2", x2)
        return x1, x2


