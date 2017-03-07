import numpy as np
from pymzn import MiniZincModel, minizinc
from pymzn._mzn._solvers import Gurobi
from textwrap import dedent

from . import get_logger, freeze


_LOG = get_logger('adt17')

_GUROBI = Gurobi(path='/opt/libminizinc/bin/mzn-gurobi')


class Problem(object):
    def __init__(self, template, num_attributes):
        self.template = template
        self.num_attributes = num_attributes

    def _constraints(self, x):
        raise NotImplementedError()

    def infer(self, w, transform=None):
        assert not hasattr(self, 'cost_matrix')

        transformed_w = w
        if transform is not None:
            a, b = transform
            transformed_w = a * w + b
            assert (transformed_w >= 0).all()

        _LOG.debug(dedent('''\
                RUNNING INFERENCE
                w =
                {}
                transformed w =
                {}
            ''').format(w, transformed_w))

        model = MiniZincModel(self.template)

        model.par('w', transformed_w)
        model.par('ATTRS', set(range(1, self.num_attributes + 1)))

        model.var('array[ATTRS] of var {0, 1}', 'x')
        model.var('var float', 'utility', 'sum(z in ATTRS)(w[z] * x[z])')

        model.solve('maximize utility')

        for constraint in self._constraints('x'):
            model.constraint(constraint)

        x = minizinc(model, output_vars=['x'], solver=_GUROBI)[0]['x']
        x = np.array(x)

        _LOG.debug('inferred {}'.format(x))

        return x

    def select_query(self, dataset, set_size, alpha, transform=None):
        assert not hasattr(self, 'cost_matrix')

        _LOG.debug(dedent('''\
                RUNNING QS k={}, alpha={}
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

        model = MiniZincModel(self.template)

        model.par('ATTRS', set(range(1, self.num_attributes + 1)))
        model.par('QUERY', set(range(1, set_size + 1)))
        model.var('array[1 .. 3] of float', 'ALPHA', alpha)
        model.par('W_MIN', w_min)
        model.par('W_MAX', w_max)
        model.par('W_TOP', w_top)

        model.var('array[QUERY, ATTRS] of var {0, 1}', 'x')
        model.var('array[QUERY, ATTRS] of var 0.0 .. infinity', 'w')
        model.var('array[QUERY, QUERY, ATTRS] of var 0.0 .. infinity', 'p')
        model.var('var 0.0 .. infinity', 'margin')

        obj_slacks = ''
        if len(dataset) >= 1:
            model.par('DATASET', dataset)
            model.par('EXAMPLES', set(range(1, len(dataset) + 1)))
            model.var('array[QUERY, EXAMPLES] of var 0.0 .. infinity', 'slack')
            obj_slacks = '- ALPHA[1] * sum([slack[i,h] | i in QUERY, h in EXAMPLES]) '

        model.var('var float', 'objective',
                  ('margin ' +
                   obj_slacks +
                   '- ALPHA[2] * sum([w[i,z] | i in QUERY, z in ATTRS]) ' +
                   '+ ALPHA[3] * sum([p[i,i,z] | i in QUERY, z in ATTRS]) '))
        model.solve('maximize objective')

        # dataset constraints
        for s in range(1, len(dataset) + 1):
            model.constraint('''\
                forall(i in QUERY)(
                    sum(z in ATTRS)(w[i,z] * row(DATASET, {s})[z]) >= margin - slack[i,{s}])
                '''.format(s=s))

        # eq 6
        model.constraint('''\
            forall(i, j in QUERY where i != j)(
                sum([p[i,i,z] - p[i,j,z] | z in ATTRS]) >= margin)
            ''')

        # eq 7a
        model.constraint('''\
            forall(i in QUERY)(
                forall(z in ATTRS)(
                    p[i,i,z] <= (W_TOP * x[i,z])))
            ''')

        # eq 7b
        model.constraint('''\
            forall(i in QUERY)(
                forall(z in ATTRS)(
                    p[i,i,z] <= w[i,z]))
            ''')

        # eq 8
        model.constraint('''\
            forall(i, j in QUERY where i != j)(
                forall(z in ATTRS)(
                    p[i,j,z] >= (w[i,z] - 2 * W_TOP * (1 - x[j,z]))))
            ''')

        # eq 9a
        model.constraint('''\
            forall(i in QUERY)(
                forall(z in ATTRS)(
                    w[i,z] >= W_MIN[z]))
            ''')

        # eq 9b
        model.constraint('''\
            forall(i in QUERY)(
                forall(z in ATTRS)(
                    w[i,z] <= W_MAX[z]))
            ''')

        # work around unbounded problems
        if set_size == 1 and len(dataset) == 0:
            model.constraint('margin == 0')

        # add hard constraints
        for k in range(1, set_size + 1):
            xk = 'x{}'.format(k)
            model.var('array[ATTRS] of var {0, 1}', xk, 'row(x, {})'.format(k))
            for constraint in self._constraints(xk):
                model.constraint(constraint)

        assignments = minizinc(model, output_vars=['x', 'w'], solver=_GUROBI)
        assert len(assignments) > 0, 'unsatisfiable'
        w = np.array(assignments[0]['w'])
        x = np.array(assignments[0]['x'])

        _LOG.debug(dedent('''\
                selected
                w =
                {}
                x =
                {}
            ''').format(w, x))

        return w, x
