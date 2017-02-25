#!/usr/bin/env python3

import numpy as np
import itertools as it

_ALL_ALPHAS = list(it.product(
    (20, 20, 5, 1),
    (10, 1, 0.1, 0.01),
    (10, 1, 0.1, 0.01),
))


def _crossvalidate(problem, dataset, hyperparams, set_size, num_folds=5):
    log = get_logger('setmargin')

    folds = KFold(len(dataset), n_folds=num_folds)

    alpha_to_perfs = {}
    for alpha in hyperparams:
        perfs = []
        for tr, ts in folds:
            tr_dataset = dataset[tr]
            ts_dataset = dataset[ts]
            W, _ = _solve_setmargin(problem, dataset[tr], set_size, alpha)
            utils = np.dot(ts_dataset, W)
            perfs.append(np.sum(utils[utils > 0]).mean())
        alpha_to_perfs[alpha] = sum(perfs) / len(perfs)

    best_alpha = sorted(alpha_to_perfs.items(),
                        key=lambda pair: pair[-1])[-1][0]

    log.info('''\
        crossvalidation
        alpha_to_perfs = {alpha_to_perfs}
        best_alpha = {best_alpha}
    ''', **locals())

    return best_alpha

def select_query_miqp(problem, dataset, set_size, alpha, num_threads=1):
    pass

def select_query_milp(problem, dataset, set_size, alpha, num_threads=1):
    log = get_logger('setmargin')

    if not all([a >= 0 for a in alphas]):
        raise ValueError('invald alphas {}'.format(alpha))

    # TODO check that all attributes are either Boolean or lin-dep reals
    w_top = 10000

    log.debug('''\
        selecting query (set_size={} alpha={})
        w = {}
    ''', **locals())

    model = MiniZincModel(problem.template)
    model.par('CONFIGS', set(range(1, len(set_size))))
    model.par('EXAMPLES', set(range(1, len(dataset))))

    output_vars = []
    for i in range(1, self.set_size + 1):

        attributes = problem._mzn_attributes('x{}_'.format(i))
        for attr, attr_type in attributes.items():
            model.var(attr_type, attr)
            output_vars.append(attr)

        weight_vector = 'w{}'.format(i)
        model.var('array[ATTRIBUTES] of var 0 .. infinity', weight_vector)
        output_vars.append(weight_vector)

    model.var('array[CONFIGS, CONFIGS, ATTRIBUTES] of var 0.0 .. infinity', 'p')
    model.var('array[CONFIGS, EXAMPLES] of var 0.0 .. infinity', 'xi')
    model.var('var 0.0 .. infinity', 'margin')

    model.par('alpha', alpha)
    model.var('var float', 'objective',
              'margin - alpha[1]*sum(xi) - alpha[2]*sum(w) + alpha[3]*sum(p[i,i,z] | i in CONFIGS, z in ATTRIBUTES)')
    model.solve('maximize objective')

    # Eq. 9
    for k, (dx, y) in enumerate(dataset):
        k = k + 1
        score = 'sum(w[i,z] * delta[z] | z in ATTRIBUTES)'
        if y == 0:
            model.constraint('forall(i in CONFIGS)({score} <= xi[i,{k}])'.format(**locals()))
            model.constraint('forall(i in CONFIGS)({score} >= -xi[i,{k}])'.format(**locals()))
        else:
            model.constraint('forall(i in CONFIGS)({score} >= xi[i,{k}])'.format(**locals()))

    # Eq. 10
    model.constraint('forall(i, j in CONFIGS where i != j)(sum(p[i,i,z] - p[i,j,z] | z in ATTRIBUTES) >= margin)')

    # Eq. 11
    model.constraint('forall(i in CONFIGS, z in ATTRIBUTES)(p[i,i,z] <= w_max*x[i,z])')

    # Eq. 12
    model.constraint('forall(i in CONFIGS, z in ATTRIBUTES)(p[i,i,z] <= w[i,z])')

    # Eq. 13
    model.constraint('''\
        forall(i, j in in CONFIGS where i != j, z in ATTRIBUTES)(
            p[i,j,z] >= (w[i,z] - 2*w_max*(1 - x[j,z])
        ''')

    # Eq. 15
    model.constraint('forall(i in CONFIGS, z in ATTRIBUTES)(w[i,z] <= w_max)')

    # Eq. 20
    if set_size == 1 and all(y == 0 for _, y in dataset):
        # NOTE avoids getting an unbounded problem in this specific case
        model.constraint('margin = 0')

    # TODO add hard constraints
#            for body, head in dataset.x_constraints:
#                model.addConstr((1 - x[body]) + grb.quicksum([x[atom] for atom in head]) >= 1)

    assignment = minizinc(model,
                          timeout=timeout,
                          suppress_segfault=timeout is not None,
                          parallel=num_threads)[0]

    attributes = self._mzn_attributes()
    query_set = []
    for i in range(1, set_size + 1):
        prefix = 'x{}_'.format(i)
        query_set.append({assignment[prefix + attr] for attr in attributes})

    return query_set


def setmargin(problem, user, set_size=2, max_iters=100, alpha='auto'):
    log = get_logger('setmargin')

    if max_iters <= 0:
        raise ValueError('max_iters must be >= 0, got {}'.format(max_iters))

    w_star, x_star = user.w_star, user.x_star
    alpha = (1, 1, 1)

    log.info('''\
        starting user
        w* = {w_star}
        x* = {x_star}
    ''', **locals())

    trace, dataset = [], np.empty((0, len(w_star)))
    for t in range(max_iters):

        t0 = time()
        W, Q = select_query(problem, dataset, set_size, alpha)
        t0 = time() - t0

        i_star = user.query_choice(Q)

        t1 = time1()
        deltas = [Q[i_star] - Q[i] for i in range(set_size) if i != i_star]
        map(lambda delta: np.append(dataset, delta, axis=0), deltas)

        _, x = select_query(problem, dataset, 1, alpha)
        regret = user.regret(x)
        t1 = time() - t1

        log.debug('''\
            ITERATION {t:3d}
            W = {W}
            Q = {Q}
            x = {x}
            i_star = {i_star}
            regret = {regret}
            alpha = {alpha}
        ''', **locals())

        trace.append((regret, t0 + t1))

        if user.is_satisfied:
            log.info('user satisfied in {} iterations'.format(t))
            break

        if t >= 2 and cv_hyperparams is not None:
            alpha = _crossvalidate(problem, dataset, cv_hyperparams, set_size)

    else:
        log.info('user not satisfied after {} iterations'.format(t))

    return trace
