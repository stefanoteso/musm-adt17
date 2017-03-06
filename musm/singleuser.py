import numpy as np
from sklearn.cross_validation import KFold
import itertools as it
from time import time

from . import get_logger


_LOG = get_logger('adt17')

_ALL_ALPHAS = list(it.product(
        (20, 20, 5, 1),
        (10, 1, 0.1, 0.01),
        (10, 1, 0.1, 0.01),
    ))


def _crossvalidate(problem, dataset, set_size, num_folds=5):
    folds = KFold(len(dataset), n_folds=num_folds)

    alpha_to_perfs = {}
    for alpha in _ALL_ALPHAS:
        perfs = []
        for tr, ts in folds:
            tr_dataset = dataset[tr]
            ts_dataset = dataset[ts]
            weights, _ = problem.select_query(dataset[tr], set_size, alpha)
            signs = np.dot(weights, ts_dataset.T) > 0
            perfs.append(signs.mean())
        alpha_to_perfs[alpha] = sum(perfs) / len(perfs)

    best_alpha = sorted(alpha_to_perfs.items(),
                        key=lambda pair: pair[-1])[-1][0]

    _LOG.info('''\
            crossvalidation
            alpha_to_perfs = {alpha_to_perfs}
            best_alpha = {best_alpha}
        ''', **locals())

    return best_alpha


def setmargin(problem, user, set_size=2, max_iters=100, cv=True, **kwargs):
    w_star, x_star = user.w_star, user.x_star

    _LOG.info('''\
            starting user
            w* = {w_star}
            x* = {x_star}
        ''', **locals())

    alpha = (1, 1, 1)

    trace, dataset = [], np.empty((0, len(w_star)))
    for t in range(max_iters):

        t0 = time()
        weights, query_set = problem.select_query(dataset, set_size, alpha)
        t0 = time() - t0

        i_star = user.query_choice(query_set)

        t1 = time()
        deltas = [query_set[i_star] - query_set[i] for i in range(set_size)
                  if i != i_star]
        for delta in deltas:
            dataset = np.append(dataset, delta.reshape(1, -1), axis=0)

        _, x = problem.select_query(dataset, 1, alpha)
        x = x[0]
        regret = user.regret(x)
        t1 = time() - t1

        _LOG.debug('''\
                ITERATION {t:3d}
                w = {weights}
                q = {query_set}
                i_star = {i_star}
                dataset =
                {dataset}
                x = {x}
                regret = {regret}
                alpha = {alpha}
            ''', **locals())

        trace.append((regret, t0 + t1))

        if user.is_satisfied(x):
            _LOG.info('user satisfied in {} iterations'.format(t))
            break

        if cv and (t - 4) % 5 == 0:
            alpha = _crossvalidate(problem, dataset, set_size)

    else:
        _LOG.info('user not satisfied after {} iterations'.format(t))

    return trace
