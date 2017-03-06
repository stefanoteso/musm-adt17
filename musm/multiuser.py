import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from time import time

from . import get_logger


_LOG = get_logger('adt17')


def select_user(datasets, similarity, quality):
    # TODO experiment with other selection mechanisms
    return np.argmax(quality)


def compute_kernel_row(weights, u):
    row = np.zeros(len(weights))
    wu = weights[u]
    for v, wv in enumerate(weights):
        # TODO normalize
        row[v] = np.sum(np.dot(wu.T, wv))
    return row


def musm(problem, group, set_size=2, max_iters=100):
    num_users, num_attributes = len(group), problem.num_attributes

    group_str = '\n'.join([str(user) for user in group])
    _LOG.info('''\
            running multi-user setmargin ({num_users} users, {set_size}, {max_iters})
            {group_str}
        ''', **locals())

    datasets = [np.empty((0, num_attributes)) for _ in group]
    weights = np.zeros((num_users, set_size, num_attributes))
    similarity, quality = np.eye(num_users), np.ones(num_users)

    alpha = (1, 1, 1)

    trace = []
    for t in range(max_iters):
        t0 = time()
        u = select_user(datasets, similarity, quality)
        w, query_set = problem.select_query(datasets[u], set_size, alpha)
        t0 = time() - t0

        i_star = group[u].query_choice(query_set)

        t1 = time()
        deltas = [query_set[i_star] - query_set[i] for i in range(set_size)
                  if i != i_star]
        for delta in deltas:
            datasets[u] = np.append(datasets[u], delta.reshape(1, -1), axis=0)

        weights[u] = w

        similarity[u,:] = similarity[:,u] = compute_kernel_row(weights, u)

        dist = 2 * pdist(w) / (set_size * (set_size - 1))
        quality[u] = 1 / (1 + np.exp(dist))

        regrets, num_satisfied = [], 0
        for user in group:
            _, x = problem.select_query(datasets[u], 1, alpha)
            x = x[0]
            regrets.append(user.regret(x))
            # XXX what to do with satisfied users?
            num_satisfied += user.is_satisfied(x)
        t1 = time() - t1

        dataset = datasets[u]
        _LOG.debug('''\
                ITERATION {t:3d}
                u = {u}
                w = {w}
                q = {query_set}
                i_star = {i_star}
                dataset =
                {dataset}
                x = {x}
                regrets =
                {regrets}
            ''', **locals())

        trace.append((u, regrets, t0 + t1))

        # TODO crossvalidation

    _LOG.info('{} users satisfied after {} iterations'.format(num_satisfied, t))

    return trace
