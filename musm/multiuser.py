import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state
from time import time

from . import get_logger


_LOG = get_logger('adt17')


def select_user(variance, satisfied_users, rng):
    """Select a user to be queried. Ties are broken at random."""
    temp = np.array(variance)
    temp[satisfied_users] = -1
    pvals = np.array([var == temp.max() for var in temp])
    pvals = pvals / len(pvals)
    return np.argmax(rng.multinomial(1, pvals=pvals))



def compute_kernel_row(weights, u):
    """Compute a row of the kernel matrix."""
    row = np.zeros(len(weights))
    wu = weights[u]
    for v, wv in enumerate(weights):
        # TODO normalize
        row[v] = np.sum(np.dot(wu.T, wv))
    return row


def compute_transformation(similarity, variance, u, transform='sumk'):
    m = len(variance)
    if transform == 'sumk':
        a = 1,
        b = similarity[u,[v for v in range(m) if m != u]].sum()
    return a, b


def musm(problem, group, set_size=2, max_iters=100, transform='sumk',
         rng=None):
    """Runs the multi-user setmargin algorithm."""
    rng = check_random_state(rng)

    num_users, num_attributes = len(group), problem.num_attributes

    group_str = '\n'.join([str(user) for user in group])
    _LOG.info('''\
            running multi-user setmargin ({num_users} users, {set_size}, {max_iters})
            {group_str}
        ''', **locals())

    datasets = [np.empty((0, num_attributes)) for _ in group]
    weights = np.zeros((num_users, set_size, num_attributes))
    similarity, variance = np.eye(num_users), np.ones(num_users)

    alpha = (1, 1, 1)

    trace, satisfied_users = [], []
    for t in range(max_iters):
        t0 = time()
        u = select_user(variance, satisfied_users, rng)
        p = compute_transformation(similarity, variance, u,
                                   transform=transform)
        w, query_set = problem.select_query(datasets[u], set_size, alpha,
                                            transform=p)
        t0 = time() - t0

        i_star = group[u].query_choice(query_set)

        t1 = time()
        deltas = [query_set[i_star] - query_set[i] for i in range(set_size)
                  if i != i_star]
        for delta in deltas:
            datasets[u] = np.append(datasets[u], delta.reshape(1, -1), axis=0)

        weights[u] = w

        similarity[u,:] = similarity[:,u] = compute_kernel_row(weights, u)

        variance[u] = np.sum(pdist(w)**2)

        regrets, num_satisfied = [], 0
        for user in group:
            p = compute_transformation(similarity, variance, u,
                                       transform=transform)
            _, x = problem.select_query(datasets[u], 1, alpha, transform=p)
            x = x[0]
            regrets.append(user.regret(x))
            if user.is_satisfied(x):
                satisfied_users.append(u)
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

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), t))

    return trace
