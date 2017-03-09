import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state
from time import time

from . import get_logger


_LOG = get_logger('adt17')


def select_user(var, satisfied_users, rng):
    """Select a user to be queried. Ties are broken at random."""
    temp = np.array(var)
    temp[satisfied_users] = -np.inf
    pvals = np.array([var == temp.max() for var in temp])
    pvals = pvals / pvals.sum()
    return np.argmax(rng.multinomial(1, pvals=pvals))



def compute_kernel_row(weights, u):
    """Compute a row of the kernel matrix."""
    row = np.zeros(len(weights))
    wu = weights[u]
    for v, wv in enumerate(weights):
        # TODO normalize
        row[v] = np.sum(np.dot(wu.T, wv))
    return row


def compute_transformation(cov, var, u, transform='sumk'):
    if transform is None:
        return None
    others = [v for v in range(len(var)) if v != u]
    if transform == 'sumk':
        a = 1
        b = cov[u,others].sum()
    else:
        raise NotImplementedError()
    return a, b


def musm(problem, group, set_size=2, max_iters=100, transform='sumk',
         rng=None):
    """Runs the multi-user setmargin algorithm."""
    rng = check_random_state(rng)

    num_users, num_attributes = len(group), problem.num_attributes

    group_str = '\n'.join([str(user) for user in group])
    _LOG.info('''\
            running multi-user setmargin on {num_users} users (k={set_size}, T={max_iters})
            {group_str}
        ''', **locals())

    datasets = [np.empty((0, num_attributes)) for _ in group]
    weights = np.zeros((num_users, set_size, num_attributes))
    cov, var = np.eye(num_users), np.ones(num_users)

    alpha = (1, 1, 1)

    trace, satisfied_users = [], []
    for t in range(max_iters):
        t0 = time()

        u = select_user(var, satisfied_users, rng)
        assert u not in satisfied_users
        dataset = datasets[u]

        p = compute_transformation(cov, var, u, transform=transform)
        w, query_set = problem.select_query(dataset, set_size, alpha,
                                            transform=p)
        t0 = time() - t0

        i_star = group[u].query_choice(query_set)

        t1 = time()

        for i in range(set_size):
            if i != i_star:
                delta = (query_set[i_star] - query_set[i]).reshape(1, -1)
                datasets[u] = dataset = np.append(dataset, delta , axis=0)

        weights[u] = w
        var[u] = np.sum(pdist(w)**2)
        cov[u,:] = cov[:,u] = compute_kernel_row(weights, u)

        regrets, num_satisfied = [], 0
        for v, user in enumerate(group):
            p = compute_transformation(cov, var, v, transform=transform)
            _, x = problem.select_query(dataset, 1, alpha, transform=p)
            x = x[0]

            regrets.append(user.regret(x))

            if user.is_satisfied(x):
                satisfied_users.append(v)

        t1 = time() - t1

        _LOG.debug('''\
                ITERATION {t:3d}
                u = {u}
                w = {w}
                q = {query_set}
                i_star = {i_star}
                datasets =
                {datasets}
                x = {x}
                regrets =
                {regrets}
            ''', **locals())

        trace.append((u, regrets, t0 + t1))

        # TODO crossvalidation

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), t))

    return trace
