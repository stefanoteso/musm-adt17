import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from time import time

from . import get_logger


_LOG = get_logger('adt17')

_ALPHAS = list(it.product(
         [10, 1],
         [1, 0.1, 0.01],
         [1, 0.1, 0.01],
    ))
_I_TO_ALPHA = {i: alpha for i, alpha in enumerate(_ALPHAS)}

_DEFAULT_ALPHA = 1, 0.1, 0.1
_NUM_FOLDS = 5


def crossvalidate(problem, dataset, set_size, uid, var, cov, transform):
    if len(dataset) < _NUM_FOLDS:
        return _DEFAULT_ALPHA

    kfold = KFold(len(dataset), n_folds=_NUM_FOLDS)
    p = compute_transform(cov, var, transform, uid)

    avg_accuracy = np.zeros(len(_ALPHAS))
    for i, alpha in enumerate(_ALPHAS):
        accuracies = []
        for tr_indices, ts_indices in kfold:
            w, _ = problem.select_query(dataset[tr_indices], set_size, alpha,
                                        transform=p)
            utilities = np.dot(w, dataset[ts_indices].T)
            accuracies.append((utilities > 0).mean())
        avg_accuracy[i] = sum(accuracies) / len(accuracies)

    alpha = _I_TO_ALPHA[np.argmax(avg_accuracy)]

    _LOG.debug('''\
            crossvalidation says:
            avg_accuracy = {avg_accuracy}
            best alpha = {alpha}
        ''', **locals())

    return alpha


def select_user(var, satisfied_users, rng):
    temp = np.array(var)
    temp[list(satisfied_users)] = -np.inf
    pvals = np.array([var == temp.max() for var in temp])
    pvals = pvals / pvals.sum()
    return np.argmax(rng.multinomial(1, pvals=pvals))


def compute_var_cov(w):
    if w.ndim == 2:
        num_users, set_size = len(w), 1
    else:
        num_users, set_size = w.shape[0], w.shape[1]

    var = np.zeros(num_users)
    for uid in range(num_users):
        for i, j in it.product(range(set_size), repeat=2):
            diff = w[uid,i] - w[uid,j]
            var[uid] += np.dot(diff, diff)
    var = var / (2 * set_size**2)

    cov = np.zeros((num_users, num_users))
    for uid1, uid2 in it.product(range(num_users), repeat=2):
        p = np.dot(w[uid1], w[uid2].T).sum()
        q1 = np.dot(w[uid1], w[uid1].T).sum()
        q2 = np.dot(w[uid2], w[uid2].T).sum()
        cov[uid1, uid2] = p / np.sqrt(q1 * q2)

    return var, cov


def compute_transform(cov, var, transform, uid):
    if transform is None:
        return 1, 1
    others = [i for i in range(len(var)) if i != uid]
    if transform == 'sumk':
        a, b = 1, cov[uid, others].sum()
    else:
        raise NotImplementedError('transform = {}'.format(transform))
    assert a >= 0 and (b >= 0).all(), 'invalid transform {}, {}'.format(a, b)
    return a, b


def musm(problem, group, set_size=2, max_iters=100, enable_cv=False,
         transform='sumk', rng=None):
    rng = check_random_state(rng)

    num_users, num_attributes = len(group), problem.num_attributes

    group_str = '\n'.join([str(user) for user in group])
    _LOG.info('''\
            running multi-user setmargin on {num_users} users (k={set_size}, T={max_iters})
            {group_str}
        ''', **locals())

    datasets = [np.empty((0, num_attributes)) for _ in group]

    weights = rng.uniform(0, 1, size=(num_users, set_size, num_attributes))
    var, cov = compute_var_cov(weights)

    alphas = [_DEFAULT_ALPHA for _ in group]

    trace, satisfied_users = [], set()
    for t in range(max_iters):
        t0 = time()

        uid = select_user(var, satisfied_users, rng)
        assert uid not in satisfied_users

        p = compute_transform(cov, var, transform, uid)
        w, query_set = problem.select_query(datasets[uid], set_size, alphas[uid],
                                            transform=p)
        t0 = time() - t0

        i_star = group[uid].query_choice(query_set)

        t1 = time()

        for i in range(set_size):
            if i != i_star:
                delta = (query_set[i_star] - query_set[i]).reshape(1, -1)
                datasets[uid] = np.append(datasets[uid], delta , axis=0)

        weights[uid] = w
        var, cov = compute_var_cov(weights)

        regrets = np.zeros(len(group))
        for vid, user in enumerate(group):
            p = compute_transform(cov, var, transform, vid)
            _, x = problem.select_query(datasets[vid], 1, alphas[uid],
                                        transform=p)
            x = x[0]

            regrets[vid] = user.regret(x)
            if user.is_satisfied(x):
                satisfied_users.add(vid)

        t1 = time() - t1

        _LOG.debug('''\
                ITERATION {t:3d}
                uid = {uid}
                var = {var}
                cov = {cov}
                w = {w}
                q = {query_set}
                i_star = {i_star}
                datasets =
                {datasets}
                x = {x}
                regrets = {regrets}({satisfied_users})
            ''', **locals())

        trace.append(list(regrets) + [uid, t0 + t1])

        if len(satisfied_users) == len(group):
            break

        if enable_cv:
            alphas[uid] = crossvalidate(problem, datasets[uid], set_size, uid,
                                        var, cov, transform)

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), max_iters))

    return trace
