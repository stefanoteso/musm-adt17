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


def crossvalidate(problem, dataset, set_size, uid, weights, var, cov,
                  transform, old_alpha):
    if len(dataset) % _NUM_FOLDS != 0:
        return old_alpha

    kfold = KFold(len(dataset), n_folds=_NUM_FOLDS)
    p = compute_transform(weights, cov, var, transform, uid)

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
    uid = np.argmax(rng.multinomial(1, pvals=pvals))
    assert not uid in satisfied_users
    return uid


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


def compute_transform(weights, cov, var, transform, uid):
    if transform is None:
        return 1, 1

    avg_weights = weights.mean(axis=1)
    others = [i for i in range(len(var)) if i != uid]
    if transform == 'sumcov':
        a = 1
        b = np.dot(cov[uid, others], avg_weights[others])
    elif transform == 'varsumvarcov':
        negvar = 1 - var
        a = negvar[uid]
        b = var[uid] * np.dot(negvar[others] * cov[uid, others],
                              avg_weights[others])
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

        p = compute_transform(weights, cov, var, transform, uid)
        w, query_set = problem.select_query(datasets[uid], set_size,
                                            alphas[uid], transform=p)
        weights[uid] = w

        regrets_k = np.zeros(len(group))
        for vid, user in enumerate(group):
            regrets_k[vid] = min([user.regret(x) for x in query_set])

        t0 = time() - t0

        i_star = group[uid].query_choice(query_set)

        t1 = time()

        for i in range(set_size):
            if i != i_star:
                delta = (query_set[i_star] - query_set[i]).reshape(1, -1)
                datasets[uid] = np.append(datasets[uid], delta , axis=0)

        var, cov = compute_var_cov(weights)

        regrets_1 = np.zeros(len(group))
        for vid, user in enumerate(group):
            vid_p = compute_transform(weights, cov, var, transform, vid)
            _, x = problem.select_query(datasets[vid], 1,
                                        alphas[uid], transform=vid_p)
            regrets_1[vid] = user.regret(x[0])
            if user.is_satisfied(x[0]):
                satisfied_users.add(vid)

        t1 = time() - t1

        _LOG.debug('''\
                ITERATION {t:3d}
                user = {uid} {satisfied_users} {var}
                regrets@1 = {regrets_1}
                regrets@k = {regrets_k}
                var = {var}
                cov = {cov}
                w = {w}
                transform = {p}
                q = {query_set}
                i_star = {i_star}
                datasets =
                {datasets}
            ''', **locals())

        t2 = time()
        if enable_cv:
            alphas[uid] = crossvalidate(problem, datasets[uid], set_size, uid,
                                        weights, var, cov, transform, alphas[uid])
        t2 = time() - t2

        trace.append(list(regrets_1) + list(regrets_k) + [uid, t0 + t1 + t2])

        if len(satisfied_users) == len(group):
            break

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), max_iters))

    return trace
