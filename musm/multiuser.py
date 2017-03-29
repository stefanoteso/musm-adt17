import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from textwrap import dedent
from time import time

from . import get_logger


_LOG = get_logger('adt17')

_ALPHAS = list(it.product(
         [10, 1],
         [1, 0.1, 0.01],
         [1, 0.1, 0.01],
    ))
_I_TO_ALPHA = {i: alpha for i, alpha in enumerate(_ALPHAS)}

_DEFAULT_ALPHA = 10, 0.1, 0.1
_NUM_FOLDS = 3


def normalize(w):
    if w.ndim == 1:
        return w / np.linalg.norm(w)
    else:
        v = np.array(w, copy=True)
        for i in range(len(v)):
            v[i] /= (np.linalg.norm(v[i]) + 1e-13)
        return v


def crossvalidate(problem, dataset, set_size, uid, w, var, cov,
                  transform, lmbda, old_alpha):
    if len(dataset) % _NUM_FOLDS != 0:
        return old_alpha

    kfold = KFold(len(dataset), n_folds=_NUM_FOLDS)
    f = compute_transform(uid, w, var, cov, transform, lmbda)

    avg_accuracy = np.zeros(len(_ALPHAS))
    for i, alpha in enumerate(_ALPHAS):
        accuracies = []
        for tr_indices, ts_indices in kfold:
            w, _ = problem.select_query(dataset[tr_indices], set_size, alpha,
                                        transform=f)
            utilities = np.dot(w, dataset[ts_indices].T)
            accuracies.append((utilities > 0).mean())
        avg_accuracy[i] = sum(accuracies) / len(accuracies)

    alpha = _I_TO_ALPHA[np.argmax(avg_accuracy)]

    _LOG.debug('''\
            alpha accuracies = {avg_accuracy}
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


def compute_var_cov(uid_to_w):
    num_users = len(uid_to_w)

    known_users = [uid for uid, w in uid_to_w.items() if w is not None]
    set_size = uid_to_w[known_users[0]].shape[0] if len(known_users) else 1

    var = np.ones(num_users)
    for uid in known_users:
        for i, j in it.product(range(set_size), repeat=2):
            if j <= i:
                continue
            diff = uid_to_w[uid][i] - uid_to_w[uid][j]
            var[uid] += np.dot(diff, diff)
    if set_size >= 2:
        # XXX this is a loose upper bound to the max variance
        var[known_users] /= 2 * set_size * (set_size - 1)

    assert (var >= 0).all(), 'var is negative'
    assert (var <= 1).all(), 'var is too large'

    cov = np.eye(num_users)
    for uid, vid in it.product(known_users, known_users):
        sqdists = []
        for i, j in it.product(range(set_size), repeat=2):
            diff = uid_to_w[uid][i] - uid_to_w[vid][j]
            sqdists.append(np.dot(diff, diff) / 4)
        cov[uid,vid] = np.exp(-np.array(sqdists).mean())

    assert (cov >= 0).all(), 'cov is negative'
    assert (cov <= 1 + 1e-1).all(), 'cov is too large'

    return var, cov


def compute_transform(uid, uid_to_w, var, cov, transform, lmbda):
    others = sorted(vid for vid, w in uid_to_w.items()
                    if vid != uid and w is not None)

    if transform == 'indep' or not len(others):
        return 1, 0

    uid_to_w1 = {vid: uid_to_w[vid].mean(axis=0) for vid in others}

    if transform == 'sumcov':
        a = lmbda
        b = (1 - lmbda) * sum([cov[uid,vid] * uid_to_w1[vid]
                               for vid in others])
    elif transform == 'varsumvarcov':
        a = (1 - var[uid])
        b = var[uid] * sum([cov[uid,vid] * (1 - var)[vid] * uid_to_w1[vid]
                            for vid in others])
    else:
        raise NotImplementedError('invalid transform, {}'.format(transform))

    # NOTE this clamps close-to-zero negative values to zero; in theory it
    # should not be required...
    b[np.logical_and(-1e-2 <= b, b <= 0)] = 0

    if a <= 0 or (b < 0).any():
        eigval, _ = np.linalg.eigh(cov)
        msg = dedent('''\
                uid = {uid}, others = {others}
                var = {var}
                cov =
                {cov}
                eigval(cov) =
                {eigval}
                a = {a}
                b = {b}
            ''').format(**locals())
        raise RuntimeError(msg)

    return a, b


def musm(problem, group, set_size=2, max_iters=100, enable_cv=False,
         transform='indep', sources='all', lmbda=0.5, rng=None):
    rng = check_random_state(rng)
    num_users = len(group)

    _LOG.info('running musm, {num_users} users, k={set_size}, T={max_iters}',
              **locals())

    w, _ = problem.select_query([], set_size, _DEFAULT_ALPHA)
    uid_to_w = {uid: normalize(w) for uid in range(num_users)}

    var, cov = compute_var_cov(uid_to_w)

    uid_to_w1 = {uid: None for uid in range(num_users)}

    satisfied_users = set()
    datasets = [np.empty((0, problem.num_attributes)) for _ in group]
    alphas = [_DEFAULT_ALPHA for _ in group]

    trace = []
    for t in range(max_iters):
        t0 = time()
        uid = select_user(var, satisfied_users, rng)

        f = compute_transform(uid, uid_to_w1, var, cov, transform, lmbda)
        w, query_set = problem.select_query(datasets[uid], set_size,
                                            alphas[uid], transform=f)
        uid_to_w[uid] = normalize(w)
        t0 = time() - t0

        i_star = group[uid].query_choice(query_set)

        t1 = time()
        for i in range(set_size):
            if i != i_star:
                delta = (query_set[i_star] - query_set[i]).reshape(1, -1)
                if (delta == 0).all():
                    _LOG.warning('all-zero delta added!')
                datasets[uid] = np.append(datasets[uid], delta, axis=0)

        var, cov = compute_var_cov(uid_to_w)

        f = compute_transform(uid, uid_to_w1, var, cov, transform, lmbda)
        w, x = problem.select_query(datasets[uid], 1,
                                    alphas[uid], transform=f)
        uid_to_w1[uid] = normalize(w)
        t1 = time() - t1

        regrets1 = np.zeros(num_users)
        for vid, user in enumerate(group):
            ff = compute_transform(vid, uid_to_w1, var, cov, transform, lmbda)
            w, x = problem.select_query(datasets[vid], 1,
                                        alphas[vid], transform=ff)
            regrets1[vid] = user.regret(x[0])

            if user.is_satisfied(x[0]):
                satisfied_users.add(vid)

        _LOG.debug('''\
                {t:3d} var={var} regrets={regrets1} uid={uid} {satisfied_users}
                cov = {cov}
                transform = {f}
                q = {query_set}
                i_star = {i_star}
                datasets =
                {datasets}
                uid_to_w =
                {uid_to_w}
                uid_to_w1 =
                {uid_to_w1}
            ''', **locals())

        t2 = time()
        if enable_cv:
            alphas[uid] = crossvalidate(problem, datasets[uid], set_size, uid,
                                        source_w, var, cov, transform, lmbda,
                                        alphas[uid])
        t2 = time() - t2

        trace.append(list(regrets1) + [uid, t0 + t1 + t2])

        if len(satisfied_users) == len(group):
            break

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), max_iters))

    return trace
