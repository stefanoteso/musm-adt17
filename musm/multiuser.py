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
    """Normalizes the rows of a matrix w.r.t. their Euclidean norm."""
    if w.ndim != 2:
        raise ValueError('I only work with 2D arrays')
    v = np.array(w, copy=True)
    for i in range(len(v)):
        v[i] /= (np.linalg.norm(v[i]) + 1e-13)
    return v


def crossvalidate(problem, dataset, set_size, uid, w, var, cov,
                  transform, old_alpha, lmbda=0.5):
    """Finds the best hyperparameters using cross-validation.

    Parameters
    ----------
    WRITEME

    Returns
    -------
    alpha : tuple
        The best hyperparameter.
    """

    if len(dataset) % _NUM_FOLDS != 0:
        return old_alpha

    kfold = KFold(len(dataset), n_folds=_NUM_FOLDS)
    f = compute_transform(uid, w, var, cov, transform, lmbda=lmbda)

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


def pseudoregret(w, x):
    temp = -np.inf
    set_size = len(w)
    for i in range(set_size):
        for j in list(range(0, i)) + list(range(i + 1, set_size)):
            temp = max(temp, np.dot(w[i], x[i] - x[j]))
    return temp


def select_user(var, datasets, uid_to_w, uid_to_x, regrets, satisfied_users, pick, rng):
    """Selects the user to query."""
    satisfied_users = list(satisfied_users)

    def max_to_1(values):
        values = np.array(values, dtype=float)
        values[satisfied_users] = -np.inf
        max_value = values.max()
        return np.array([np.isclose(value, max_value) for value in values])

    def min_to_1(values):
        values = np.array(values, dtype=float)
        values[satisfied_users] = -np.inf
        min_value = values.max()
        return np.array([np.isclose(value, min_value) for value in values])

    if pick == 'random':
        pvals = np.ones_like(var)
    elif pick == 'maxvar':
        pvals = max_to_1(var)
    elif pick == 'numqueries':
        pvals = min_to_1(list(map(len, datasets)))
    elif pick == 'pseudoregret':
        pvals = max_to_1([pseudoregret(uid_to_w[uid], uid_to_x[uid])
                          for uid in range(len(var))])
    elif pick == 'regret': # NOTE for debug only
        pvals = max_to_1(regrets)
    else:
        raise ValueError('invalid pick')

    pvals[list(satisfied_users)] = 0
    pvals = pvals / pvals.sum()
    uid = np.argmax(rng.multinomial(1, pvals=pvals))
    assert not uid in satisfied_users

    return uid


def compute_var_cov(uid_to_w, tau=0.25):
    """Computes the variance (spread) and covariance (kernel) of users."""
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
            sqdists.append(np.dot(diff, diff))
        cov[uid,vid] = np.exp(-tau * np.array(sqdists).mean())

    assert (cov >= 0).all(), 'cov is negative'
    assert (cov <= 1 + 1e-1).all(), 'cov is too large'

    return var, cov


def compute_transform(uid, uid_to_w, var, cov, transform, lmbda=0.5):
    """Computes the linear transformation of the aggregate utility function."""
    others = sorted(vid for vid, w in uid_to_w.items()
                    if vid != uid and w is not None)

    if transform == 'indep' or not len(others):
        return 1, np.array([0])

    uid_to_w1 = {vid: uid_to_w[vid].mean(axis=0) for vid in others}

    if transform == 'sumcov':
        a = (1 - lmbda)
        b = lmbda * sum([cov[uid,vid] * uid_to_w1[vid] for vid in others])
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
         pick='maxvar', transform='indep', lmbda=0.5, tau=0.25, rng=None):
    """Runs the MUSM algorithm on a group of users.

    Parameters
    ----------
    group : list of User
        The group of users to interact with.
    set_size : int, defaults to 2
        The size of the query set.
    max_iters : int, defaults to 100
        Maximum number of iterations to run for.
    enable_cv : bool, defaults to False
        Whether to perform hyperparameter cross-validation.
    pick : str, defaults to 'maxvar'
        Criterion use for picking a user at each iteration.
    transform : str, defaults to 'indep'
        What kind of transform to use for computing the aggregate (multi-user)
        utility.
    tau : float > 0, defaults to 0.25
        Inverse temperature of the inter-user kernel.
    rng : WRITEME
        The RNG.

    Returns
    -------
    trace : list
        It holds one element per iteration; each element holds the regret of
        all users, the ID of the selected user, and the time spen at that
        iteration.
    """
    rng = check_random_state(rng)
    num_users = len(group)

    _LOG.info('running musm, {num_users} users, k={set_size}, T={max_iters}',
              **locals())

    uid_to_w1 = {uid: None for uid in range(num_users)}

    w, x = problem.select_query([], set_size, _DEFAULT_ALPHA)
    uid_to_w = {uid: normalize(w) for uid in range(num_users)}
    uid_to_x = {uid: x for uid in range(num_users)}
    var, cov = compute_var_cov(uid_to_w, tau=tau)

    uid_to_w_star = {uid: normalize(group[uid].w_star.reshape(1,-1))
                     for uid in range(num_users)}
    var_star, cov_star = compute_var_cov(uid_to_w_star, tau=tau)

    _LOG.debug(dedent('''\
            initial var =
            {var}
            initial cov =
            {cov}
            var* =
            {var_star}
            cov* =
            {cov_star}
        ''').format(**locals()))

    satisfied_users = set()
    datasets = [np.empty((0, problem.num_attributes)) for _ in group]
    alphas = [_DEFAULT_ALPHA for _ in group]

    regrets = np.ones(num_users)

    trace = []
    for t in range(max_iters):
        t0 = time()
        uid = select_user(var, datasets, uid_to_w, uid_to_x, regrets, satisfied_users,
                          pick, rng)

        f = compute_transform(uid, uid_to_w1, var, cov, transform, lmbda=lmbda)
        w, query_set = problem.select_query(datasets[uid], set_size,
                                            alphas[uid], transform=f)
        uid_to_w[uid] = normalize(w)
        uid_to_x[uid] = query_set
        t0 = time() - t0

        i_star = group[uid].query_choice(query_set)

        t1 = time()
        for i in range(set_size):
            if i != i_star:
                delta = (query_set[i_star] - query_set[i]).reshape(1, -1)
                if (delta == 0).all():
                    _LOG.warning('all-zero delta added!')
                datasets[uid] = np.append(datasets[uid], delta, axis=0)

        var, cov = compute_var_cov(uid_to_w, tau=tau)

        f = compute_transform(uid, uid_to_w1, var, cov, transform, lmbda=lmbda)
        w, x = problem.select_query(datasets[uid], 1,
                                    alphas[uid], transform=f)
        uid_to_w1[uid] = normalize(w)
        t1 = time() - t1

        for vid, user in enumerate(group):
            ff = compute_transform(vid, uid_to_w1, var, cov, transform, lmbda=lmbda)
            w, x = problem.select_query(datasets[vid], 1,
                                        alphas[vid], transform=ff)
            regrets[vid] = user.regret(x[0])

            if user.is_satisfied(x[0]):
                satisfied_users.add(vid)

        _LOG.debug('''\
                {t:3d} var={var} regrets={regrets} uid={uid} {satisfied_users}
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
                                        uid_to_w1, var, cov, transform,
                                        alphas[uid], lmbda=lmbda)
        t2 = time() - t2

        trace.append(list(regrets) + [uid, t0 + t1 + t2])

        if len(satisfied_users) == len(group):
            break

    _LOG.info('{} users satisfied after {} iterations'
              .format(len(satisfied_users), max_iters))

    return trace
