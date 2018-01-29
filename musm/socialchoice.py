import numpy as np
import itertools as it
from scipy.spatial.distance import pdist
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from textwrap import dedent
import time
import csv


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
    if w.ndim != 2:
        raise ValueError('I only work with 2D arrays')
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


def select_user(var, datasets, satisfied_users, pick, rng):
    # TODO implement centrality
    if pick == 'random':
        pvals = np.ones_like(var)
    elif pick == 'maxvar':
        temp = np.array(var)
        temp[list(satisfied_users)] = -np.inf
        maxvar = temp.max()
        pvals = np.array([np.isclose(v, maxvar) for v in var])
    elif pick == 'invnumqueries':
        pvals = [1 / (1 + len(datasets[u])) for u in range(len(var))]
    else:
        raise ValueError('invalid pick')
    pvals[list(satisfied_users)] = 0
    pvals = pvals / pvals.sum()
    uid = np.argmax(rng.multinomial(1, pvals=pvals))
    assert not uid in satisfied_users
    return uid


def compute_var_cov(uid_to_w, tau=0.25):
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


def compute_transform(uid, uid_to_w, var, cov, transform, lmbda):
    others = sorted(vid for vid, w in uid_to_w.items()
                    if vid != uid and w is not None)

    if transform == 'indep' or not len(others):
        return 1, np.array([0])

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


def musm(problem, group,gid, set_size=2, max_iters=100, enable_cv=False,
         pick='maxvar', transform='indep', tau=0.25, lmbda=0.5, rng=None,):
    rng = check_random_state(rng)
    num_users = len(group)

    _LOG.info('running musm, {num_users} users, k={set_size}, T={max_iters}',
              **locals())
    # we create the matrix aggragate !!

    W = np.vstack((u.w_star for u in group)).T

    #print (W.shape)
    #print (" W = ", W)
    # Initialize omega at random
    omega_star = rng.rand(len(group))
    print ("Initial_Omega_star = ", omega_star)

    ni = 1  # learning rate

    loss = []
    cumulative_time =[]


    x_star = problem.benchmark(W,omega_star)

    #omega = rng.rand(len(group))
    omega = np.zeros(len(group))
    print("Group ID ====================", gid)

    print ("start omega =========", omega)
    start = time.time()
    for t in range(max_iters):


        x1,x2 = problem.infer_query(W,omega) # finding x1 and x2 that maximize the objective function

        # Group Response (Social choice(x with greater aggregate_utility))

        ws_new = np.dot(W, omega)

        util1  = np.dot(ws_new,x1)
        util2  = np.dot(ws_new,x2)

        if util1 > util2:
            delta = (x1-x2).reshape(1, -1)

        else:
            delta = (x2 - x1).reshape(1, -1)

        """delta is x with greater aggregate_utility minius x with smaller aggregate_utility (group)"""

        # perceptrone update
        omega += ni * np.dot(delta, W).ravel()
        print ("Learned_Omega = ", omega)



        # Compute Utility loss
        ws_true = np.dot(W, omega_star)
        if util1 > util2:
            print ("x_star",x_star)
            print ("x1",x1)

            #utility_loss =  np.dot(ws_true,x_star) - np.dot(ws_true,x1)
            utility_loss =  np.dot(ws_true,x_star) - np.dot(ws_true,x1)

        else:
            print ("x_star",x_star)
            print ("x2", x2)
            #utility_loss =  np.dot(ws_true,x_star) - np.dot(ws_true,x2)
            utility_loss =  np.dot(ws_true,x_star) - np.dot(ws_true,x2)

        print('utility_loss ======================================', utility_loss)
        loss.append(utility_loss)

        end = time.time()
        T = end - start
        cumulative_time.append(T)
        print("Time = ", cumulative_time)

        # Normalize Utility loss

    loss = np.squeeze(np.asarray(loss))
    #print('loss =', loss)

    loss_normed = loss
        #(loss - loss.min(0)) / loss.ptp(0)
    print ( "Loss_normalized =",loss_normed)

    # Save Loss data
    '''csvfile = "/Users/bereket/Documents/Social/setmargin/musm-adt17/result_data/synthetic_loss_20_1_normal_1.0.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in loss:
            print(val)
            writer.writerow([val])'''




     # Save Normalized Loss_data

    '''csvfile = "/Users/bereket/Documents/Social/setmargin/musm-adt17/result_data/synthetic_loss_20_1_uniform"
    with open(csvfile+"_"+str(gid)+".csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in loss_normed:
            print(val)
            writer.writerow([val])


    # Save cumulative time

    csvfile = "/Users/bereket/Documents/Social/setmargin/musm-adt17/result_data/pc_time_20_2_uniform.csv"
    with open(csvfile+"_"+str(gid)+".csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cumulative_time:
            print(val)
            writer.writerow([val])'''


        #print('Loss_data = ', loss)
	 # update function modifies omega according to the direction of delta, so delta will tell us how to modify omega

    _LOG.info('{} omega learned after {} iterations/ convergence'.format(len(omega),max_iters))

    return omega




