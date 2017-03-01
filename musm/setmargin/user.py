import numpy as np
from scipy.misc import logsumexp
from sklearn.utils import check_random_state


def _sparsify(w, density, rng):
    if not (0 < density <= 1):
        raise ValueError('density must be in (0, 1], got {}'.format(density))
    w = np.array(w)
    perm = rng.permutation(len(w))
    w[perm[:round((1 - density) * len(w))]] = 0
    return w


def sample_users(problem, mode='normal', density=1, non_negative=False,
                 rng=None):
    n = problem.num_attributes
    rng = check_random_state(rng)
    if mode == 'uniform':
        w = rng.uniform(0, 1, size=n)
    else:
        w = rng.normal(-1, 1, size=n)
    if non_negative:
        w = np.abs(w)
    return _sparsify(w, density, rng)


class User(object):
    def __init__(self, problem, w_star, min_regret=0, uid=0, rng=None):
        self.problem = problem
        self.w_star = w_star
        self.min_regret = min_regret
        self.uid = uid
        self.rng = check_random_state(rng)

        self.x_star = self.problem.infer(self.w_star)
        self.u_star = self.utility(self.x_star)

    def utility(self, x):
        return np.dot(self.w_star, x)

    def regret(self, x):
        return self.u_star - self.utility(x)

    def is_satisfied(self, x):
        return self.regret(x) <= self.min_regret

    def query_choice(self, query_set):
        if len(query_set) < 2:
            raise ValueError('Expected >= 2 items, got {}'
                             .format(len(query_set)))
        utils = np.array([self.utility(x) for x in query_set])
        pvals = self._utils_to_pvals(utils)
        return np.argmax(self.rng.multinomial(1, pvals=(pvals / pvals.sum())))

    def query_ranking(self, query_set):
        raise NotImplementedError()


class NoiselessUser(User):
    def _utils_to_pvals(self, utils):
        return np.array([u == utils.max() for u in utils])


class PlackettLuceUser(User):
    def __init__(self, lmbda=1, **kwargs):
        super().__init__(**kwargs)
        self.lmbda = lmbda

    def _utils_to_pvals(self, utils):
        return np.exp(utils - logsumexp(self.lmbda * utils))
