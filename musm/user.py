import numpy as np
from scipy.misc import logsumexp
from sklearn.utils import check_random_state
from textwrap import dedent

from . import get_logger, subdict


__all__ = ['NoiselessUser', 'PlackettLuceUser']

_LOG = get_logger('adt17')


class User(object):
    def __init__(self, problem, w_star, min_regret=0, noise=0, rng=None):
        self.problem = problem
        self.w_star = w_star
        self.min_regret = min_regret
        self.noise = noise
        self.rng = check_random_state(rng)

        self.x_star = self.problem.infer(self.w_star)
        self.u_star = self.utility(self.x_star)

    def __repr__(self):
        return 'User({w_star}, {x_star}, {u_star}; {noise})'.format(**vars(self))

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
        pvals = pvals / pvals.sum()
        i_star = np.argmax(self.rng.multinomial(1, pvals=pvals))
        _LOG.debug(dedent('''\
                selecting best item from
                utils = {}
                pvals = {}
                i_star = {}
            ''').format(utils, pvals, i_star))
        return i_star

    def query_ranking(self, query_set):
        raise NotImplementedError()


class NoiselessUser(User):
    def _utils_to_pvals(self, utils):
        return np.array([u == utils.max() for u in utils])


class PlackettLuceUser(User):
    def _utils_to_pvals(self, utils):
        return np.exp(utils - logsumexp(self.noise * utils))
