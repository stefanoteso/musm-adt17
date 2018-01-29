#!/usr/bin/env python3.5

import sys
import os
import logging
import numpy as np
import musm

from sklearn.utils import check_random_state
from textwrap import dedent

#1Social Choice
_LOG = musm.get_logger('adt17')

PROBLEMS = {
    'synthetic': musm.Synthetic,
    'pc': musm.PC,
}

USERS = {
    'noiseless': musm.NoiselessUser,
    'pl': musm.PlackettLuceUser,
}


def get_results_path(args):
    properties = [
        args['problem'], args['num_groups'], args['num_clusters_per_group'],
        args['num_users_per_group'], args['max_iters'], args['set_size'],
        args['pick'], args['transform'], args['tau'], args['lmbda'],
        args['enable_cv'], args['min_regret'], args['distrib'],
        args['density'], args['response_model'], args['noise'], args['seed'],
    ]
    return os.path.join('results', '_'.join(map(str, properties)) + '.pickle')


def _sparsify(w, density, rng):
    if not (0 < density <= 1):
        raise ValueError('density must be in (0, 1], got {}'.format(density))
    w = np.array(w, copy=True)
    perm = rng.permutation(w.shape[1])
    num_zeros = round((1 - density) * w.shape[1])
    w[:,perm[:min(num_zeros, w.shape[1] - 1)]] = 0
    return w


def sample_cluster(problem, num_users=5, distrib='normal', density=1, rng=0):
    num_attributes = problem.num_attributes
    if hasattr(problem, 'cost_matrix'):
        num_attributes += problem.cost_matrix.shape[0]

    if distrib == 'uniform':
        w_mean = rng.uniform(0, 1, size=num_attributes)
    elif distrib == 'normal':
        w_mean = rng.uniform(-1, 1, size=num_attributes)
    else:
        raise ValueError('invalid distrib, got {}'.format(distrib))

    if True: # XXX
        w = w_mean + np.zeros((num_users, num_attributes))
    else:
        w = w_mean + rng.uniform(0, 25, size=(num_users, num_attributes))

    return _sparsify(np.abs(w), density, rng)


def generate_user_groups(problem, args):
    User = USERS[args['response_model']]

    rng = check_random_state(0)

    num_users_per_cluster = max(1, round(args['num_users_per_group'] /
                                         args['num_clusters_per_group']))

    user_groups = []
    for gid in range(args['num_groups']):

        w_star = []
        for cid in range(1, args['num_clusters_per_group'] + 1):
            if cid == args['num_clusters_per_group']:
                num_users_in_cluster = args['num_users_per_group'] - len(w_star)
            else:
                num_users_in_cluster = num_users_per_cluster

            temp = sample_cluster(problem,
                                  num_users=num_users_in_cluster,
                                  distrib=args['distrib'],
                                  density=args['density'],
                                  rng=rng)

            ttemp = temp
            if hasattr(problem, 'cost_matrix'):
                num_costs = problem.cost_matrix.shape[0]
                temp_bools = temp[:, :-num_costs]
                temp_costs = temp[:, -num_costs:]
                ttemp = temp_bools + np.dot(temp_costs, problem.cost_matrix)

            _LOG.debug(dedent('''\
                    CLUSTER {cid}:
                    true user weights =
                    {temp}
                    true user weights transformed by cost matrix =
                    {ttemp}
                ''').format(**locals()))

            if len(w_star) == 0:
                w_star = ttemp
            else:
                w_star = np.append(w_star, ttemp, axis=0)

        user_groups.append([User(problem,
                                 w_star[uid],
                                 min_regret=args['min_regret'],
                                 noise=args['noise'],
                                 rng=rng)
                           for uid in range(args['num_users_per_group'])])

    return user_groups


def run(args):
    problem = PROBLEMS[args['problem']]()

    try:
        user_groups = musm.load(args['groups'])
    except:
        user_groups = generate_user_groups(problem,
                                           musm.subdict(args, nokeys={'problem'}))
        if args['groups'] is not None:
            musm.dump(args['groups'], user_groups)

    rng = check_random_state(args['seed'])

    traces = []
    for gid in range(args['num_groups']):
        traces.append(musm.musm(problem,
                                user_groups[gid],
                                gid,
                                set_size=args['set_size'],
                                max_iters=args['max_iters'],
                                enable_cv=args['enable_cv'],
                                pick=args['pick'],
                                transform=args['transform'],
                                tau=args['tau'],
                                lmbda=args['lmbda'],

                                rng=0))

    musm.dump(get_results_path(args), {'args': args, 'traces': traces})


def main():
    import argparse

    np.seterr(all='raise')
    np.set_printoptions(precision=2, linewidth=1000000)

    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt)

    group = parser.add_argument_group('Experiment')
    group.add_argument('problem', type=str,
                        help='the problem, any of {}'
                             .format(sorted(PROBLEMS.keys())))
    group.add_argument('-N', '--num-groups', type=int, default=20,
                       help='number of user groups')
    group.add_argument('-C', '--num-clusters-per-group', type=int, default=1,
                       help='number of clusters in a group')
    group.add_argument('-M', '--num-users-per-group', type=int, default=5,
                       help='number of users in a group')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='maximum number of elicitation iterations')
    group.add_argument('-s', '--seed', type=int, default=0,
                       help='RNG seed')
    group.add_argument('-v', '--verbose', action='store_true',
                       help='enable debug spew')

    group = parser.add_argument_group('Algorithm')
    group.add_argument('-K', '--set-size', type=int, default=2,
                       help='set size')
    group.add_argument('-P', '--pick', type=str, default='maxvar',
                       help='critertion used for picking users')
    group.add_argument('-F', '--transform', type=str, default='indep',
                       help='user-user transformation to use')
    group.add_argument('-t', '--tau', type=float, default=0.25,
                       help='kernel inverse temperature parameter')
    group.add_argument('-L', '--lmbda', type=float, default=0.5,
                       help='transform importance')
    group.add_argument('-X', '--enable-cv', action='store_true',
                       help='enable hyperparameter cross-validation')

    group = parser.add_argument_group('User Simulation')
    group.add_argument('--min-regret', type=float, default=0,
                       help='minimum regret for satisfaction')
    group.add_argument('-G', '--groups', type=str, default=None,
                       help='path to pickle with user weights')
    group.add_argument('-u', '--distrib', type=str, default='normal',
                       help='distribution of user weights')
    group.add_argument('-d', '--density', type=float, default=1,
                       help='proportion of non-zero user weights')
    group.add_argument('-R', '--response-model', type=str, default='pl',
                       help='user response model for choice queries')
    group.add_argument('-n', '--noise', type=float, default=1,
                       help='amount of user response noise')

    args = parser.parse_args()

    handlers = []
    if args.verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers,
                        format='%(levelname)-6s %(name)-6s %(funcName)-12s: %(message)s')

    run(vars(args))

if __name__ == '__main__':
    main()
