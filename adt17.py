#!/usr/bin/env python3

import sys
import os
import logging
import numpy as np
import musm

from sklearn.utils import check_random_state


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
        args['problem'], args['num_groups'], args['num_users_per_group'],
        args['max_iters'], args['set_size'],
        args['min_regret'], args['user_distrib'], args['density'],
        args['response_model'], args['noise'],
        args['seed'],
    ]
    return os.path.join('results', '_'.join(map(str, properties)) + '.pickle')


def _sparsify(w, density, rng):
    if not (0 < density <= 1):
        raise ValueError('density must be in (0, 1], got {}'.format(density))
    w = np.array(w)
    perm = rng.permutation(len(w))
    num_zeros = round((1 - density) * len(w))
    w[perm[:min(num_zeros, len(w) - 1)]] = 0
    return w


def sample_user_group(problem, num_users=5, user_distrib='normal', density=1,
                      rng=0, **kwargs):
    if user_distrib == 'uniform':
        w = rng.uniform(25, 25/4, size=(num_users, problem.num_attributes))
    elif user_distrib == 'normal':
        w = rng.uniform(1, 100+1, size=(num_users, problem.num_attributes))
    else:
        raise ValueError('invalid user_distrib, got {}'.format(user_distrib))
    for u in range(len(w)):
        w[u] = _sparsify(np.abs(w[u]), density, rng)
    return w


def generate_user_groups(problem, nopargs):
    User = USERS[nopargs['response_model']]

    # we use a fixed RNG here for reproducibility
    rng = check_random_state(0)

    user_groups = []
    for gid in range(nopargs['num_groups']):
        w_star = sample_user_group(problem, rng=rng, **nopargs)
        user_groups.append([User(problem, w_star[u],
                                 min_regret=nopargs['min_regret'],
                                 noise=nopargs['noise'],
                                 rng=rng)
                           for u in range(nopargs['num_users_per_group'])])

    return user_groups


def run(args):
    # build the problem instance
    problem = PROBLEMS[args['problem']]()

    # build the user groups
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
                                set_size=args['set_size'],
                                max_iters=args['max_iters'],
                                rng=0))

    musm.dump(get_results_path(args), {'args': args, 'traces': traces})


def main():
    import argparse

    np.seterr(all='raise')

    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt)
    parser.add_argument('problem', type=str,
                        help='the problem, any of {}'
                             .format(sorted(PROBLEMS.keys())))
    parser.add_argument('-N', '--num-groups', type=int, default=20,
                        help='number of user groups')
    parser.add_argument('-M', '--num-users-per-group', type=int, default=5,
                        help='number of users per group')
    parser.add_argument('-T', '--max-iters', type=int, default=100,
                        help='maximum number of elicitation iterations')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable debug spew')

    group = parser.add_argument_group('Query Selection')
    group.add_argument('-k', '--set-size', type=int, default=2,
                       help='set size')

    group = parser.add_argument_group('User Simulation')
    group.add_argument('--min-regret', type=float, default=0,
                       help='minimum regret for satisfaction')
    group.add_argument('-G', '--groups', type=str, default=None,
                       help='path to pickle with user weights')
    group.add_argument('-u', '--user-distrib', type=str, default='normal',
                       help='distribution of user weights')
    group.add_argument('-d', '--density', type=float, default=1,
                       help='percentage of non-zero user weights')
    group.add_argument('-R', '--response-model', type=str, default='pl',
                       help='user response model for choice queries')
    group.add_argument('-n', '--noise', type=float, default=1,
                       help='amount of user response noise')

    args = parser.parse_args()

    handlers = []
    if args.verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers,
                        format='%(levelname)-6s %(name)-14s: %(message)s')

    run(vars(args))

if __name__ == '__main__':
    main()
