#!/usr/bin/env python3

import sys
import os
import logging
import numpy as np
import musm

from sklearn.utils import check_random_state


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
        args.problem, args.num_users, args.max_iters, args.set_size,
        args.min_regret, args.user_distrib, args.density, args.non_negative,
        args.response_model, args.noise, args.seed,
    ]
    return os.path.join('results', '_'.join(map(str, properties)) + '.pickle')


def generate_users(problem, nopargs):
    User = USERS[args.response_model]

    rng = check_random_state(0)

    users = []
    for uid in range(1, args.num_users + 1):
        w_star = musm.sample_users(problem, rng=rng, **nopargs)
        users.append(User(problem, w_star, **nopargs))

    return users


def run(args):
    problem = PROBLEMS[args.problem]()

    rng = check_random_state(args.seed)
    nopargs = musm.subdict(args, nokeys={'problem'})

    try:
        users = musm.load(args.users)
    except:
        users = generate_users(problem, nopargs)
        musm.dump(args['weights'], users)

    for uid in range('num_users'):
        traces.append(musm.setmargin(problem, user, rng=rng))

    musm.dump(get_results_path(args),
                   {'args': var(args), 'traces': traces})


def main():
    import argparse

    np.seterr(all='raise')

    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt)
    parser.add_argument('problem', type=str,
                        help='the problem, any of {}'
                             .format(sorted(PROBLEMS.keys())))
    parser.add_argument('-N', '--num-users', type=int, default=20,
                        help='number of users in the experiment')
    parser.add_argument('-T', '--max-iters', type=int, default=100,
                        help='number of trials')
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
    group.add_argument('-U', '--users', type=str, default=None,
                       help='path to pickle with ')
    group.add_argument('-u', '--user-distrib', type=str, default='normal',
                       help='user weight distribution')
    group.add_argument('-d', '--density', type=float, default=1,
                       help='percentage of non-zero user weights')
    group.add_argument('--non-negative', action='store_true', default=False,
                       help='whether the weights should be non-negative')
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

    run(args)

if __name__ == '__main__':
    main()
