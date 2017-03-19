#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from glob import glob
import musm


plt.style.use('ggplot')


def pad(array, length):
    full = np.zeros((length,) + array.shape[1:], dtype=array.dtype)
    full[:len(array)] = array
    return full


def load(path):
    data = musm.load(path)
    num_users = data['args']['num_users_per_group']
    max_iters = data['args']['max_iters']
    loss1_matrix, lossk_matrix, time_matrix = [], [], []
    for trace in data['traces']:
        trace = np.array(trace)
        loss1_matrix.append(pad(trace[:,:num_users], max_iters))
        lossk_matrix.append(pad(trace[:,num_users:2*num_users], max_iters))
        time_matrix.append(pad(trace[:,-1], max_iters))
    return np.array(loss1_matrix), \
           np.array(lossk_matrix), \
           np.array(time_matrix), \
           data['args']


def prettify(ax, max_iters, title):
    xtick = 5 if max_iters <= 50 else 10
    xticks = np.hstack([[1], np.arange(xtick, max_iters + 1, xtick)])
    ax.set_xticks(xticks)

    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.grid(True)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_linestyle('-.')

    ax.set_title(title, fontsize=18)
    legend = ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)


def get_style(args):
    return {
            'indep': ('#000000', 'indep.'),
            'sumcov': ('#FF0000', 'sum k w'),
            'varsumvarcov': ('#00FF00', 'q sum q k w'),
        }[args['transform']]


def draw(args):
    loss1_fig, loss1_ax = plt.subplots(1, 1)
    lossk_fig, lossk_ax = plt.subplots(1, 1)
    time_fig, time_ax = plt.subplots(1, 1)

    data = []
    for path in args.pickles:
        data.append(load(path))

    # TODO plot selected users
    # TODO plot per-run debug plots

    max_regret1, max_regretk, max_time = -np.inf, -np.inf, -np.inf
    for loss1_matrix, lossk_matrix, time_matrix, info in data:
        color, label = get_style(info)

        max_iters = info['max_iters']
        x = np.arange(1, max_iters + 1)

        # regret

        loss1_matrix = loss1_matrix.mean(axis=2) # average over all users
        y = np.median(loss1_matrix, axis=0)[:max_iters]
        yerr = np.std(loss1_matrix, axis=0)[:max_iters] \
                      / np.sqrt(loss1_matrix.shape[0])
        max_regret1 = max(max_regret1, y.max())

        loss1_ax.plot(x, y, linewidth=2, color=color, label=label,
                      marker='o', markersize=6)
        loss1_ax.fill_between(x, y - yerr, y + yerr, linewidth=0, color=color,
                              alpha=0.35)

        # query-set regret

        lossk_matrix = lossk_matrix.mean(axis=2) # average over all users
        y = np.median(lossk_matrix, axis=0)[:max_iters]
        yerr = np.std(lossk_matrix, axis=0)[:max_iters] \
                      / np.sqrt(lossk_matrix.shape[0])
        max_regretk = max(max_regretk, y.max())

        lossk_ax.plot(x, y, linewidth=2, color=color, label=label,
                      marker='o', markersize=6)
        lossk_ax.fill_between(x, y - yerr, y + yerr, linewidth=0, color=color,
                              alpha=0.35)

        # cumulative time

        cumtime_matrix = time_matrix.cumsum(axis=1)
        y = np.mean(cumtime_matrix, axis=0)[:max_iters]
        yerr = np.std(cumtime_matrix, axis=0)[:max_iters] \
                   / np.sqrt(time_matrix.shape[0])
        max_time = max(max_time, y.max())

        time_ax.plot(x, y, linewidth=2, color=color, label=label,
                     marker='o', markersize=6)
        time_ax.fill_between(x, y - yerr, y + yerr, linewidth=0, color=color,
                             alpha=0.35)

    loss1_ax.set_ylabel('regret')
    loss1_ax.set_xlim([1, max_iters])
    loss1_ax.set_ylim([0, 1.05 * max_regret1])
    prettify(loss1_ax, max_iters, 'Regret')
    loss1_fig.savefig(args.png_basename + '_loss1.png', bbox_inches='tight',
                      pad_inches=0, dpi=120)

    lossk_ax.set_ylabel('min set-wise regret')
    lossk_ax.set_xlim([1, max_iters])
    lossk_ax.set_ylim([0, 1.05 * max_regretk])
    prettify(lossk_ax, max_iters, 'Query Set Regret')
    lossk_fig.savefig(args.png_basename + '_lossk.png', bbox_inches='tight',
                      pad_inches=0, dpi=120)

    time_ax.set_ylabel('cumulative time (s)')
    time_ax.set_xlim([1, max_iters])
    time_ax.set_ylim([0, 1.05 * max_time])
    prettify(time_ax, max_iters, 'Time')
    time_fig.savefig(args.png_basename + '_time.png', bbox_inches='tight',
                     pad_inches=0, dpi=120)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of result pickles')
    args = parser.parse_args()

    draw(args)
