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


def pad(array, length):
    full = np.zeros((length,) + array.shape[1:], dtype=array.dtype)
    full[:len(array)] = array
    return full


def load(path):
    data = musm.load(path)
    max_iters = data['args']['max_iters']
    loss_matrix, time_matrix = [], []
    for trace in data['traces']:
        trace = np.array(trace)
        loss_matrix.append(pad(trace[:,:-2], max_iters))
        time_matrix.append(pad(trace[:,-1], max_iters))
    info = {**{'method': 'musm'}, **data['args']}
    return np.array(loss_matrix), np.array(time_matrix), info


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


def draw(args):
    plt.style.use('ggplot')

    data = []
    for path in args.pickles:
        data.append(load(path))

    loss_fig, loss_ax = plt.subplots(1, 1)
    time_fig, time_ax = plt.subplots(1, 1)

    max_regret, max_time = -np.inf, -np.inf
    for loss_matrix, time_matrix, info in data:
        loss_matrix = loss_matrix.mean(axis=2) # average over all users

        max_iters = info['max_iters']
        x = np.arange(1, max_iters + 1)

        y = np.median(loss_matrix, axis=0)[:max_iters]
        yerr = np.std(loss_matrix, axis=0)[:max_iters] \
                   / np.sqrt(loss_matrix.shape[0])
        max_regret = max(max_regret, y.max())

        loss_ax.plot(x, y, linewidth=2, color='#000000', label='label',
                     marker='o', markersize=6)
        loss_ax.fill_between(x, y - yerr, y + yerr, linewidth=0, color='#000000',
                             alpha=0.35)

        cumtime_matrix = time_matrix.cumsum(axis=1)
        y = np.mean(cumtime_matrix, axis=0)[:max_iters]
        yerr = np.std(cumtime_matrix, axis=0)[:max_iters] \
                   / np.sqrt(time_matrix.shape[0])
        max_time = max(max_time, y.max())

        time_ax.plot(x, y, linewidth=2, color='#000000', label='label',
                     marker='o', markersize=6)
        time_ax.fill_between(x, y - yerr, y + yerr, linewidth=0, color='#000000',
                             alpha=0.35)

    loss_ax.set_ylabel('regret')
    loss_ax.set_xlim([1, max_iters])
    loss_ax.set_ylim([0, 1.05 * max_regret])
    prettify(loss_ax, max_iters, "you'll regret this!")
    loss_fig.savefig(args.png_basename + '_loss.png', bbox_inches='tight',
                     pad_inches=0, dpi=120)

    time_ax.set_ylabel('cumulative time (s)')
    time_ax.set_xlim([1, max_iters])
    time_ax.set_ylim([0, 1.05 * max_time])
    prettify(time_ax, max_iters, "you'll time this!")
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
