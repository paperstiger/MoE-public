#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
draw_rollout_result_figures.py

Draw for each problem the rollout results on validation set.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
import pyLib.plot as pld


# pld.setGlobalFontSize(14, 'half')
pld.setGlobalFontSize(14)


def main():
    fig, ax = plt.subplots(2, 2)
    lstk = [[3, 5, 10, 20], [5, 10, 20, 40], [5, 10, 20, 40, 80], [5, 10, 20, 40, 80]]
    snn = [0.863, 0.673, 0.119, 0.314]
    lstval = [[0.969, 0.999, 1.000, 1.000], [0.917, 0.999, 0.991, 0.991],
            [0.111, 0.0905, 0.0563, 0.0593, 0.0619], [0.302, 0.213, 0.158, 0.134, 0.145]]
    xlabels = [r'\# Clusters' + '\n' + name for name in ['(a) Pendulum', '(b) Vehicle', '(c) Drone-One-Obs', '(d) Drone-Two-Obs']]
    draw_one_cmp(ax[0][0], snn[0], lstk[0], lstval[0], xlabels[0], 'Success Rate')
    draw_one_cmp(ax[0][1], snn[1], lstk[1], lstval[1], xlabels[1], 'Success Rate')
    draw_one_cmp(ax[1][0], snn[2], lstk[2], lstval[2], xlabels[2], 'Constraint Violation')
    draw_one_cmp(ax[1][1], snn[3], lstk[3], lstval[3], xlabels[3], 'Constraint Violation')
    ax[1][0].set_ylim(0, ax[1][0].get_ylim()[1])
    ax[1][1].set_ylim(0, ax[1][1].get_ylim()[1])
    ax[0][0].set_ylim(0, 1.05)
    ax[0][1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig('gallery/moe_snn_number_cmp.pdf')
    plt.show()


def draw_one_cmp(ax, snn, lstk, lstval, xname, yname):
    ax.axhline(snn, ls='--', label='SNN')
    num_k = len(lstk)
    ax.plot(range(num_k), lstval, marker='o', label='MoE')
    if xname is None:
        ax.set_xlabel('Cluster Number\n(a)Pendulum')
    else:
        ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    # set tick
    ax.set_xticks(range(num_k))
    ax.set_xticklabels(map(str, lstk))
    ax.legend()


if __name__ == '__main__':
    main()
