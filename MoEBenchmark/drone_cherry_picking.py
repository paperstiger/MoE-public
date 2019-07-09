#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
drone_cherry_picking.py

Draw a few figures on the drone example.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
from pyLib.train import modelLoader, MoMNet
import pyLib.plot as pld

import util


pld.setGlobalFontSize(14)


def parseX(sol):
    """Parse a sol into plotable piece"""
    tX = np.reshape(sol[:20*12], (20, 12))
    tU = np.reshape(sol[240: 240 + 4*19], (19, 4))
    tf = sol[-1]
    return tX, tU, tf


def main():
    args = util.get_args('show')
    cfg, lbl_name = util.get_label_cfg_by_args(args)
    show_picky_states(cfg, lbl_name, args)


def show_picky_states(cfg, lbl_name, args):
    """Show a few states and their predictions"""
    # play with initial states
    use_obs = np.array([0, 4, 4, 3])
    use_x0 = np.array([[0, 8, 8], [0, 8, 9]])
    n_x0 = use_x0.shape[0]

    uid = cfg['uniqueid']
    lbl_name = 'pca_kmean_label'
    # load functions
    snn_fun = modelLoader(cfg['snn_path'])
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    cls, regs = result[20]  # let me try this one
    net = MoMNet(cls, regs)

    # load optimal solutions
    sol = np.load('data/droneone/the_two_sol.npy')

    # create figure
    fig, ax = pld.get3dAxis()
    pld.addSphere(ax, *use_obs)
    legend_sets = []
    for i in range(n_x0):
        tmp_set = []
        x0 = np.concatenate((use_x0[i], use_obs))
        y0 = snn_fun(x0)
        predy = net.getPredY(x0)
        tX, _, _ = parseX(y0)

        hi, = ax.plot(tX[:, 0], tX[:, 1], tX[:, 2], ls=':', color='C%d' % i, label='SNN %d' % (i + 1))
        tXmoe, _, _ = parseX(predy)
        hj, = ax.plot(tXmoe[:, 0], tXmoe[:, 1], tXmoe[:, 2], ls='--', color='C%d' % i, label='MoE %d' % (i + 1))

        true_sol = sol[i]
        tXt, _, _ = parseX(true_sol)
        hk, = ax.plot(tXt[:, 0], tXt[:, 1], tXt[:, 2], color='C%d' % i, label='Opt. %d' % (i + 1))
        legend_sets.append([hi, hj, hk])

        ax.scatter(*use_x0[i], marker='*', color='C%d' % i)  #, label='Start %d' % i)
        ax.text(use_x0[i][0], use_x0[i][1], use_x0[i][2], 'Start%d' % (i + 1))
    ax.scatter(0, 0, 0, marker='o', color='r')  # label='Goal'
    ax.text(0, 0, 0, "Goal")
    lgd = ax.legend(handles=legend_sets[0], loc=4, bbox_to_anchor=(0.8, 0.1))
    ax.add_artist(lgd)
    ax.legend(handles=legend_sets[1], loc=2, bbox_to_anchor=(0.15, 0.85))
    # ax.legend()
    ax.set_xlabel(r'$x$(m)')
    ax.set_ylabel(r'$y$(m)')
    ax.set_zlabel(r'$z$(m)')
    ax.view_init(elev=11, azim=10)
    fig.tight_layout()
    fig.savefig('gallery/droneone/moe_vs_snn_example.pdf')
    plt.show()


if __name__ == '__main__':
    main()
