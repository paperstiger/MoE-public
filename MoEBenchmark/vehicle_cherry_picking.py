#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
vehicle_cherry_picking.py

For vehicle benchmark problem, show examples that SNN fails badly.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs, npload
from pyLib.train import modelLoader, MoMNet
from pyLib.math import Query
import pyLib.plot as pld

import util


pld.setGlobalFontSize(16)


def parseX(sol, *args):
    """Parse a sol into plotable piece"""
    tX = np.reshape(sol[:25*4], (25, 4))
    tU = np.reshape(sol[100: 100 + 2*24], (24, 2))
    tf = sol[-1]
    return tX, tU, tf


def main():
    args = util.get_args('show')
    cfg, lbl_name = util.get_label_cfg_by_args(args)
    show_picky_states(cfg, lbl_name, args)


def show_picky_states(cfg, lbl_name, args):
    """Select a few states and draw predictions."""
    uid = cfg['uniqueid']
    lbl_name = 'pca_kmean_label'
    # load all training data and validation data
    data = npload(cfg['file_path'], uid)
    xname, yname = cfg['x_name'], cfg['y_name']
    datax, datay = data[xname], data[yname]
    # create a query instance
    query = Query(datax, scale=True)
    vdata = np.load(cfg['valid_path'])
    vx, vy = vdata[xname], vdata[yname]
    # snn model
    snn_fun = modelLoader(cfg['snn_path'])
    # moe model
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    cls, regs = result[10]  # let me try this one
    net = MoMNet(cls, regs)
    # load cluster labels
    lbl_data_dct = np.load('data/%s/%s.npz' % (uid, lbl_name))
    label = lbl_data_dct['10']

    # eval snn on validation set and extract the one with largest prediction error
    pred_vy = snn_fun(vx)
    diff_vy = pred_vy - vy
    error_y = np.linalg.norm(diff_vy, axis=1)
    error_order = np.argsort(error_y)
    for i in range(7, 20):
        vx_idx = error_order[-1 - i]
        bad_x0 = vx[vx_idx]
        bad_sol = vy[vx_idx]
        snn_pred = pred_vy[vx_idx]
        moe_pred = net.getPredY(bad_x0)

        predX, _, _ = parseX(snn_pred)
        realX, _, _ = parseX(bad_sol)
        predXMoE, _, _ = parseX(moe_pred)

        # get neighbors
        index = query.getIndex(bad_x0)
        print('index ', index, 'label ', label[index])
        # draw them
        fig, axes = plt.subplots(1, 2)
        shown_cluster = []
        for ind in index:
            nnX, _, _ = parseX(datay[ind])
            if label[ind] not in shown_cluster:
                axes[1].plot(nnX[:, 0], nnX[:, 1], color='C%d' % label[ind], label='Cluster %d' % label[ind])
                shown_cluster.append(label[ind])
            else:
                axes[1].plot(nnX[:, 0], nnX[:, 1], color='C%d' % label[ind])
        axes[0].plot(predX[:, 0], predX[:, 1], color='#ff7f0e', linewidth=2, ls='--', label='SNN')
        axes[0].plot(predXMoE[:, 0], predXMoE[:, 1], color='g', linewidth=2, ls='--', label='MoE')
        axes[0].plot(realX[:, 0], realX[:, 1], color='k', linewidth=2, label='Opt.')
        finalAgl = predX[-1, 2]
        direc = [1*np.sin(finalAgl), 1*np.cos(finalAgl)]
        xf = predX[-1]
        for i in range(2):
            ax = axes[i]
            if i == 0:
                ax.arrow(xf[0], xf[1], direc[0], direc[1], color='#ff7f0e', linewidth=2, width=0.1)
            finalAgl = predXMoE[-1, 2]
            direc = [1*np.sin(finalAgl), 1*np.cos(finalAgl)]
            xf = predXMoE[-1]
            ax.arrow(xf[0], xf[1], direc[0], direc[1], color='g', linewidth=2, width=0.1)
            ax.scatter(0, 0, s=50, color='r')
            ax.annotate('Goal', (0, 0), xytext=(0.2, 0.2), textcoords='data')
            ax.scatter(bad_x0[0], bad_x0[1], s=50, color='k', marker='*')
            if i == 0:
                ax.annotate('Start', (bad_x0[0], bad_x0[1]), xytext=(-1 + bad_x0[0], -0.8 + bad_x0[1]), textcoords='data')
            else:
                ax.annotate('Start', (bad_x0[0], bad_x0[1]), xytext=(bad_x0[0], 0.3 + bad_x0[1]), textcoords='data')
            ax.set_xlabel(r'$x$')
            ax.axis('equal')
            if i == 0:
                xlim = ax.get_xlim()
                ax.set_ylabel(r'$y$')
            if i == 0:
                ax.legend()
            else:
                ax.legend(loc=4)
            if i == 0:
                ax.set_xlim(-2.5, xlim[1] + 1)
            else:
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0] - 1, xlim[1] + 1.5)
        fig.tight_layout()
        fig.savefig('gallery/car/car_snn_vs_moe_traj.pdf')
        plt.show()


if __name__ == '__main__':
    main()
