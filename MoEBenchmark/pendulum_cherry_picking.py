#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
pendulum_cherry_picking.py

A cherry picking example for the pendulum swing up problem.
I recreate those images for the font issue
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
import pyLib.plot as pld
from pyLib.train import MoMNet, modelLoader

import util


def main():
    args = util.get_args()
    cfg, lbl = util.get_label_cfg_by_args(args)
    show_picky_examples(cfg)


def parseX(sol, N, dimx, dimu):
    tX = np.reshape(sol[:N*dimx], (N, dimx))
    return tX, None, None


def show_picky_examples(cfg):
    """I will use manual labels based on final angle"""
    SIZE = 24
    font = pld.getSizeFont(SIZE)
    N, dimx, dimu = 25, 2, 1

    data = np.load(cfg['file_path'])
    x0 = data['x0']
    Sol = data['Sol']
    nData = x0.shape[0]
    # find labels
    thetaf = (Sol[:, 48] + np.pi) / 2 / np.pi
    label = thetaf.astype(int)
    print([np.sum(label == i) for i in range(3)])
    # we use state (1.0, 0.6)
    # first we plot all-black the trajectories
    # find the data point
    onex0 = np.array([2.4, 1.6])
    diff = np.linalg.norm(Sol[:, :dimx] - onex0, axis=1)
    thetaf = np.reshape(Sol[:, 48], (61, 21)).T
    theind = np.argmin(diff)
    thex0 = Sol[theind, :dimx]
    print('x0 is {}'.format(thex0))

    blackName = 'gallery/pen/blackPendulum.pdf'
    if True:
        fig, ax = plt.subplots()
        for i in range(nData):
            row, col = np.unravel_index(i, (21, 61), order='F')
            todo = False
            if row == 0 or row == 20:
                todo = True
            elif col == 0 or col == 60:
                todo = True
            if not todo:
                neighborthetaf = [thetaf[row-1,col], thetaf[row+1, col], thetaf[row, col-1], thetaf[row, col+1]]
                if not np.allclose(neighborthetaf, thetaf[row, col]):
                    todo = True
            if todo:
                tX, _, _ = parseX(Sol[i], N, dimx, dimu)
                ax.plot(tX[:, 0], tX[:, 1], color='k')
        # draw the target
        ax.plot(np.pi, 0, color='r', marker='o', markersize=5)
        ax.plot(-np.pi, 0, color='r', marker='o', markersize=5)
        ax.plot(3*np.pi, 0, color='r', marker='o', markersize=5)
        pld.setTickSize(ax, SIZE)
        ax.set_xlabel(r'$\theta$', fontproperties=font)
        ax.set_ylabel(r'$\omega$', fontproperties=font)
        pld.savefig(fig, blackName)

    colorName = 'gallery/pen/colorPendulum.pdf'
    if True:
        ncluster = 3
        colors = pld.getColorCycle()
        fig, ax = plt.subplots()
        for k in range(ncluster):
            mask =  label ==  k
            x0_ = x0[mask]
            Sol_ = Sol[mask]
            ndata = len(x0_)
            for i in range(ndata):
                # check if it is at boundary
                todo = False
                theta, omega = x0_[i]
                # find with the same theta, if omega is at boundary
                vind = np.where(np.abs(x0_[:, 0] - theta) < 1e-2)[0]
                omegas = x0_[vind, 1]
                if omega < omegas.min() + 1e-4 or omega > omegas.max() - 1e-4:
                    todo = True
                if not todo:
                    # find same omega, check theta
                    vind = np.where(np.abs(x0_[:, 1] - omega) < 1e-2)[0]
                    thetas = x0_[vind, 0]
                    if theta < thetas.min() + 1e-4 or theta > thetas.max() - 1e-4:
                        todo = True
                if todo:
                    tX, _, _ = parseX(Sol_[i], N, dimx, dimu)
                    ax.plot(tX[:, 0], tX[:, 1], color=colors[k])
        # draw the target
        ax.plot(np.pi, 0, color='r', marker='o', markersize=5)
        ax.plot(-np.pi, 0, color='r', marker='o', markersize=5)
        ax.plot(3*np.pi, 0, color='r', marker='o', markersize=5)
        ax.set_xlabel(r'$\theta$', fontproperties=font)
        ax.set_ylabel(r'$\omega$', fontproperties=font)
        pld.setTickSize(ax, SIZE)
        pld.savefig(fig, colorName)

    snnbadName = 'gallery/pen/penSNNbadPred.pdf'
    if True:
        mdl = modelLoader(cfg['snn_path'])
        # find the one closest to one
        fig, ax = plt.subplots()
        tX, _, _ = parseX(Sol[theind], N, dimx, dimu)
        ax.plot(tX[:, 0], tX[:, 1], color='k', label='Optimal')
        # make prediction
        predy = mdl(thex0)
        tX, _, _ = parseX(predy, N, dimx, dimu)
        ax.plot(tX[:, 0], tX[:, 1], color='k', linestyle='--', label='SNN Pred.')
        ax.plot(3*np.pi, 0, color='r', marker='o', markersize=5)
        ax.set_xlabel(r'$\theta$', fontproperties=font)
        ax.set_ylabel(r'$\omega$', fontproperties=font)
        ax.legend(fontsize=SIZE)
        pld.setTickSize(ax, SIZE)
        pld.savefig(fig, snnbadName)

    momgoodName = 'gallery/pen/penMoMgoodPred.pdf'
    lbl_name = 'pca_kmean_label'
    result = util.get_clus_reg_by_dir('models/pen/pca_kmean_label')
    cls, regs = result[5]
    net = MoMNet(cls, regs)
    if True:
        fig, ax = plt.subplots()
        tX, _, _ = parseX(Sol[theind], N, dimx, dimu)
        ax.plot(tX[:, 0], tX[:, 1], color='k', label='Optimal')
        # make prediction
        predy = net.getPredY(thex0)
        tX, _, _ = parseX(predy, N, dimx, dimu)
        ax.plot(tX[:, 0], tX[:, 1], color='k', linestyle='--', label='MoE Pred.')
        ax.plot(3*np.pi, 0, color='r', marker='o', markersize=5)
        ax.set_xlabel(r'$\theta$', fontproperties=font)
        ax.set_ylabel(r'$\omega$', fontproperties=font)
        ax.legend(fontsize=SIZE)
        pld.setTickSize(ax, SIZE)
        pld.savefig(fig, momgoodName)

    plt.show()




if __name__ == '__main__':
    main()
