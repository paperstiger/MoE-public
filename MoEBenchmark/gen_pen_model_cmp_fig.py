#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""

"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
from scipy.interpolate import griddata
import cPickle as pkl

from pyLib.io import getArgs, npload
import pyLib.plot as pld
from pyLib.math import l1loss

import datanames


pld.setGlobalFontSize(14)
cfg = datanames.PENDULUM
KEYS = ['SNN', 'MoE Cost', 'MoE', 'MoE Gate', 'k-Means-3', 'k-Means-5', 'k-Means-10']


def main():
    # theta draws final prediction of theta
    # label draws final labels from each model
    # error shows error on validation set
    # success shows success on rollout
    args = getArgs('theta', 'hist', 'label', 'error', 'success')
    if args.theta:
        data = load_pred_y()
        showThetaf(data)
    if args.label:
        show_labels(args)
    if args.error:
        show_validation_error(args)
    if args.success:
        show_validation_success(args)


def show_validation_error(args):
    """Show the validation error for each model."""
    # load valid data
    valid_data = np.load(cfg['valid_path'])
    x0 = valid_data[cfg['x_name']]
    Sol = valid_data[cfg['y_name']]
    # load model prediction
    data = load_pred_y()
    for key in KEYS:
        pred = data[key]
        loss = l1loss(Sol, pred)
        print(np.mean(loss))


def show_validation_success(args):
    """Show the success rate for each model"""
    with open('data/pen/snn_rollout_result.pkl', 'rb') as f:
        snn_rst = pkl.load(f)
    with open('data/pen/moe_rollout_result.pkl', 'rb') as f:
        moe_rst = pkl.load(f)
    with open('data/pen/mom_rollout_result.pkl', 'rb') as f:
        mom_rst = pkl.load(f)
    with open('data/pen/gate_expert_rollout_result.pkl', 'rb') as f:
        ge_rst = pkl.load(f)
    with open('data/pen/pca_kmean_rollout_result.pkl', 'rb') as f:
        kmean_rst = pkl.load(f)
    k3_rst = kmean_rst['3']
    k5_rst = kmean_rst['5']
    k10_rst = kmean_rst['10']
    snn_ok = np.sum([tmp['status'] for tmp in snn_rst])
    moe_ok = np.sum([tmp['status'] for tmp in moe_rst])
    mom_ok = np.sum([tmp['status'] for tmp in mom_rst])
    ge_ok = np.sum([tmp['status'] for tmp in ge_rst])
    k3_ok = np.sum([tmp['status'] for tmp in k3_rst])
    k5_ok = np.sum([tmp['status'] for tmp in k5_rst])
    k10_ok = np.sum([tmp['status'] for tmp in k10_rst])
    print(snn_ok, moe_ok, mom_ok, ge_ok, k3_ok, k5_ok, k10_ok)


def show_labels(args):
    """Show the labels on a graph"""
    pld.setGlobalFontSize(16)
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    keys = ['MoE Cost', 'MoE Gate', 'k-Means-3', 'k-Means-5']
    titles = ['MoE I', 'MoE II', '$k$-Means-3', '$k$-Means-5']
    moelbl = np.load('data/pen/moe_label.npy')
    momlbl = np.load('data/pen/mom_label.npy')
    gatelbl = np.load('data/pen/gate_expert_label.npy')
    kmeanlbl = np.load('data/pen/pca_kmean_label.npz')
    k3lbl = kmeanlbl['3']
    k5lbl = kmeanlbl['5']

    data = npload(cfg['file_path'], cfg['uniqueid'])
    x0 = data[cfg['x_name']]

    markers = ['s', 'o', 'x', 'd', '*']
    colors = ['b', 'g', 'r', 'c', 'k']
    cm = plt.get_cmap('jet')
    norm = mpl.colors.Normalize(0, 5)

    def show_label_on_axis(ax, x, lbl):
        nlbl = np.amax(lbl) + 1
        ax.imshow(np.reshape(lbl, (61, 21)).T, cmap=cm, origin='lower', norm=norm, extent=[0, 2*np.pi, -2.0, 2.0])
        # for i in range(nlbl):
        #     mask = lbl == i
        #     ax.scatter(x[mask, 0], x[mask, 1], s=3, marker=markers[i], color=colors[i])

    show_label_on_axis(ax[0][0], x0, moelbl)
    ax[0][0].set_title(titles[0])
    show_label_on_axis(ax[0][1], x0, gatelbl)
    ax[0][1].set_title(titles[1])
    show_label_on_axis(ax[1][0], x0, k3lbl)
    ax[1][0].set_title(titles[2])
    show_label_on_axis(ax[1][1], x0, k5lbl)
    ax[1][1].set_title(titles[3])
    ax[1][0].set_xlabel(r'$\theta$')
    ax[1][1].set_xlabel(r'$\theta$')
    ax[0][0].set_xticklabels([])
    ax[0][1].set_xticklabels([])
    ax[0][0].set_ylabel(r'$\omega$')
    ax[1][0].set_ylabel(r'$\omega$')
    ax[0][1].set_yticklabels([])
    ax[1][1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig('gallery/pen/pen_label_assign.pdf')
    plt.show()


def load_pred_y():
    """Load for each model, the model prediction of trajectory.

    I will load snn, moe, mom, 3, 5, 10
    """
    snn = np.load('data/pen/snn_validation_predict.npy')
    moe = np.load('data/pen/moe_valid_data.npz')['y']
    mom = np.load('data/pen/mom_valid_data.npz')['y']
    gate = np.load('data/pen/gate_expert_valid_data.npz')['y']
    kmean = np.load('data/pen/pca_kmean_label_validation_predict.npz')
    k3 = kmean['3']
    k5 = kmean['5']
    k10 = kmean['10']
    return OrderedDict(**{'SNN': snn, 'MoE Cost': moe, 'MoE': mom, 'MoE Gate': gate, 'k-Means-3': k3, 'k-Means-5': k5, 'k-Means-10': k10})


def showThetaf(data):
    """Show thetaf of each model prediction"""
    pld.setGlobalFontSize(16)
    # load validation set
    valid_data = np.load(cfg['valid_path'])
    x0 = valid_data[cfg['x_name']]
    Sol = valid_data[cfg['y_name']]

    usetitle = True
    cm = plt.get_cmap('jet')
    xi = np.linspace(0, 2*np.pi,100)
    yi = np.linspace(-2.0,2.0,50)


    # calculate those angle error
    vthetaf = []
    keys = ['SNN', 'MoE Cost', 'MoE Gate', 'k-Means-3', 'k-Means-5', 'k-Means-10']
    titles = ['SNN', 'MoE I', 'MoE II', '$k$-Means-3', '$k$-Means-5', '$k$-Means-10']
    for key in keys:
        vthetaf.append(data[key][:, 48] - Sol[:, 48])

    # get figures for this
    nmdl = len(vthetaf)
    num = int(np.ceil(np.sqrt(nmdl)))
    nrow = nmdl // num
    if nmdl % num != 0:
        nrow += 1
    fig, axes = plt.subplots(nrow, num, figsize=(7, 3.5))
    # get val
    minval = np.amin([np.amin(thetaf) for thetaf in vthetaf])
    maxval = np.amax([np.amax(thetaf) for thetaf in vthetaf])
    norm = mpl.colors.Normalize(-1, 7)
    levels = np.linspace(minval, maxval, 10)
    for i in range(nmdl):
        row = i // num
        col = i % num
        if nrow > 1:
            ax = axes[row][col]
        else:
            ax = axes[col]
        zi = griddata((x0[:, 0], x0[:, 1]), np.abs(vthetaf[i]), (xi[None,:], yi[:,None]), method='nearest')
        cs = ax.imshow(np.reshape(zi, (50, 100)), cmap=cm, origin='lower', norm=norm, extent=[0, 2*np.pi, -2.0, 2.0])
        # cs = ax.tricontourf(x0[:, 0], x0[:, 1], np.abs(vthetaf[i]), cmap=cm, origin='lower', norm=norm)
        if row == nrow - 1:
            ax.set_xlabel(r'$\theta$')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(r'$\omega$')
        else:
            ax.set_yticklabels([])
        if usetitle:
            ax.set_title(titles[i])
    divider = make_axes_locatable(ax)
    fig.tight_layout()
    fig.colorbar(cs, ax=axes.ravel().tolist())
    fig.savefig('gallery/pen/penPredictionThetaf.pdf')
    plt.show()


if __name__ == '__main__':
    main()
