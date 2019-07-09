#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
cluster_eval.py

A systematic evaluation of cluster assignment for each dataset.
Basically, I will propose several metric and evaluates them for each cluster.
I will report those numbers and possibly with tracking results afterwards.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from pyLib.io import getArgs
from pyLib.math import getStandardData, getNNIndex, getAffinityXY, getPCA
import pyLib.plot as pld

import datanames
from util import construct_distance_graph, get_vio_dst_fun, get_label_cfg_by_args

marker_choice = ['o', 'v', 's', 'p', '*']


def main():
    # pen, car, drone specifies which dataset to use
    # the next five clustering approach specifies which data to load
    args = getArgs('pen', 'car', 'drone', 'kmean', 'pcakmean', 'speuclid', 'spdydx', 'spvio', 'neighbor10', 'umap', 'pca')
    cfg, lbl_name = get_label_cfg_by_args(args)
    if args.umap:  # I am done using this
        draw_umap_label(cfg, lbl_name)
    elif args.pca:
        draw_pca_label(cfg, lbl_name)
    else:
        check_cluster_label(cfg, lbl_name, args.neighbor)


def draw_pca_label(cfg, lbl_name):
    y = np.load(cfg['file_path'])[cfg['y_name']]
    prj = getPCA(y, 2)
    umap = prj  # fake to save code
    uid = cfg['uniqueid']
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    keys = label_data.keys()
    keys.sort(key=lambda x: int(x))
    nkey = len(keys)
    if nkey >= 10:
        cmap = plt.get_cmap('jet')

    fig, ax = pld.subplots(nkey)
    for i, key in enumerate(keys):
        label = label_data[key]
        n_label = int(key)
        print(n_label)
        for j in range(n_label):
            mask = np.where(j == label)[0]
            print(mask.shape[0])
            if mask.shape[0] > 1000:
                choice = np.random.choice(mask.shape[0], 1000, replace=False)
                mask = mask[choice]
            if nkey < 10:
                ax[i].scatter(umap[mask, 0], umap[mask, 1], s=2, marker=marker_choice[j % 5])
            else:
                color = cmap(float(j) / n_label)
                ax[i].scatter(umap[mask, 0], umap[mask, 1], s=10, marker=marker_choice[j % 5], color=color)
    fig.tight_layout()
    fig.savefig('gallery/%s/pca_%s.pdf' % (uid, lbl_name))
    plt.show()


def draw_umap_label(cfg, lbl_name):
    """Use umap to show something.

    :param cfg: a dictionary for problem specification
    :param lbl_name: str, file name where label is stored
    """
    umap = np.load(cfg['umap_path'])
    uid = cfg['uniqueid']
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    keys = label_data.keys()
    keys.sort(key=lambda x: int(x))
    nkey = len(keys)
    if nkey >= 10:
        cmap = plt.get_cmap('jet')

    fig, ax = pld.subplots(nkey)
    for i, key in enumerate(keys):
        label = label_data[key]
        n_label = int(key)
        for j in range(n_label):
            mask = np.where(j == label)[0]
            if mask.shape[0] > 1000:
                choice = np.random.choice(mask.shape[0], 1000, replace=False)
                mask = mask[choice]
            if nkey < 10:
                ax[i].scatter(umap[mask, 0], umap[mask, 1], s=2, marker=marker_choice[j % 5])
            else:
                color = cmap(float(j) / n_label)
                ax[i].scatter(umap[mask, 0], umap[mask, 1], s=2, marker=marker_choice[j % 5], color=color)
    fig.tight_layout()
    fig.savefig('gallery/%s/umap_%s.pdf' % (uid, lbl_name))
    plt.show()


def check_cluster_label(cfg, lbl_name, neighbor=10):
    """Check the label assigned by kmeans algorithm"""
    uid = cfg['uniqueid']
    # load the vio_dist_function
    vio_dist = get_vio_dst_fun(cfg)
    # load data
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    # load label
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    keys = label_data.keys()
    keys.sort(key=lambda x: int(x))
    # create a list of dicts
    out_dct = OrderedDict()
    for key in keys:
        label = label_data[key]
        dcti = evaluate_one_cluster_assignment(x, y, int(key), label, vio_dist, neighbor)
        # lst_max_val.append([np.amax(dcti['dy']), np.amax(dcti['dydx']), np.amax(dcti['vio'])])
        out_dct[key] = dcti
        # lst_dct.append(dcti)
    np.savez('data/%s/%s_metric_values.npz' % (uid, lbl_name), **out_dct)
    # for each metric (col of max_val_mat), calculate bin for each k
    for key in ['dy', 'dydx', 'vio']:
        lst_val = [out_dct[key_][key] for key_ in keys]
        fig, ax = plt.subplots()
        ax.hist(lst_val, bins=20, label=keys)
        bot, top = ax.get_ylim()
        ax.set_yscale('log', nonposy='clip')
        ax.set_ylim(1, top)
        ax.set_ylabel('Count')
        if key == 'dy':
            ax.set_xlabel(r'$\Delta y$')
        elif key == 'dydx':
            ax.set_xlabel(r'$\Delta y / \Delta x$')
        else:
            ax.set_xlabel(r'Constr. Violation')
        ax.legend()
        fig.savefig('gallery/%s/%s_hist_metric_%s.pdf' % (uid, lbl_name, key))
    plt.show()


def evaluate_one_cluster_assignment(x, y, n_cluster, label, vio_dist_fun, n_neighbor=10):
    """Given a dataset of x, y; label assignment; function for violation distance

    Report some results to evaluate the cluster assignment strategy.
    I shall report both dy/dx and constraint violation within each cluster (maybe also inter-cluster?)
    """
    cmap = plt.get_cmap('jet')
    x_scaled = getStandardData(x, cols=True)
    n_data = x.shape[0]
    n_dist_vec_size = n_data * n_neighbor
    disty_vector = np.zeros(n_dist_vec_size)
    dy_over_dx_vector = np.zeros(n_dist_vec_size)
    vio_dist_fun_vector = np.zeros(n_dist_vec_size)
    # loop over them
    idx = 0
    for i in range(n_cluster):
        mask = label == i
        mx, my = x[mask], y[mask]
        mxs = x_scaled[mask]
        n_sub_data = mx.shape[0]
        print(n_sub_data)
        use_neighbor = n_neighbor
        if n_sub_data < n_neighbor:
            use_neighbor = n_sub_data - 1
        if use_neighbor <= 0:
            continue
        idxf = n_sub_data * use_neighbor + idx
        # calculate those metrics
        nn_ind = getNNIndex(mxs, None, use_neighbor + 1, scale=False)  # since mxs is already scaled
        distx, disty, row, col = getAffinityXY(mx, my, nn_ind, 2)
        dy_over_dx_vector[idx: idxf] = disty / distx
        disty_vector[idx: idxf] = disty
        if vio_dist_fun is not None:
            vio_dist, vio_row, vio_col = construct_distance_graph(mx, my, nn_ind, vio_dist_fun)
            vio_dist_fun_vector[idx: idxf] = vio_dist
        idx = idxf  # update index
    # return a dictionary of those contents
    return {'dy': disty_vector, 'dydx': dy_over_dx_vector, 'vio': vio_dist_fun_vector}


def show_one_cluster_on_umap(n_cluster, label, prj_y, svnm=None, show=False):
    """Show one cluster on a umap"""
    cmap = plt.get_cmap('jet')
    fig, ax = plt.subplots()
    for i in range(n_cluster):
        color = cmap(float(i) / n_cluster)
        mask = label == i
        ax.scatter(prj_y[mask, 0], prj_y[mask, 1], color=color, s=2)
    if svnm is not None:
        fig.savefig(svnm)
    if show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
