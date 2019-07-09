#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
show_dataset.py

To write the paper, I need figures and numbers of the dataset.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
import pyLib.plot as pld
from pyLib.math import getPCA
import util


pld.setGlobalFontSize(mode='half')


def main():
    # num shows dataset size
    # drawpca draws pca results for datasets
    # umap draws scatter plot of umap projection
    # penscatter draws scatter plot on pendulum dataset, we draw scatter plot of x0
    args = util.get_args('num', 'drawpca', 'umap', 'penscatter')
    cfg, lbl = util.get_label_cfg_by_args(args)
    if args.num:
        show_dataset_size(cfg)
    if args.drawpca:
        drawPCA(cfg)
    if args.umap:
        drawUMap(cfg)
    if args.penscatter:
        drawPenScatter(cfg)


def drawPenScatter(cfg):
    uid = cfg['uniqueid']
    assert uid == 'pen'
    data = np.load(cfg['file_path'])
    x0 = data[cfg['x_name']]
    # load cluster file
    clus_data = np.load('data/%s/pca_kmean_label.npz' % uid)
    print(clus_data.keys())
    label = clus_data['5']
    print(label.shape)
    fig, ax = plt.subplots()
    for i in range(5):
        mask = label == i
        ax.scatter(x0[mask, 0], x0[mask, 1], s=10)
    plt.show()


def drawUMap(cfg):
    uid = cfg['uniqueid']
    if 'umap_path' in cfg:
        umap = np.load(cfg['umap_path'])
    else:
        data = np.load(cfg['file_path'])
        y = data['y_name']
    n_data = umap.shape[0]
    prj = umap
    fig, ax = plt.subplots()
    if n_data > 10000:
        ind = np.random.choice(n_data, 10000, replace=False)
        prj = prj[ind]
    ax.scatter(prj[:, 0], prj[:, 1], s=2)
    fig.tight_layout()
    fig.savefig('gallery/%s/umap_projection_two.pdf' % uid)
    plt.show()


def drawPCA(cfg):
    uid = cfg['uniqueid']
    a = np.load(cfg['file_path'])
    print(a.keys())
    y = a[cfg['y_name']]
    n_data = y.shape[0]
    useN = 10000
    print(y.shape)
    prj = getPCA(y, 2)
    fig, ax = plt.subplots()
    if n_data < 10000:
        pld.plot(prj, ax=ax, scatter=True, use_num=n_data, s=2)
    else:
        pld.plot(prj, ax=ax, scatter=True, use_num=useN, s=2)
    ax.set_xlabel('1st Component')
    ax.set_ylabel('2nd Component')
    fig.tight_layout()
    fig.savefig('gallery/%s/pca_projection_two.pdf' % uid)
    plt.show()


def show_dataset_size(cfg):
    a = np.load(cfg['file_path'])
    print(a.keys())
    x = a[cfg['x_name']]
    print(x.shape)
    if 'flag' in a.keys():
        print(np.sum(a['flag']))
    vld = np.load(cfg['valid_path'])
    print(vld.keys())
    if 'valid_x_name' in vld.keys():
        print(vld[cfg['valid_x_name']].shape)
    else:
        print(vld[cfg['x_name']].shape)
    if 'flag' in vld.keys():
        print(np.sum(vld['flag']))


if __name__ == '__main__':
    main()
