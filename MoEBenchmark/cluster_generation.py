#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
cluster_generation.py

In this script, I will explore several approaches on generation of cluster files for each problem.
The output will be simply a npz file with names containing information on approach and # clusters.
The format of cluster is simply a ndarray of integers.

Note that this file only deals with k-means type clustering and is solely using trajectory.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyLib.io import getArgs, sharedmemory as sm, load_source, ddctParse
from pyLib.math import getStandardData, Query

import datanames
import util


def main():
    # kmean means we directly run kmeans on solution space with scaling
    # pcakmean means we run kmeans after running pca
    # speuclid means we do spectral clustering with euclidean distance (might be sparse)
    # spdydx means we do spectral clustering with dy/dx as distance
    # spvio means spectral clustering with constraint violation as distance
    # pen means we handle the pendulum dataset
    # car means we deal with the vehicle dataset
    # drone means we handle the drone obstacle problem
    # dtwo means drone with two obstacles
    # append means we append new k to existing data
    # neighbor is an integer of number of neighbors used for constructing sparse graph
    args = util.get_args('append', 'neighbor10')
    # args = getArgs('kmean', 'pcakmean', 'speuclid', 'spdydx', 'spvio', 'pen', 'car', 'drone', 'dtwo', 'append', 'neighbor10')
    if args.pen:
        cfg = datanames.PENDULUM
        k = [3, 5, 10, 20]
        target_dir = 'data/pen'
    elif args.car:
        cfg = datanames.VEHICLE
        k = [5, 10, 20, 40]
        target_dir = 'data/car'
    elif args.drone:
        cfg = datanames.DRONE
        k = [5, 10, 20, 40, 80, 160]
        target_dir = 'data/drone'
    elif args.dtwo:
        cfg = datanames.DRONETWO
        k = [5, 10, 20, 40, 80]
        target_dir = 'data/dronetwo'
    elif args.done:
        cfg = datanames.DRONEONE
        k = [5, 10, 20, 40, 80]
        target_dir = 'data/droneone'
    else:
        print('You must choose from existing datasets')
        raise SystemExit
    if args.kmean:
        generate_kmean_label(cfg, k, target_dir, args.append)
    if args.pcakmean:
        generate_pca_kmean_label(cfg, k, target_dir, args.append)
    try:
        if args.speuclid or args.spdydx or args.spvio:
            generate_spectral_label(cfg, k, target_dir, args.neighbor, args)
    except:
        pass


def generate_spectral_label(cfg, k, target_dir, n_neighbor, args):
    """Use spectral clustering to generate labels. The user can specify distance metric in args

    Parameters
    ----------
    cfg: dict, specifying data path, names we care
    k: list of int, choices of number of clusters
    target_dir: directory for storing the data
    n_neighbor: int, number of neighbors used for graph construction
    args: arguments, it provides choice of several options
        args.append means we append new number of clusters to existing problem
        args.pca means we first use PCA to perform dimensionality reduction
        args.speuclid means we use euclidean distance in y
        args.spdydx means we use dy/dx as distance metric
        args.spvio means we use constraint violation as distance metric
    """
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    query = Query(x, None, n_neighbor + 1, scale=True)
    x_scaled = query.A  # this is scaled data
    nn_ind = query.getIndex(x)  # Is this why I was wrong? Is pyflann still working fine?
    n_data = x.shape[0]
    # build sparse graph based on neighboring distances, I shall use a distance function for evaluation
    if args.speuclid:
        def dst_fun(x0, y0, x1, y1):
            return np.linalg.norm(y0 - y1)
        out_fnm = os.path.join(target_dir, 'sp_euclid_label.npz')
    if args.spdydx:
        def dst_fun(x0, y0, x1, y1):
            return np.linalg.norm(y0 - y1) / np.linalg.norm(x0 - x1)
        out_fnm = os.path.join(target_dir, 'sp_dydx_label.npz')
    if args.spvio:
        out_fnm = os.path.join(target_dir, 'sp_vio_label.npz')
        sys.path.insert(0, cfg['script_path'])
        import libserver
        if args.pen:
            libserver.init(cfg['cfg_path'])
            def dst_fun(x0, y0, x1, y1):
                xmid = (x0 + x1) / 2
                ymid = (y0 + y1) / 2
                c = libserver.eval(ymid.astype(np.float64))
                return np.linalg.norm(c[1:])
        if args.car:
            solver = libserver.pysolver()
            solver.initfnm(cfg['cfg_path'])
            def dst_fun(x0, y0, x1, y1):
                xmid = (x0 + x1) / 2
                ymid = (y0 + y1) / 2
                c = solver.constrEval(ymid.astype(np.float64))
                return np.linalg.norm(c[1:])
        if args.drone:
            solver = libserver.pysolver()
            solver.initfnm(cfg['cfg_path'])
            def dst_fun(x0, y0, x1, y1):
                xmid = (x0 + x1) / 2  # this is not used, maybe not good but who knows
                ymid = (y0 + y1) / 2
                solver.updateObstacle(xmid[3:])
                c = solver.constrEval(ymid.astype(np.float64))
                return np.linalg.norm(c[1:])
    dist, row, col = construct_distance_graph(x, y, nn_ind, dst_fun, rm_col_one=True)
    print('distance matrix construction finished')
    aff_mat = sp.coo_matrix((dist, (row, col)), shape=(n_data, n_data))
    # prepare for output
    if args.append and os.path.exists(out_fnm):
        result = ddctParse(out_fnm)
    else:
        result = {}
    # perform spectral clustering
    for k_ in k:
        print('run spectral clustering with %d' % k_)
        sc = SpectralClustering(k_, eigen_solver='amg', affinity='precomputed', assign_labels='discretize', n_jobs=-1)
        sc.fit(aff_mat)
        label = sc.labels_
        result['%d' % k_] = label
    np.savez(out_fnm, **result)


@numba.jit
def construct_distance_graph(x, y, nn_ind, dst_fun, rm_col_one=True):
    """Construct a distance graph based on neighbors in x and custom distance function.

    Parameters
    ----------
    x: ndarray, the features
    y: ndarray, the response
    nn_ind: ndarray, int, index of nearest neighbors
    dst_fun: callable, a distance function of type f(x, y, x', y')
    rm_col_one: bool, if we ignore first column of nn_ind
    Returns
    -------
    weight, row, col: essential components for a sparse distance matrix
    """
    col_nn_ind = nn_ind.shape[1]
    n_data = x.shape[0]
    if rm_col_one:
        n_neighbor = col_nn_ind - 1
        start_idx = 1
    else:
        n_neighbor = col_nn_ind
        start_idx = 0
    # construct row, col, dist
    nnz = n_data * n_neighbor
    row = np.zeros(nnz, dtype=int)
    col = np.zeros(nnz, dtype=int)
    dist = np.zeros(nnz)
    # loop over what we want to calculate
    idx = 0
    for i in range(n_data):
        row[idx: idx + n_neighbor] = i
        col[idx: idx + n_neighbor] = nn_ind[i, start_idx: col_nn_ind]
        for j in range(start_idx, col_nn_ind):
            dist[idx] = dst_fun(x[i], y[i], x[nn_ind[i, j]], y[nn_ind[i, j]])
            idx += 1
    return dist, row, col


def generate_kmean_label(cfg, k, target_dir, append=False):
    """Generate kmeans labels

    Parameters
    ----------
    cfg: dict, specifying data path, names we care
    k: list of int, choices of number of clusters
    target_dir: directory for storing the data
    append: bool, if we append data to existing file (if it exists)
    """
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    n_data = x.shape[0]
    # prepare for output
    out_fnm = os.path.join(target_dir, 'kmean_label.npz')
    if append and os.path.exists(out_fnm):
        result = ddctParse(out_fnm)
    else:
        result = {}
    for k_ in k:
        if n_data < 10000:
            cluster = KMeans(k_)
        else:
            cluster = MiniBatchKMeans(k_, batch_size=1000)
        best_inertia = 1e10
        for i in range(3):
            cluster.fit(y_scaled)
            if cluster.inertia_ < best_inertia:
                best_inertia = cluster.inertia_
                best_label = cluster.labels_.copy()  # so not a reference
        print('inertia is %f' % best_inertia)
        result['%d' % k_] = best_label
    np.savez(out_fnm, **result)


def generate_pca_kmean_label(cfg, k, target_dir, append=False):
    """Generate labels by running kmeans after PCA

    see :ref:`generate_kmean_label` for arguments
    """
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    print(x.shape)
    scaler = StandardScaler()
    y_scale = scaler.fit_transform(y)
    n_data = x.shape[0]
    # perform pca for dim reduction
    pca = PCA(0.9, svd_solver='full')
    y_tran = pca.fit_transform(y_scale)
    print(pca.n_components_)
    print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))
    # prepare for output
    out_fnm = os.path.join(target_dir, 'pca_kmean_label.npz')
    if append and os.path.exists(out_fnm):
        result = ddctParse(out_fnm)
    else:
        result = {}
    for k_ in k:
        if n_data < 10000:
            cluster = KMeans(k_)
        else:
            cluster = MiniBatchKMeans(k_, batch_size=1000)
        best_inertia = 1e10
        for i in range(3):
            cluster.fit(y_tran)
            if cluster.inertia_ < best_inertia:
                best_inertia = cluster.inertia_
                best_label = cluster.labels_.copy()  # so not a reference
        print('inertia is %f' % best_inertia)
        result['%d' % k_] = best_label
    np.savez(out_fnm, **result)


if __name__ == '__main__':
    main()
