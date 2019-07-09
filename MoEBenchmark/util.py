#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
util.py

Several utility functions used across this domain
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import OrderedDict
import numba

from pyLib.io import getArgs

import datanames


def get_args(*args):
    """Get arguments for a problem, there are some default arguments"""
    return getArgs('pen', 'car', 'drone', 'dtwo', 'done', 'kmean', 'pcakmean', *args)


def get_label_cfg_by_args(args):
    if args.pen:
        cfg = datanames.PENDULUM
    elif args.car:
        cfg = datanames.VEHICLE
    elif args.drone:
        cfg = datanames.DRONE
    elif args.dtwo:
        cfg = datanames.DRONETWO
    else:
        print('You must choose from existing datasets')
        raise SystemExit
    if args.kmean:
        lbl_name = 'kmean_label'
    elif args.pcakmean:
        lbl_name = 'pca_kmean_label'
    else:
        try:
            if args.speuclid:
                lbl_name = 'sp_euclid_label'
            elif args.spdydx:
                lbl_name = 'sp_dydx_label'
            elif args.spvio:
                lbl_name = 'sp_vio_label'
        except:
            lbl_name = None
    return cfg, lbl_name


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


def get_vio_dst_fun(cfg):
    """Return the constraint violation function."""
    mode = cfg['uniqueid']
    sys.path.insert(0, cfg['script_path'])
    try:
        import libserver
    except:
        print('Oops, cannot import libserver, it has to be compiled')
    if mode == 'pen':
        libserver.init(cfg['cfg_path'])
        def dst_fun(x0, y0, x1, y1):
            xmid = (x0 + x1) / 2
            ymid = (y0 + y1) / 2
            c = libserver.eval(ymid.astype(np.float64))
            return np.linalg.norm(c[1:])
    elif mode == 'car':
        solver = libserver.pysolver()
        solver.initfnm(cfg['cfg_path'])
        def dst_fun(x0, y0, x1, y1):
            xmid = (x0 + x1) / 2
            ymid = (y0 + y1) / 2
            c = solver.constrEval(ymid.astype(np.float64))
            return np.linalg.norm(c[1:])
    elif mode == 'drone':
        solver = libserver.pysolver()
        solver.initfnm(cfg['cfg_path'])
        def dst_fun(x0, y0, x1, y1):
            xmid = (x0 + x1) / 2  # this is not used, maybe not good but who knows
            ymid = (y0 + y1) / 2
            solver.updateObstacle(xmid[3:])
            c = solver.constrEval(ymid.astype(np.float64))
            c[-19:] = np.maximum(0, c[-19:])
            return np.linalg.norm(c[1:-19]) + np.amax(np.sqrt(c[-19:]))
    elif mode == 'droneone':
        solver = libserver.pysolver()
        solver.initfnm(cfg['cfg_path'])
        def dst_fun(x0, y0, x1, y1):
            xmid = (x0 + x1) / 2  # this is not used, maybe not good but who knows
            ymid = (y0 + y1) / 2
            solver.updateObstacle(xmid[3:])
            c = solver.constrEval(ymid.astype(np.float64))
            c[-19:] = np.maximum(0, c[-19:])
            return np.linalg.norm(c[1:-19]) + np.amax(np.sqrt(c[-19:]))
    return dst_fun


def get_xy_vio_fun(cfg):
    """Return a function that evaluates constraint violation for one example."""
    mode = cfg['uniqueid']
    sys.path.insert(0, cfg['script_path'])
    try:
        import libserver
    except:
        print('Oops, cannot import libserver, it has to be compiled')
    if mode == 'pen':
        libserver.init(cfg['cfg_path'])
        def vio_fun(x, y):
            c = libserver.eval(y.astype(np.float64))
            return np.linalg.norm(c[1:])
    elif mode == 'car':
        solver = libserver.pysolver()
        solver.initfnm(cfg['cfg_path'])
        def vio_fun(x, y):
            c = solver.constrEval(y.astype(np.float64))
            return np.linalg.norm(c[1:])
    elif mode == 'drone':
        solver = libserver.pysolver()
        solver.initfnm(cfg['cfg_path'])
        def vio_fun(x, y):
            solver.updateObstacle(x[3:].astype(np.float64))
            c = solver.constrEval(y.astype(np.float64))
            maxc = np.amax(c[-19:])
            # vio = np.linalg.norm(c[1:-19])
            vio = 0
            if maxc >= 0:
                vio += x[6] - np.sqrt(x[6]**2 - maxc)
            return vio
    elif mode == 'droneone':
        vio_fun = drone_eval_pred_vio
    elif mode == 'dronetwo':
        vio_fun = drone_two_eval_pred_vio
    return vio_fun


def get_clus_reg_by_dir(path):
    """Get all the classifier and regressors within an folder.

    It should be noted that that path contains several groups of k
    """
    folders = glob.glob(os.path.join(path, '*/'))
    result = OrderedDict()
    folder_k = [int(re.findall('\/(\d+)\/$', x)[-1]) for x in folders]
    folder_order = np.argsort(folder_k)
    folders = [folders[idx] for idx in folder_order]
    for k, folder in enumerate(folders):
        files = glob.glob(os.path.join(folder, '*'))
        clas = []
        reg = []
        for file_ in files:
            if 'classifier_' in file_:
                clas.append(file_)
            else:
                reg.append(file_)
        reg.sort(key=lambda x: int(re.findall('\/model_(\d+)_', x)[-1]))
        result[folder_k[folder_order[k]]] = (clas[0], reg)
    return result


@numba.njit(fastmath=True)
def drone_two_eval_pred_vio(obs, pred):
    """obs and pred are all 1D array"""
    N = 20
    dimx = 12
    pathi = pred
    obsi0 = obs[3:7]
    obsi1 = obs[7:]
    worst_vio = 0
    tX = pathi[0:N*dimx].reshape((N, dimx))
    for j in range(N):
        constrvio = np.linalg.norm(tX[j, :3] - obsi0[:3]) - obsi0[3]
        if constrvio < worst_vio:
            worst_vio = constrvio
        constrvio = np.linalg.norm(tX[j, :3] - obsi1[:3]) - obsi1[3]
        if constrvio < worst_vio:
            worst_vio = constrvio
    return worst_vio


@numba.njit(fastmath=True)
def drone_eval_pred_vio(obs, pred):
    """obs and pred are 1D array"""
    N = 20
    dimx = 12
    pathi = pred
    obsi0 = obs[3:7]
    worst_vio = 0
    tX = pathi[0:N*dimx].reshape((N, dimx))
    for j in range(N):
        constrvio = np.linalg.norm(tX[j, :3] - obsi0[:3]) - obsi0[3]
        if constrvio < worst_vio:
            worst_vio = constrvio
    return worst_vio
