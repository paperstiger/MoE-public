#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
train_models.py

Train models based on cluster strategy proposed before.
"""
from __future__ import print_function, division
import torch
import scipy.linalg
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

from pyLib.io import getArgs, joinNumber
from pyLib.train import genTrainConfig, trainOne
import pyLib.parallel as pp

import util


DEBUG = False


def main():
    # pen, car, drone still means which problem we want to look into
    # pcakmean specifies the clustering approach we intend to use
    # clas means we train classifier
    # debug enables debug mode
    # miss trains models that are missing, but why?
    args = util.get_args('debug', 'clas', 'miss')
    global DEBUG
    if args.debug:
        DEBUG = True
    cfg, lbl_name = util.get_label_cfg_by_args(args)
    if args.pen:
        def archi_fun(ratio):
            return gen_archi_fun([2, 300, 75], ratio, 20)
        clas = [2, 100, -1]
    elif args.car:
        def archi_fun(ratio):
            return gen_archi_fun([4, 200, 200, 149], ratio, 20)
        clas = [4, 500, -1]
    elif args.drone:
        def archi_fun(ratio):
            return gen_archi_fun([7, 1000, 1000, 317], ratio, 20)
        clas = [7, 1000, -1]
    elif args.dtwo:
        def archi_fun(ratio):
            return gen_archi_fun([11, 2000, 2000, 317], ratio, 30)
        clas = [11, 1000, -1]
    elif args.done:
        def archi_fun(ratio):
            return gen_archi_fun([7, 1000, 1000, 317], ratio, 30)
        clas = [7, 1000, -1]
    else:
        print('You have to choose one dataset')
        raise SystemExit
    if args.clas:
        train_model_classifier(cfg, lbl_name, clas)
    elif args.miss:
        train_missing_models(cfg, lbl_name, archi_fun)
    else:
        train_model_by_data(cfg, lbl_name, archi_fun)


def train_model_classifier(cfg, lbl_name, clas):
    """Train classifier for selected dataset.

    cfg: the config which specifies which dataset we work on
    lbl_name: specifying which clustering approach to unpack
    clas: list, int, the architecture of the classifier
    """
    uid = cfg['uniqueid']
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    data = np.load(cfg['file_path'])
    x = data[cfg['x_name']]
    assert clas[0] == x.shape[1]

    keys = label_data.keys()
    keys.sort(key=int)
    nkey = len(keys)
    nProcess = 8
    if nkey < nProcess:
        nProcess = nkey
    x_arr = pp.getSharedNumpy(x)
    lst_task = pp.getTaskSplit(nkey, nProcess)
    mp = pp.MulProcess(train_one_classifier, lst_task, nProcess, keys, uid, lbl_name, clas, label_data, x_arr)
    mp.run(wait=0.1)


def train_one_classifier(task, keys, uid, lbl_name, clas, label_data, x_arr):
    """A single process that trains one classifier.

    Paramters
    ---------
    task: list of two integers, specifying which tasks we use
    keys: list of keys, work with task to locate cluster strategy
    clas: list, integer, neural network architecture
    label_data: dict, mapping from key to label
    x_arr: sharedNumpy, the feature vectors
    """
    x = pp.getNumpy(x_arr)
    for i in range(task[0], task[1]):
        key = keys[i]
        n_cluster = int(key)
        clas[-1] = n_cluster
        model_directory = 'models/%s/%s/%s' % (uid, lbl_name, key)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        label = label_data[key]
        datai = {'x': x, 'label': label, 'n_label': n_cluster}
        config = genTrainConfig(network=clas, batch_size=256, test_batch_size=2048,
                outdir=model_directory, outname='classifier_%d_of_%s.pt' % (n_cluster, joinNumber(clas)))
        trainOne(config, datai, is_reg_task=False)  # switch to classifier


def train_missing_models(cfg, lbl_name, archi_fun):
    """Figure out which models are missing and train them."""
    uid = cfg['uniqueid']
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    n_data = x.shape[0]

    exist_mdls = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    exist_mdl_key = exist_mdls.keys()

    keys = label_data.keys()
    keys.sort(key=int)

    for key in keys:
        n_cluster = int(key)
        model_directory = 'models/%s/%s/%s' % (uid, lbl_name, key)
        files = glob.glob(os.path.join(model_directory, '*'))
        # create an array of missing flag
        miss_flag = np.ones(n_cluster, dtype=int)
        for file_ in files:
            if 'classifier_' in file_:
                continue
            mdl_idx = int(re.findall('\/model_(\d+)_', file_)[-1])
            miss_flag[mdl_idx] = 0  # this is not missing
        miss_idx = np.where(miss_flag == 1)[0]
        if miss_idx.shape[0] == 0:
            print('No missing model for key ', key)
        else:
            print('missing models are ', miss_idx)
            label = label_data[key]
            x_arr, y_arr, label_arr = pp.getSharedNumpy(x, y, label)
            for idx in miss_idx:
                train_one_model([idx, idx + 1], model_directory, archi_fun, x_arr, y_arr, label_arr)


def train_model_by_data(cfg, lbl_name, archi_fun, evaluate=False):
    """Train a model based on a few things specified before.

    archi_fun is a callable that takes in x, y, integer of data size and output a model layer list
    """
    uid = cfg['uniqueid']
    label_data = np.load('data/%s/%s.npz' % (uid, lbl_name))
    data = np.load(cfg['file_path'])
    x, y = data[cfg['x_name']], data[cfg['y_name']]
    n_data = x.shape[0]

    keys = label_data.keys()
    keys.sort(key=int)

    for key in keys:
        n_cluster = int(key)
        model_directory = 'models/%s/%s/%s' % (uid, lbl_name, key)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        label = label_data[key]
        nProcess = 8
        if DEBUG:
            nProcess = 1
            n_cluster = 1
        lst_task = pp.getTaskSplit(n_cluster, nProcess)
        x_arr, y_arr, label_arr = pp.getSharedNumpy(x, y, label)
        mp = pp.MulProcess(train_one_model, lst_task, nProcess, model_directory, archi_fun, x_arr, y_arr, label_arr)
        mp.run(wait=0.1)


def train_one_model(task, mdl_dir, archi_fun, x_arr, y_arr, label_arr):
    """Function run by a single process."""
    x, y, label = pp.getNumpy(x_arr, y_arr, label_arr)
    n_data = x.shape[0]
    for i in range(task[0], task[1]):
        maski = i == label
        datai = {'x': x[maski], 'y': y[maski]}
        net = archi_fun(datai['x'].shape[0] / float(n_data))
        config = genTrainConfig(network=net, batch_size=64, test_batch_size=1024,
                outdir=mdl_dir, outname='model_%d_of_%s.pt' % (i, joinNumber(net)))
        trainOne(config, datai)


def gen_archi_fun(default_list, ratio, min_neuron):
    dimx, dimy = default_list[0], default_list[-1]
    n_layer = len(default_list)
    total_param = np.sum([default_list[i] * default_list[i + 1] for i in range(n_layer - 1)])
    if n_layer == 3:  # only one hidden layer
        mid_num = int(ratio * default_list[1])
        if mid_num < min_neuron:
            mid_num = min_neuron
        return [dimx, mid_num, dimy]
    elif n_layer == 4:
        n_param = int(ratio * total_param)
        U = np.roots([1, dimx + dimy, -n_param])
        idx = 0 if U[0] > 0 else 1
        if U[idx] < min_neuron:
            use_n = min_neuron
        else:
            use_n = int(U[idx])
        return [dimx, use_n, use_n, dimy]
    else:
        print('Currently support up to 2 hidden layers')
        raise SystemExit


if __name__ == '__main__':
    main()
