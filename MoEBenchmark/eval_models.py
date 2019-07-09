#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
eval_models.py

Evaluation of models for those problems on a validation set.
"""
from __future__ import print_function, division
import torch
import scipy.linalg
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import glob
import numba
import cPickle as pkl

from pyLib.io import getArgs
from pyLib.train import MoMNet, modelLoader, GaoNet
from pyLib.math import l1loss

import util


DEBUG = False


def main():
    # pen, car, drone still means which problem we want to look into
    # pcakmean specifies the clustering approach we intend to use
    # error means we calculate evaluation error on the validation set
    # constr means we evaluate constraint violation
    # eval just evaluates data on validation set and save into a npz file for rollout validation
    # snn means we evaluate SNN network
    # roll means look into rollout results and extract useful information. It turns out they all fail, somehow
    args = util.get_args('debug', 'error', 'constr', 'eval', 'snn', 'roll')
    global DEBUG
    if args.debug:
        DEBUG = True
    cfg, lbl_name = util.get_label_cfg_by_args(args)
    if args.error:
        eval_valid_error(cfg, lbl_name, args)
    if args.constr:
        eval_valid_constr_vio(cfg, lbl_name, args)
    if args.eval:
        eval_on_valid(cfg, lbl_name, args)
    if args.roll:
        check_rollout_results(cfg, lbl_name, args)


def check_rollout_results(cfg, lbl_name, args):
    """Check rollout results"""
    uid = cfg['uniqueid']
    if args.snn:
        datanm = 'data/%s/snn_rollout_result.pkl' % uid
        with open(datanm, 'rb') as f:
            Rst = pkl.load(f)
        keys = Rst.keys()
        print(keys)
        if args.dtwo or args.done or args.drone:
            # load flag of validation set, if necessary
            vdata = np.load(cfg['valid_path'])
            vio = Rst['vio']
            if 'flag' in vdata.keys():
                mask = vdata['flag'] == 1
            else:
                mask = np.ones(vio.shape[0], dtype=bool)
            print('valid set mask size ', np.sum(mask))
            vio = vio[mask]
            fig, ax = plt.subplots()
            ax.hist(vio, bins=20)
            plt.show()
            print('mean vio ', np.sum(vio[vio < 0]) / vio.shape[0])
            print('max vio ', np.amin(vio))
    else:
        datanm = 'data/%s/%s_rollout_result.pkl' % (uid, lbl_name)
        datanm = datanm.replace('_label', '')
        with open(datanm, 'rb') as f:
            Rst = pkl.load(f)

        if args.pen or args.car:
            keys = Rst.keys()
            keys.sort(key=int)

            for key in keys:
                print('key = ', key)
                key_rst = Rst[key]
                if args.pen:
                    status = np.array([tmp['status'] for tmp in key_rst])
                    print(np.sum(status == 1))
                elif args.car:
                    vXf = np.array([rst['statef'] for rst in key_rst])
                    # fix for angle
                    vXf[:, 2] = np.mod(vXf[:, 2], 2*np.pi)
                    inds = vXf[:, 2] > np.pi
                    vXf[inds, 2] = 2 * np.pi - vXf[inds, 2]
                    normXf = np.linalg.norm(vXf, axis=1)
                    print(np.sum(normXf < 0.5))
        elif args.dtwo or args.done or args.drone:
            vvio = Rst['vio']
            vdata = np.load(cfg['valid_path'])
            if 'flag' in vdata.keys():
                mask = vdata['flag'] == 1
            else:
                mask = np.ones(vio.shape[0], dtype=bool)
            print('valid set mask size ', np.sum(mask))
            for vio_ in vvio:
                vio = vio_[mask]
                print('mean vio ', np.sum(vio[vio < 0]) / vio.shape[0], ' max vio ', np.amin(vio))
            fig, ax = plt.subplots()
            ax.hist(vvio, bins=20)
            plt.show()


def eval_on_valid(cfg, lbl_name, args):
    """Just perform evaluation on validation set, save the outputs into some file."""
    # load the validation set
    vdata = np.load(cfg['valid_path'])
    if 'valid_x_name' in cfg:
        x = vdata[cfg['valid_x_name']]
    else:
        x = vdata[cfg['x_name']]
    uid = cfg['uniqueid']
    if args.snn:
        mdlfun = modelLoader(cfg['snn_path'])
        predy = mdlfun(x)
        np.save('data/%s/snn_validation_predict.npy' % uid, predy)
        return
    # load MoE models from desired directory
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    keys = result.keys()
    print('existing keys ', keys)
    out_dict = {}
    for key in keys:
        print('For key ', key)
        cls, regs = result[key]
        net = MoMNet(cls, regs)
        predy = net.getPredY(x)
        out_dict[str(key)] = predy
    np.savez('data/%s/%s_validation_predict.npz' % (uid, lbl_name), **out_dict)


def eval_valid_error(cfg, lbl_name, args):
    """Evaluation of trained models on validation set."""
    # load the validation set
    vdata = np.load(cfg['valid_path'])
    if 'valid_x_name' in cfg:
        x = vdata[cfg['valid_x_name']]
    else:
        x = vdata[cfg['x_name']]
    if 'valid_y_name' in cfg:
        y = vdata[cfg['valid_y_name']]
    else:
        y = vdata[cfg['y_name']]
    if 'flag' in vdata.keys():
        mask = np.where(vdata['flag'] == 1)
        x = x[mask]
        y = y[mask]
    print('validation set size ', x.shape, y.shape)
    uid = cfg['uniqueid']
    if args.snn:
        mdlfun = modelLoader(cfg['snn_path'])
        predy = mdlfun(x)
        error = np.mean(l1loss(y, predy), axis=1)  # get error for each instance
        print(np.mean(error))
        fig, ax = plt.subplots()
        ax.hist(error, bins=20)
        ax.set_yscale('log', nonposy='clip')
        ax.set_ylim(1, ax.get_ylim()[1])
        ax.legend()
        plt.show()
        return
    # load MoE models from desired directory
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    keys = result.keys()
    print('existing keys ', keys)
    v_error = []
    for key in keys:
        print('For key ', key)
        cls, regs = result[key]
        net = MoMNet(cls, regs)
        predy = net.getPredY(x)
        error = np.mean(l1loss(y, predy), axis=1)  # get error for each instance
        v_error.append(error)
    v_mean_error = [np.mean(error) for error in v_error]
    print('mean error is ', v_mean_error)
    # show histogram
    fig, ax = plt.subplots()
    ax.hist(v_error, bins=20, label=keys)
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(1, ax.get_ylim()[1])
    ax.legend()
    plt.show()


def eval_valid_constr_vio(cfg, lbl_name, args):
    """Evaluate models by violation of constraints."""
    # load violation evaluation function
    vio_fun = util.get_xy_vio_fun(cfg)
    # get validation dataset
    vdata = np.load(cfg['valid_path'])
    if 'valid_x_name' in cfg:
        x = vdata[cfg['valid_x_name']]
    else:
        x = vdata[cfg['x_name']]
    if 'valid_y_name' in cfg:
        y = vdata[cfg['valid_y_name']]
    else:
        y = vdata[cfg['y_name']]
    if 'flag' in vdata.keys():
        mask = np.where(vdata['flag'] == 1)
        x = x[mask]
        y = y[mask]
    uid = cfg['uniqueid']
    # first we try the snn case
    if args.snn:
        mdlfun = modelLoader(cfg['snn_path'])
        predy = mdlfun(x)
        n_data = x.shape[0]
        error = np.zeros(n_data)
        for i in range(n_data):
            error[i] = vio_fun(x[i], predy[i])
        print('average is %f' % (np.sum(error[error < 0]) / n_data))
        print('max error is ', np.amin(error))
        fig, ax = plt.subplots()
        ax.hist(error)
        plt.show()
        return
    # load MoE models from desired directory
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    keys = result.keys()
    print('existing keys ', keys)
    v_error = []
    fig, ax = plt.subplots()
    for key in keys:
        print('For key ', key)
        cls, regs = result[key]
        net = MoMNet(cls, regs)
        predy = net.getPredY(x)
        n_data = x.shape[0]
        error = np.zeros(n_data)
        for i in range(n_data):
            error[i] = vio_fun(x[i], predy[i])
        v_error.append(error)
        merror = get_moving_average(error)
        ax.plot(merror)
    v_mean_error = [np.mean(error) for error in v_error]
    print('mean error is ', v_mean_error)
    print('mean neg error ', [np.sum(error[error < 0]) / error.shape[0] for error in v_error])
    print('max error is ', [np.amin(error) for error in v_error])
    # show histogram
    fig, ax = plt.subplots()
    ax.hist(v_error, bins=20, label=keys)
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(1, ax.get_ylim()[1])
    ax.legend()
    ax.set_xlabel('Constraint Violation')
    ax.set_ylabel('Count')
    fig.savefig('gallery/%s/%s_valid_constr_vio_hist.pdf' % (uid, lbl_name))
    plt.show()


@numba.njit
def get_moving_average(x):
    """Obtain the moving average"""
    nx = x.shape[0]
    mx = np.zeros(nx)
    val = 0
    for i in range(nx):
        val += x[i]
        mx[i] = val / (i + 1)
    return mx


if __name__ == '__main__':
    main()
