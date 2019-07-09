#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
train_mom_net.py

Train the predefined mom net.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs, npload
from pyLib.math import stdify, destdify
from pyLib.train import GaoNet, genTrainConfig, trainOne, MoMNet, momLoader

import util


def main():
    # run trains the model
    # eval evaluate the model and dump into some file
    # prob evaluate probability to find assignment
    # k is the number of experts
    # valid changes model evaluation on the validation set
    # label evaluates label
    args = util.get_args('run', 'eval', 'prob', 'k5', 'valid', 'label')
    if args.run:
        run_the_training(args)
    if args.eval:
        eval_model(args)
    if args.valid:
        eval_model_on_valid(args)
    if args.label:
        eval_final_label(args)


def eval_final_label(args):
    """Evaluation of labels on the training set."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/mom' % uid
    outname='mom_model.pt'
    mdl_path = os.path.join(outdir, outname)
    eval_fun = momLoader(mdl_path, True)

    data = npload(cfg['file_path'], uid)
    datax = data[cfg['x_name']]
    p, v = eval_fun(datax)

    label = np.argmax(p, axis=1)

    np.save('data/pen/mom_label.npy', label)


def eval_model_on_valid(args):
    """Evaluate model on the validation set."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/mom' % uid
    outname='mom_model.pt'
    eval_fun = momLoader(os.path.join(outdir, outname), args.prob, False)
    valid_set = np.load(cfg['valid_path'])
    valid_x = valid_set[cfg['x_name']]
    valid_y = valid_set[cfg['y_name']]
    predy = eval_fun(valid_x)
    # dump output into some file
    np.savez('data/%s/mom_valid_data.npz' % uid, x=valid_x, y=predy)


def eval_model(args):
    """Evaluate a model."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    eval_rst = run_the_training(args)
    y = eval_rst['y']
    if args.prob:
        x = eval_rst['x']
        p, predy = eval_rst['predy']
        print(p.shape, predy.shape)
        argmax_idx = np.argmax(p, axis=1)
        n_model = np.amax(argmax_idx) + 1
        fig, ax = plt.subplots()
        for i in range(n_model):
            mask = argmax_idx == i
            ax.scatter(*x[mask].T)
        plt.show()
    else:
        predy = eval_rst['predy']
        n_data = y.shape[0]
        # get error
        error_y = y - predy
        error_norm = np.linalg.norm(error_y, axis=1)
        # get histogram
        fig, ax = plt.subplots()
        ax.hist(error_norm, bins='auto')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')

        diff_angle = y[:, 48] - predy[:, 48]
        fig, ax = plt.subplots()
        ax.hist(diff_angle, bins='auto')
        ax.set_xlabel(r'$\theta_f$ Error')
        ax.set_ylabel('Count')
        # show difference
        error_order = np.argsort(error_norm)
        for i in range(n_data // 10):
            fig, ax = plt.subplots()
            for j in range(10):
                idx = -1 - (i*10 + j)
                ax.plot(y[idx], color='C%d' % j)
                ax.plot(predy[idx], color='C%d' % j, ls='--')
            plt.show()


def run_the_training(args):
    """Run the MoE training without using any clustering information but let it find it on its own."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    data = npload(cfg['file_path'], uid)
    data_feed = {'x': data[cfg['x_name']], 'y': data[cfg['y_name']]}
    dimx = data_feed['x'].shape[1]
    dimy = data_feed['y'].shape[1]
    n_model = args.k
    # create the network
    net = MoMNet([dimx, 100, n_model], [[dimx, int(np.ceil(300.0 / n_model)), dimy] for _ in range(n_model)])
    net.argmax = False
    config = genTrainConfig(outdir='models/%s/mom' % uid, outname='mom_model.pt', overwrite=False)
    if args.eval:
        mdl_path = os.path.join(config['outdir'], config['outname'])
        eval_fun = momLoader(mdl_path, withclus=args.prob, argmax=False)
        predy = eval_fun(data_feed['x'])
        return {'x': data_feed['x'], 'y': data_feed['y'], 'predy': predy}
    trainOne(config, data_feed, net=net)


if __name__ == '__main__':
    main()
