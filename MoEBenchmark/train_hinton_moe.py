#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
train_hinton_moe.py

I just dropped by Hinton's mixture of experts paper and find people train MoE in other ways.
The weighted average based on probability is on loss function instead of model prediction.
I wish the reviewers do not recognize this.
"""
from __future__ import print_function, division
import torch
import scipy.linalg
import torch.nn.functional as f
import torch.nn as nn
from torch.autograd import Variable
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs, npload
from pyLib.math import stdify, destdify
from pyLib.train import GaoNet, genTrainConfig, trainOne

import util


def main():
    # test is dummy arguments
    # run trains the model
    # eval evaluate the model
    # prob evaluate probability to find assignment
    # valid means we evaluate model on the validation set
    # label means we get prediction of label
    args = util.get_args('test', 'run', 'eval', 'prob', 'valid', 'label')
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
    outdir='models/%s/moe' % uid
    outname='moe_model.pt'
    mdl_path = os.path.join(outdir, outname)
    eval_fun = get_moe_loader(mdl_path, True)

    data = npload(cfg['file_path'], uid)
    datax = data[cfg['x_name']]
    p, v = eval_fun(datax)

    label = np.argmax(p, axis=1)

    np.save('data/pen/moe_label.npy', label)


def eval_model_on_valid(args):
    """Evaluate model on the validation set."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/moe' % uid
    outname='moe_model.pt'
    mdl_path = os.path.join(outdir, outname)
    eval_fun = get_moe_loader(mdl_path, args.prob)

    valid_set = np.load(cfg['valid_path'])
    valid_x = valid_set[cfg['x_name']]
    valid_y = valid_set[cfg['y_name']]
    predy = eval_fun(valid_x)
    # dump output into some file
    np.savez('data/%s/moe_valid_data.npz' % uid, x=valid_x, y=predy)


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
    n_model = 5
    # create the network
    net = MoENet([dimx, 100, n_model], [[dimx, int(np.ceil(300.0 / n_model)), dimy] for _ in range(n_model)])
    config = genTrainConfig(outdir='models/%s/moe' % uid, outname='moe_model.pt')
    if args.eval:
        mdl_path = os.path.join(config['outdir'], config['outname'])
        eval_fun = get_moe_loader(mdl_path, args.prob)
        predy = eval_fun(data_feed['x'])
        return {'x': data_feed['x'], 'y': data_feed['y'], 'predy': predy}
    trainOne(config, data_feed, net=net, loss=moe_loss)


class MoENet(nn.Module):
    """The network of mixture of experts."""
    def __init__(self, clas, v_reg):
        """Initialize this object by a classifier and several regressor"""
        nn.Module.__init__(self)
        self.clus = GaoNet(clas)
        self.mdls = [GaoNet(reg) for reg in v_reg]

    @property
    def num_model(self):
        return len(self.mdls)

    def parameters(self):
        self.clusPara = []
        self.clusPara.extend(self.clus.parameters())
        self.mdlsPara = []
        [self.mdlsPara.extend(mdl.parameters()) for mdl in self.mdls]
        return self.clusPara + self.mdlsPara

    def cuda(self):
        self.clus.cuda()
        for mdl in self.mdls:
            mdl.cuda()

    def cpu(self):
        self.clus.cpu()
        for mdl in self.mdls:
            mdl.cpu()

    def forward(self, x):
        """Evaluate model values, return both probability and value. x is tensor."""
        clus_x = self.clus(x)
        p = f.softmax(clus_x)
        # get a output value of n_model by minibatch by dimy
        y = torch.stack([mdl(x) for mdl in self.mdls])
        return p, y

    def get_p_y(self, x):
        """Input is numpy array, return both p and y, this is for comparison"""
        xdim = x.ndim
        if xdim == 1:
            x = np.expand_dims(x, axis=0)
        inx = Variable(torch.from_numpy(x).float(), volatile=True)
        outp, outy = self.forward(inx)
        p = outp.data.numpy()
        y = outy.data.numpy()
        if xdim == 1:
            p = np.squeeze(p, axis=0)
            y = np.squeeze(y, axis=1)
        return p, y

    def get_y(self, x):
        """Input is numpy array, return model prediction"""
        p, y = self.get_p_y(x)
        if p.ndim == 1:
            p_idx = np.argmax(p)
            return y[p_idx]
        else:
            p_idx = np.argmax(p, axis=1)
            return y[p_idx, np.arange(x.shape[0])]


def get_moe_loader(pt_path, withp=False):
    """Return a loader function for moe model given model path. We manually take care of mean and std.

    if withp is True, the model also predicts p values...
    """
    the_thing = torch.load(pt_path)
    mdl = the_thing['model']
    mdl.cpu()
    xScale = the_thing.get('xScale', None)
    yScale = the_thing.get('yScale', None)
    if xScale is None:
        xmean, xstd = None, None
    else:
        xmean, xstd = xScale
    if yScale is None:
        ymean, ystd = None, None
    else:
        ymean, ystd = yScale

    if withp:
        def get_moe_p_y(x):
            xscaled = stdify(x, xmean, xstd)
            p, y = mdl.get_p_y(xscaled)
            y = destdify(y, ymean, ystd)
            return p, y
        return get_moe_p_y
    else:
        def get_moe_y(x):
            xscaled = stdify(x, xmean, xstd)
            y = mdl.get_y(xscaled)
            return destdify(y, ymean, ystd)
        return get_moe_y


def moe_loss(predy, feedy):
    """Return the loss function for moe output and data feed"""
    p, y = predy
    diff = f.smooth_l1_loss(y, feedy.expand_as(y), False, False)
    # take average along last dim
    diff_mat = torch.mean(diff, -1)  # get nmdl by minibatch matrix
    batch_loss = torch.mul(p, diff_mat.t())
    return torch.mean(batch_loss)  # get average over this minibatch


if __name__ == '__main__':
    main()
