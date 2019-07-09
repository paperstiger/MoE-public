#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
train_jacob_moe.py

So this guy Jacob proposed another cost function which trains experts and classifier individually.
I use a gan style architecture to implement this.
"""
from __future__ import print_function, division
import torch
import scipy.linalg
import torch.nn.functional as f
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs, npload
from pyLib.math import stdify, destdify, l1loss
from pyLib.train import GaoNet, genTrainConfig, trainOne, LabelFactory, SubFactory, KeyFactory, DataLoader
import pyLib.plot as pld

import util


KLLOSS = True  # enable kl divergence loss for classifier


def main():
    # test is dummy arguments
    # run trains the model
    # prob evaluate probability to find assignment
    # valid means we evaluate model on the validation set
    # label means we get prediction of label
    # draw directly shows figure
    # argmax is used to enable argmax mode of model when it is loaded from file
    # warm checks how the new training approach recovers from bad initialization from k-Means
    args = util.get_args('test', 'run', 'eval', 'prob', 'valid', 'label', 'draw', 'argmax', 'warm')
    if args.run:
        if args.warm:
            train_model_warm(args)
        else:
            run_the_training(args)
    if args.valid:
        eval_model_on_valid(args)
    if args.label:
        eval_final_label(args)
    if args.eval:
        eval_model(args)


def train_model_warm(args):
    """Train that model warmly by loading some pre-trained models.

    We say warm we mean we pretrain gate or experts. Not sure which I should use instead.
    Maybe I really need a function to modify weights of GaoNet to consider xScale and yScale
    """
    cfg, lbl_name = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    result = util.get_clus_reg_by_dir('models/%s/%s' % (uid, lbl_name))
    cls, regs = result[5]  # we know this is bad but could possibly work
    cls_data = torch.load(cls)
    print(cls_data.keys())
    cls_net = cls_data['model']
    xmean, xstd = cls_data['xScale']
    print('xmean', xmean, 'xstd', xstd)
    # cls_net.extendXYScale((xmean, xstd))
    expert = Experts([[2, 60, 75]] * 5)
    run_the_training(args, cls_net, expert)


def eval_model(args):
    """Evaluate model on training data."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/gate_expert' % uid
    outname='gate_expert_model.pt'
    if KLLOSS:
        outname = 'gate_expert_kldiv_model.pt'
    if args.warm:
        outname = outname.replace('.pt', '_warm.pt')
    mdl_path = os.path.join(outdir, outname)
    gate_expert = GateExpertNet(mdl_path, args.argmax)
    eval_fun = gate_expert.get_y

    data = npload(cfg['file_path'], uid)
    datax = data[cfg['x_name']]
    datay = data[cfg['y_name']]
    evaly = eval_fun(datax)
    print(np.histogram(evaly[:, 48]))
    fig, ax = pld.get3dAxis()
    ax.scatter(datax[:, 0], datax[:, 1], evaly[:, 48])
    loss = l1loss(evaly, datay)
    err_norm = np.mean(loss, axis=1)
    fig, ax = plt.subplots()
    ax.hist(err_norm)
    plt.show()


def eval_final_label(args):
    """Evaluation of labels on the training set."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/gate_expert' % uid
    outname='gate_expert_model.pt'
    if KLLOSS:
        outname = 'gate_expert_kldiv_model.pt'
    if args.warm:
        outname = outname.replace('.pt', '_warm.pt')
    mdl_path = os.path.join(outdir, outname)
    gate_expert = GateExpertNet(mdl_path, False)
    eval_fun = gate_expert.get_p_y

    data = npload(cfg['file_path'], uid)
    datax = data[cfg['x_name']]
    p, v = eval_fun(datax)

    label = np.argmax(p, axis=1)

    if args.draw:
        fig, ax = plt.subplots()
        n_expert = np.amax(label) + 1
        for i in range(n_expert):
            mask = label == i
            ax.scatter(datax[mask, 0], datax[mask, 1])
        plt.show()

    label_name = 'data/pen/gate_expert_label.npy'
    if KLLOSS:
        label_name = label_name.replace('_label', '_kldiv_label')
    if args.warm:
        label_name = label_name.replace('.npy', '_warm.npy')
    np.save(label_name, label)


def eval_model_on_valid(args):
    """Evaluate model on the validation set."""
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    outdir='models/%s/gate_expert' % uid
    outname='gate_expert_model.pt'
    if KLLOSS:
        outname = 'gate_expert_kldiv_model.pt'
    if args.warm:
        outname = outname.replace('.pt', '_warm.pt')
    mdl_path = os.path.join(outdir, outname)
    gate_expert = GateExpertNet(mdl_path, False)
    eval_fun = gate_expert.get_y

    valid_set = np.load(cfg['valid_path'])
    valid_x = valid_set[cfg['x_name']]
    valid_y = valid_set[cfg['y_name']]
    predy = eval_fun(valid_x)
    # dump output into some file
    valid_name = 'data/%s/gate_expert_valid_data.npz' % uid
    if KLLOSS:
        valid_name = valid_name.replace('_valid', '_kldiv_valid')
    if args.warm:
        valid_name = valid_name.replace('.npz', '_warm.npz')
    np.savez(valid_name, x=valid_x, y=predy)


def draw_clus_region(clus, x, lder):
    """Draw the region for classifier."""
    print('xmean', lder.xmean, 'xstd', lder.xstd)
    inx = Variable(torch.from_numpy(stdify(x, lder.xmean, lder.xstd)).float()).cuda()
    outy = clus(inx).cpu().data.numpy()
    label = np.argmax(outy, axis=1)
    fig, ax = plt.subplots()
    n_label = outy.shape[1]
    for i in range(n_label):
        mask = label == i
        ax.scatter(x[mask, 0], x[mask, 1])
    plt.show()


def run_the_training(args, clus=None, expert=None):
    """Run the MoE training without using any clustering information but let it find it on its own."""
    # load data
    cfg, lbl = util.get_label_cfg_by_args(args)
    uid = cfg['uniqueid']
    print('We are playing with %s' % uid)
    data = npload(cfg['file_path'], uid)
    data_feed = {'x': data[cfg['x_name']], 'y': data[cfg['y_name']]}
    dimx = data_feed['x'].shape[1]
    dimy = data_feed['y'].shape[1]

    # create gate and expert
    if clus is None:
        n_model = 5
        clus = GaoNet([dimx, 100, n_model])
        expert = Experts([[dimx, 60, dimy]] * n_model)
    # cuda it
    clus.cuda()
    expert.cuda()

    # set data loader
    xname, yname = 'x', 'y'
    factory = KeyFactory(data_feed, xname, yname, scalex=True, scaley=True)
    factory.shuffle(None)

    draw_clus_region(clus, data_feed['x'], factory)

    # create two sets
    trainsize = 0.8
    trainSet = SubFactory(factory, 0.0, trainsize)
    testSet = SubFactory(factory, trainsize, 1.0)
    batch_size = 32
    test_batch_size = -1
    trainLder = DataLoader(trainSet, batch_size=batch_size, shuffle=False)
    testLder = DataLoader(testSet, batch_size=test_batch_size, shuffle=False)

    # set up file output
    outname = 'gate_expert_model.pt'
    outdir = 'models/pen/gate_expert'
    if KLLOSS:
        outname = 'gate_expert_kldiv_model.pt'
    if args.warm:
        outname = outname.replace('.pt', '_warm.pt')

    # set optimizer
    lr = 1e-3
    opt_G = torch.optim.Adam(clus.parameters(), lr=lr)
    opt_E = torch.optim.Adam(expert.parameters(), lr=lr)

    # set other training stuff
    n_epoch = 500
    back_check_epoch = 8
    best_test_loss = np.inf
    best_test_loss_expert = np.inf
    best_test_epoch = 0

    def get_mean_error(g_y, exp_y, feedy):
        """Calculate two loss"""
        error_traj = torch.mean((exp_y - feedy.expand_as(exp_y)) ** 2, dim=2).t()
        g = f.softmax(g_y)
        log_g = f.log_softmax(g_y)
        posterior = g * torch.exp(-0.5 * error_traj)  # b by r probability, not scaled to 1
        traj_prob = torch.mean(-torch.log(torch.sum(posterior, dim=1)))
        if KLLOSS:
            posterior_scale = Variable((posterior / torch.sum(posterior, dim=1, keepdim=True)).data)  # do not use gradient of it
            div_error = f.kl_div(log_g, posterior_scale)
            return traj_prob, div_error
        else:
            Og = torch.sum(exp_y * g.t().unsqueeze(2), dim=0)
            traj_error = f.smooth_l1_loss(Og, feedy)
            return traj_prob, traj_error

    # start training
    for epoch in range(n_epoch):
        sum_train_loss = 0
        sum_train_loss_prob = 0
        for idx, batch_data in enumerate(trainLder):
            feedy = Variable(batch_data[yname], requires_grad=False).cuda()
            feedx = Variable(batch_data[xname], requires_grad=False).cuda()
            # train experts
            opt_E.zero_grad()
            opt_G.zero_grad()
            exp_y = expert(feedx)
            g_y = clus(feedx)
            g = f.softmax(g_y)  # this is prior
            log_g = f.log_softmax(g_y)
            error_traj = torch.mean((exp_y - feedy.expand_as(exp_y)) ** 2, dim=2).t()
            posterior = g * torch.exp(-0.5 * error_traj)  # b by r probability, not scaled to 1
            posterior_scale = Variable((posterior / torch.sum(posterior, dim=1, keepdim=True)).data)  # do not use gradient of it
            lossi = torch.mean(-torch.log(torch.sum(posterior, dim=1)))
            lossi.backward(retain_graph=True)
            sum_train_loss_prob += lossi.cpu().data.numpy() * feedx.size()[0]
            opt_E.step()
            # update h by regression error
            all_pred = exp_y
            if KLLOSS:
                error = f.kl_div(log_g, posterior_scale)
            else:
                Og_before = all_pred * g.t().unsqueeze(2)
                Og = torch.sum(Og_before, dim=0)
                error = f.smooth_l1_loss(Og, feedy)
            sum_train_loss += error.cpu().data.numpy() * feedx.size()[0]
            error.backward()
            opt_G.step()
            # val = clus.printWeights(3)
        mean_train_loss = sum_train_loss / trainLder.getNumData()
        mean_train_loss_prob = sum_train_loss_prob / trainLder.getNumData()

        # evaluate on test data
        sum_test_loss_gate = 0
        sum_test_loss_expert = 0
        n_test_data = testLder.getNumData()
        for idx, batch_data in enumerate(testLder):
            feedy = Variable(batch_data[yname], volatile=True).cuda()
            feedx = Variable(batch_data[xname], volatile=True).cuda()
            exp_y = expert(feedx)
            g_y = clus(feedx)
            traj_prob, div_error = get_mean_error(g_y, exp_y, feedy)
            sum_test_loss_gate += div_error.cpu().data.numpy() * feedx.size()[0]
            sum_test_loss_expert += traj_prob.cpu().data.numpy() * feedx.size()[0]
        mean_test_loss_gate = sum_test_loss_gate / n_test_data
        mean_test_loss_expert = sum_test_loss_expert / n_test_data
        print('epoch %d gate loss %f expert loss %f test gate loss %f expert loss %f' \
                % (epoch, mean_train_loss, mean_train_loss_prob, mean_test_loss_gate, mean_test_loss_expert))
        if mean_test_loss_gate < best_test_loss:
            best_test_loss = mean_test_loss_gate
            best_test_epoch = epoch
        if mean_test_loss_expert < best_test_loss_expert:
            best_test_loss_expert = mean_test_loss_expert
            best_test_epoch = epoch
        if epoch > best_test_epoch + back_check_epoch:
            break
    print('Save model now')

    # draw region for classifier
    draw_clus_region(clus, data_feed['x'], factory)

    clus.cpu()
    expert.cpu()
    model = {'gate': clus, 'expert': expert, 'xScale': [trainLder.xmean, trainLder.xstd], 'yScale': [trainLder.ymean, trainLder.ystd]}
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    torch.save(model, os.path.join(outdir, outname))


class Experts(nn.Module):
    """The network for several experts."""
    def __init__(self, v_reg):
        """Initialize this object by a classifier and several regressor"""
        nn.Module.__init__(self)
        self.mdls = [GaoNet(reg) for reg in v_reg]

    @property
    def num_model(self):
        return len(self.mdls)

    def parameters(self):
        self.mdlsPara = []
        [self.mdlsPara.extend(mdl.parameters()) for mdl in self.mdls]
        return self.mdlsPara

    def cuda(self):
        for mdl in self.mdls:
            mdl.cuda()

    def cpu(self):
        for mdl in self.mdls:
            mdl.cpu()

    def forward(self, x):
        """Evaluate model values, return both probability and value. x is tensor."""
        return torch.stack([mdl(x) for mdl in self.mdls])


class GateExpertNet(object):
    """A non-trainable guy here. It basically combines clas and regs"""
    def __init__(self, fnm, argmax=False):
        data = torch.load(fnm)
        self.xScale = data['xScale']
        self.yScale = data['yScale']
        self.gate = data['gate']
        self.expert = data['expert']
        self.argmax = argmax

    def get_p_y(self, x):
        """Input is numpy array, return both p and y, this is for comparison"""
        xdim = x.ndim
        if xdim == 1:
            x = np.expand_dims(x, axis=0)
        x = stdify(x, self.xScale[0], self.xScale[1])
        inx = Variable(torch.from_numpy(x).float(), volatile=True)
        outp = f.softmax(self.gate(inx))  # get b by p
        p = outp.data.numpy()
        outy = self.expert(inx)  # get p by b by N
        if self.argmax:
            max_idx = np.argmax(p, axis=1)
            y_three = outy.data.numpy()
            y = np.array([y_three[max_idx[i], i] for i in range(x.shape[0])])
        else:
            Og_before = outy * outp.t().unsqueeze(2)
            Og = torch.sum(Og_before, dim=0)
            y = Og.data.numpy()
        y = destdify(y, self.yScale[0], self.yScale[1])
        if xdim == 1:
            p = np.squeeze(p, axis=0)
            y = np.squeeze(y, axis=0)
        return p, y

    def get_y(self, x):
        """Input is numpy array, return model prediction"""
        p, y = self.get_p_y(x)
        return y


if __name__ == '__main__':
    main()
