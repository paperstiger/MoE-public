#! /usr/bin/env python

"""
Use the new function type approach to rewrite the dubins car problems
This time I will use 2 million states seen by the RL algorithm so they are comparable
"""
import copy
import glob
import os
import time

import numpy as np
import numba
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import externalmodel
import ppo_util as pu

from pyLib.io import getArgs
import pyLib.plot as pld


VALIDATION_SET = os.path.expanduser('~/GAO/OCPLearn/data/validate.npz')


def main():
    # train means train a model with one obstacle
    # show shows model
    # fix uses fixed states and obstacle
    # render means render the gym
    # valid means we run rollout on the validation set
    # small means sample initial states in a small support
    # medium means sample initial states within a medium region
    # medhard means medium hard difficulty level
    args = getArgs('train', 'show', 'fix', 'render', 'small', 'medium', 'medhard', 'valid')
    if args.train:
        train_the_model(args)
    if args.show:
        show_the_model(args)


def show_the_model(args):
    """Show the problem"""
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')
    env, env_name = model_build(fix=args.fix, small=args.small, medium=args.medium, medhard=args.medhard)
    config = pu.get_train_gym_config(env_name=env_name, seed=np.random.randint(10000))
    sim_n = 200
    if args.small:
        sim_n = 200
    if args.medium:
        sim_n = 500
    valid_x0 = None  # use this to load validation set
    if args.valid:
        valid_x0 = np.load(VALIDATION_SET)['x0']
        sim_n = valid_x0.shape[0]
    v_x0, v_xf, v_traj = pu.policy_rollout(env, config, sim_n, show=args.render, return_traj=True, valid_x0=valid_x0)
    v_xf_norm = np.linalg.norm(v_xf, axis=1)
    level = [0.1, 0.5, 1, 2]
    for level_ in level:
        print('level %f count %d' % (level_, np.sum(v_xf_norm < level_)))
    # find closest state
    v_min_state_norm = np.zeros(sim_n)
    for i in range(sim_n):
        v_min_state_norm[i] = np.amin(np.linalg.norm(v_traj[i]['state'], axis=1))
    datafnm = 'data/%s_all_rst.npz' % env_name
    figfnm = 'gallery/%s_rollout_xf.pdf' % env_name
    figfnm_min = 'gallery/%s_rollout_min_x.pdf' % env_name
    if args.small:
        datafnm = datafnm.replace('.npz', '_small.npz')
        figfnm = figfnm.replace('.pdf', '_small.pdf')
        figfnm_min = figfnm_min.replace('.pdf', '_small.pdf')
    if args.medium:
        datafnm = datafnm.replace('.npz', '_medium.npz')
        figfnm = figfnm.replace('.pdf', '_medium.pdf')
        figfnm_min = figfnm_min.replace('.pdf', '_medium.pdf')
    if args.medhard:
        datafnm = datafnm.replace('.npz', '_medhard.npz')
        figfnm = figfnm.replace('.pdf', '_medhard.pdf')
        figfnm_min = figfnm_min.replace('.pdf', '_medhard.pdf')
    if args.valid:
        datafnm = datafnm.replace('.npz', '_valid.npz')
        figfnm = figfnm.replace('.pdf', '_valid.pdf')
        figfnm_min = figfnm_min.replace('.pdf', '_valid.pdf')
    for level_ in level:
        print('min level %f count %d' % (level_, np.sum(v_min_state_norm < level_)))
    np.savez(datafnm, x0=v_x0, xf=v_xf, traj=v_traj)
    fig, ax = plt.subplots()
    ax.hist(v_xf_norm, bins='auto')
    ax.set_xlabel(r'$\|x_f\|$')
    ax.set_ylabel('Count')
    fig.savefig(figfnm)
    fig, ax = plt.subplots()
    ax.hist(v_min_state_norm, bins='auto')
    ax.set_xlabel(r'$\|x\|_{\min}$')
    ax.set_ylabel('Count')
    fig.savefig(figfnm_min)
    plt.show()


def train_the_model(args):
    """Train a model for this particular problem."""
    # construct the fix world thing
    env, env_name = model_build(fix=args.fix, small=args.small, medium=args.medium, medhard=args.medhard)
    num_frames = 3e5
    save_step = 1e5
    if args.small:
        num_frames = 1e6
        save_step = None
    config = pu.get_train_gym_config(env_name=env_name, log_dir='logs/dubin', seed=np.random.randint(10000),
            num_frames=num_frames, save_interval=1000, save_step=save_step)
    pu.train_a_gym_model(env, config)


def model_build(fix=True, small=False, medium=False, medhard=False):
    """Build the gym environment for this problem."""
    if fix:
        env = externalmodel.DubinCarEnv(np.array([5, 5, 0, 0]))
    elif small:
        env = externalmodel.DubinCarEnv(x0bd=(np.array([5, 5, 0, 0]), np.array([10, 10, 0, 0])))
    elif medium:
        env = externalmodel.DubinCarEnv(x0bd=(np.array([0, 5, 0, 0]), np.array([10, 10, 0, 0])))
    elif medhard:
        env = externalmodel.DubinCarEnv(x0bd=(np.array([-10, 5, 0, 0]), np.array([10, 10, 0, 0])))
    else:
        env = externalmodel.DubinCarEnv()
    env_name = 'dubinCar'
    if fix:
        env_name += '-fix'
    if small:
        env_name += '-small'
    if medium:
        env_name += '-medium'
    if medhard:
        env_name += '-medhard'
    return env, env_name


if __name__ == "__main__":
    main()
