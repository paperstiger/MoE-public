#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
Test the newly developed RL algorithm with changing problem parameter domain.
I hope to increase difficulty gradually and make some good progress.
It is mainly adapted from sensor_range_goal_problem.py
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

import ppo_util as pu
from indoor_gym_model import SensorRangeGoalProblem
from floorPlan import construct_default_floor_plan

import pyLib.all as pl


def main():
    # train trains a model
    # show shows the model
    args = pl.getArgs('train', 'show', 'num4', 'novio')
    if args.train:
        train_the_model(args)
    if args.show:
        show_the_model(args)


def show_the_model(args):
    """Show the model"""
    env, floor, update_fun = construct_env(args)
    env_name = 'update_%s' % str(env)
    config = pu.get_train_gym_config(env_name=env_name, seed=np.random.randint(100), num_frames=2e6)
    sim_n = args.num
    v_x0, v_xf, v_traj = pu.policy_rollout(env, config, sim_n, show=False, return_traj=True)
    plt.switch_backend('TkAgg')
    fig, axes = pl.subplots(sim_n)
    for i in range(sim_n):
        ax = axes[i]
        sim_rst = v_traj[i]
        x, u, dt = sim_rst['state'], sim_rst['action'], sim_rst['dt']
        floor.draw(ax)
        ax.plot(*x[:, :2].T)
    plt.savefig('gallery/%s-%d-cases.pdf' % (env_name, sim_n))
    plt.show()


def train_the_model(args):
    """Train a model"""
    env, floor, update_fun = construct_env(args)
    env_name = 'update_%s' % str(env)
    config = pu.get_train_gym_config(env_name=env_name, seed=np.random.randint(100), num_processes=1, num_frames=1e6)
    # config['warm_model'] = os.path.join(config['save_dir'], str(env) + ".pt")
    pu.train_changing_gym_model(env, config, update_fun)


def construct_env(args):
    """Return the environment"""
    floor = construct_default_floor_plan()
    glb = np.array([0.1, 0.6, 0, 0])
    gub = np.array([0.4, 0.9, 0, 0])
    env = SensorRangeGoalProblem(floor, glb, gub)
    if args.novio:
        env.out_vio_step = 1
    env.x0lb[0] = 0.1
    env.x0lb[1] = 0.6
    env.x0ub[0] = 0.4
    env.x0ub[1] = 0.9

    def update_fun(env, alpha):
        env.glb[1] = 0.6 - 0.5 * alpha  # tends to 0.1
        env.gub[0] = 0.4 + 0.5 * alpha  # tends to 0.9
        env.x0lb[1] = env.glb[1]
        env.x0ub[0] = env.gub[0]
        return env

    return env, floor, update_fun


if __name__ == '__main__':
    main()
