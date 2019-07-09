#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
No observation, just a range of goal.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

import ppo_util as pu
from indoor_gym_model import RangeGoalProblem
from floorPlan import construct_default_floor_plan

import pyLib.all as pl


def main():
    # train trains a model
    # show shows a model
    # harder samples initial states in a larger range
    # num specifies how many simulations is performed
    # masstest will not draw every figure but will calculate flags
    # final test the ultimate problem, the problem we want to solve
    args = pl.getArgs('train', 'show', 'num4', 'masstest', 'final')
    if args.train:
        train_the_model(args)
    if args.show:
        show_the_model(args)
    if args.masstest:
        perform_mass_test(args)
    if args.final:
        perform_final_test(args)


def perform_final_test(args):
    """Train a few on several seeds."""
    env, floor = construct_env()
    env.glb[:] = [0.1, 0.1, 0, 0]
    env.gub[:] = [0.9, 0.9, 0, 0]
    env.x0lb[0] = 0.1
    env.x0ub[0] = 0.9
    env.x0lb[1] = 0.1
    env.x0ub[1] = 0.9
    env_name = str(env)
    for i in range(10):
        use_env_name = env_name + '-(%d)' % i
        print('Environment is %s' % env_name)
        config = pu.get_train_gym_config(env_name=use_env_name, cuda=True, seed=np.random.randint(10000), num_processes=4, num_frames=1e7)
        pu.train_a_gym_model(env, config)


def construct_env():
    """Return the environment"""
    floor = construct_default_floor_plan()
    glb = np.array([0.8, 0.3, 0, 0])
    gub = np.array([0.8, 0.7, 0, 0])
    env = RangeGoalProblem(floor, glb, gub)
    env.x0lb[0] = 0.1  # xmin
    env.x0ub[0] = 0.9  # xmax
    env.x0lb[1] = 0.1  # ymin
    env.x0ub[1] = 0.9  # ymax
    return env, floor


def train_the_model(args):
    """Train a model for this particular problem."""
    # construct the fix world thing
    env, floor = construct_env()
    env_name = str(env)
    print('Environment is %s' % env_name)
    config = pu.get_train_gym_config(env_name=env_name, cuda=True, seed=np.random.randint(10000), num_processes=1)
    pu.train_a_gym_model(env, config)


def show_the_model(args):
    """Show the problem"""
    env, floor = construct_env()
    env_name = str(env)
    print('Environment is %s' % env_name)

    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    v_x0, v_xf, v_traj, v_goal = pu.policy_rollout(env, config, sim_n, show=False, return_traj=True, return_goal=True)
    plt.switch_backend('TkAgg')
    fig, axes = pl.subplots(sim_n)
    for i in range(sim_n):
        ax = axes[i]
        sim_rst = v_traj[i]
        goal = v_goal[i]
        x, u, dt = sim_rst['state'], sim_rst['action'], sim_rst['dt']
        floor.draw(ax)
        circle1 = plt.Circle((goal[0], goal[1]), 0.05, color='g')
        ax.add_artist(circle1)
        ax.plot(*x[:, :2].T)
    plt.savefig('gallery/%s-%d-cases.pdf' % (env_name, sim_n))
    plt.show()


def perform_mass_test(args):
    """Do a massive simulation on this problem."""
    env, floor = construct_env()
    env_name = str(env)
    env.out_vio_step = 1
    print('Environment is %s' % env_name)

    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    assert sim_n == 1000
    v_x0, v_xf, v_flag, v_goal = pu.policy_rollout(env, config, sim_n, show=False, return_success=True, return_goal=True)
    n_succeed = np.sum(v_flag == 1)
    n_collision = np.sum(v_flag == -1)
    print('succeed in %d / %d' % (n_succeed, sim_n))
    print('collision in %d / %d' % (n_collision, sim_n))
    print('0.1: ', np.sum(np.linalg.norm(v_xf - v_goal, axis=1) < 0.1))
    with open('succeed_log.txt', 'at') as f:
        f.write('Env:%s' % env_name)
        f.write('Succeed: %d' % n_succeed)
        f.write('Collision: %d' % n_collision)
    return
    floor.draw(ax)
    ax.scatter(*v_x0[mask0, :2].T, label='Succeed')
    ax.scatter(*v_x0[~mask0, :2].T, label='Failure')
    ax.legend()
    fig.savefig('gallery/%s-massive-sim-x0.pdf' % env_name)
    plt.show()


if __name__ == "__main__":
    main()
