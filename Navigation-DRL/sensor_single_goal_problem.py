#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
Study the problem where sensor reading and target-current offset is used.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

import ppo_util as pu
from indoor_gym_model import SensorSingleGoalProblem
from floorPlan import construct_default_floor_plan

import pyLib.all as pl


def main():
    # train trains a model
    # show shows a model
    # harder samples initial states in a larger range
    # num specifies how many simulations is performed
    # masstest will not draw every figure but will calculate flags
    args = pl.getArgs('train', 'show', 'harder', 'num4', 'masstest', 'novio')
    if args.train:
        train_the_model(args)
    if args.show:
        show_the_model(args)
    if args.masstest:
        perform_mass_test(args)


def train_the_model(args):
    """Train a model for this particular problem."""
    # construct the fix world thing
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SensorSingleGoalProblem(floor, goal)
    if args.novio:
        env.out_vio_step = 1
    if args.harder:
        env.x0lb[0] = 0.1
        env.x0ub[0] = 0.9
        env.x0lb[1] = 0.1
        env.x0ub[1] = 0.9
    env_name = str(env)
    print('Environment is %s' % env_name)
    config = pu.get_train_gym_config(env_name=env_name, cuda=True, seed=np.random.randint(10000))
    pu.train_a_gym_model(env, config)


def show_the_model(args):
    """Show the problem"""
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SensorSingleGoalProblem(floor, goal)
    if args.novio:
        env.out_vio_step = 1
    if args.harder:
        env.x0lb[0] = 0.1
        env.x0ub[0] = 0.9
        env.x0lb[1] = 0.1
        env.x0ub[1] = 0.9
    env_name = str(env)
    print('Environment is %s' % env_name)

    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    v_x0, v_xf, v_traj = pu.policy_rollout(env, config, sim_n, False, True)
    plt.switch_backend('TkAgg')
    fig, axes = pl.subplots(sim_n)
    for i in range(sim_n):
        ax = axes[i]
        sim_rst = v_traj[i]
        x, u, dt = sim_rst['state'], sim_rst['action'], sim_rst['dt']
        floor.draw(ax)
        circle1 = plt.Circle((goal[0], goal[1]), 0.05, color='g')
        ax.add_artist(circle1)
        ax.plot(*x[:, :2].T)
    plt.savefig('gallery/%s-%d-cases.pdf' % (env_name, sim_n))
    plt.show()


def perform_mass_test(args):
    """Do a massive simulation on this problem."""
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SensorSingleGoalProblem(floor, goal)
    if args.novio:
        env.out_vio_step = 1
    env.x0lb[0] = 0.1
    env.x0ub[0] = 0.9
    env.x0lb[1] = 0.1
    env.x0ub[1] = 0.9
    env_name = str(env)
    env.out_vio_step = 1  # do this anyway
    print('Environment is %s' % env_name)

    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    v_x0, v_xf, v_flag = pu.policy_rollout(env, config, sim_n, show=False, return_success=True)
    succeed = np.sum(v_flag == 1)
    collision = np.sum(v_flag == -1)
    print('succeed in %d / %d' % (succeed, sim_n))
    print('collision in %d / %d' % (collision, sim_n))
    print('0.1:', np.sum(np.linalg.norm(v_xf - np.array([0.8, 0.8, 0, 0]), axis=1) < 0.1))
    with open('succeed_log.txt', 'at') as f:
        f.write('Env:%s' % env_name)
        f.write('Succeed: %d' % succeed)
        f.write('Collision: %d' % collision)
    return
    fig, ax = pl.subplots()
    plt.switch_backend('TkAgg')
    floor.draw(ax)
    ax.scatter(*v_x0[mask0, :2].T, label='Succeed')
    ax.scatter(*v_x0[~mask0, :2].T, label='Failure')
    ax.legend()
    fig.savefig('gallery/%s-massive-sim-x0.pdf' % env_name)
    plt.show()


if __name__ == "__main__":
    main()
