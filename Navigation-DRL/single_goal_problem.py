#! /usr/bin/env python

"""
Here we test a problem of avoiding three d obstacle with simple dynamics.

Maybe I could use simpler dynamics
"""
import copy
import glob
import os
import sys
import time

import numpy as np
import numba
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import ppo_util as pu
from indoor_gym_model import SingleGoalProblem
from floorPlan import construct_default_floor_plan

import pyLib.all as pl


def main():
    # train means train a model with one obstacle
    args = pl.getArgs('train', 'show', 'novio', 'num1', 'masstest')
    if args.train:
        train_the_model(args)
    if args.show:
        show_the_model(args)
    if args.masstest:
        perform_massive_test(args)


def perform_massive_test(args):
    """Perform some simulation and see the results."""
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SingleGoalProblem(floor, goal)
    env.x0lb[:2] = [0.1, 0.1]
    env.x0ub[:2] = [0.9, 0.9]
    if args.novio:
        env.out_vio_step = 1
    env_name = str(env)
    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    assert sim_n == 1000
    v_x0, v_xf, v_flag = pu.policy_rollout(env, config, sim_n, show=False, return_success=True)
    mask0 = v_flag == 1
    collision = np.sum(v_flag == -1)
    print('succeed in %d / %d' % (np.sum(mask0), sim_n))
    print('collision in %d / %d' % (collision, sim_n))
    with open('succeed_log.txt', 'wt') as f:
        f.write('Env:%s' % env_name)
        f.write('Succeed: %d' % np.sum(mask0))
        f.write('Collision: %d' % collision)


def show_the_model(args):
    """Show the problem"""
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SingleGoalProblem(floor, goal)
    env.x0lb[:2] = [0.1, 0.1]
    env.x0ub[:2] = [0.9, 0.9]
    if args.novio:
        env.out_vio_step = 1
    env_name = str(env)

    config = pu.get_train_gym_config(env_name=env_name)
    sim_n = args.num
    v_x0, v_xf, v_traj = pu.policy_rollout(env, config, sim_n, False, True)
    # fig, ax = pld.get3dAxis()
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')
    fig, axes = pl.subplots(sim_n)
    for i in range(sim_n):
        sim_rst = v_traj[i]
        x, u, dt = sim_rst['state'], sim_rst['action'], sim_rst['dt']
        ax = axes[i]
        floor.draw(ax)
        circle1 = plt.Circle((0.8, 0.8), 0.05, color='g')
        ax.add_artist(circle1)
        ax.plot(*x[:, :2].T)
    fig.tight_layout()
    plt.savefig('gallery/%s-%d-cases.pdf' % (env_name, sim_n))
    plt.show()


def train_the_model(args):
    """Train a model for this particular problem."""
    # construct the fix world thing
    floor = construct_default_floor_plan()
    goal = np.array([0.8, 0.8])
    env = SingleGoalProblem(floor, goal)
    env.x0lb[:2] = [0.1, 0.1]
    env.x0ub[:2] = [0.9, 0.9]
    if args.novio:
        env.out_vio_step = 1
    # env.disable_collision()
    # env.disable_bound_check()
    env_name = str(env)
    config = pu.get_train_gym_config(env_name=env_name, seed=np.random.randint(10000))
    pu.train_a_gym_model(env, config)


if __name__ == "__main__":
    main()
