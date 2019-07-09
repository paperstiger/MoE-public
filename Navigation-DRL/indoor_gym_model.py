#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
Here I will construct the indoor navigation problem in gym framework.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pyLib.all as pl

from cameraSimulator import LidarSimulator
from floorPlan import MancraftFloorPlan


class IndoorProblem(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, plan):
        """Initialize the problem using a floor plan. Many values are default."""
        self.dt = 0.1
        self.viewer = None
        self.dimx, self.dimu = 4, 2
        self.floor = plan
        N = 72
        fov = 2 * np.pi
        depth = 0.5
        self.camera = LidarSimulator(plan, N, fov, depth)
        self.camera_resolution = N
        self.camera_depth = depth

        def get_dist(state):
            return self.camera.observe_span(state[:2], 0, 0)

        self.get_dist = get_dist  # record a function
        self.xlb = np.array([0, 0, -np.inf, -np.inf])
        self.xub = np.array([1, 1, np.inf, np.inf])
        # self.x0lb = np.array([0, 0, 0, 0], dtype=float)
        # self.x0ub = np.array([1, 1, 0, 0], dtype=float)
        self.x0lb = np.array([0.6, 0.1, 0, 0], dtype=float)
        self.x0ub = np.array([0.6, 0.1, 0, 0], dtype=float)
        self.state = np.zeros(4)
        self.prev_state = np.zeros(4)
        self.start_min_dist = 0.05
        self.action_space = spaces.Box(-0.5 * np.ones(2, dtype=float), 0.5 * np.ones(2, dtype=float))
        self.observation_space = spaces.Box(self.xlb, self.xub)
        self.max_step = 50
        self._cur_step = 0
        self._cur_out_step = 0
        self.succeed_flag = 0  # records if any collision occurs
        self.done_state = None  # this is dangerous?
        self._noneed_reset = False

    def __str__(self):
        """Return a name"""
        return "IndoorProblem"

    def get_obs(self):
        """Observation, just return current state."""
        return self.state

    def step(self, a):
        """Move state forward given action."""
        self.prev_state[:] = self.state[:]
        self.state[:2] += self.dt * self.state[2:]
        self.state[2:] += self.dt * a
        cost, finish = self.getCost(self.state, a)
        self._cur_step += 1
        if self._cur_step == self.max_step:
            finish = 1
        if finish:
            self.done_state = self.state.copy()
            self._noneed_reset = False
        return self.get_obs(), -cost, finish, {}

    def reset(self):
        if self._noneed_reset:  # not done yet
            return self.get_obs()
        self._noneed_reset = True
        while True:
            x0 = np.random.uniform(self.x0lb, self.x0ub)
            if self.floor.get_point_distance(x0[:2]) > self.start_min_dist:
                break
        self.state[:] = x0
        self._cur_step = 0
        self._cur_out_step = 0
        return self.get_obs()

    def getCost(self, state, action):
        """Return a cost given state and action, a virtual function."""
        raise NotImplementedError("Subclass has to implement this method")

    def is_path_collision(self, state):
        """Check if current state comes from collision."""
        p1 = state[:2]
        p0 = p1 - state[2:] * self.dt
        return self.floor.check_straight_line_collision(p0, p1)

    def render(self, mode='human', close=False):
        """Draw the environment for visualization."""
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return None

        screen_width, screen_height = 500, 500
        scale = screen_width / (self.xub[0] - self.xlb[0])
        carlen, carwidth = 40.0/scale, 20.0/scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            print('Create rendering env now')
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(0, 1, 0, 1)
            # draw lines
            for line in self.multi_line:
                p1, p2 = line
                x = [p1[0], p2[0]]
                y = [p1[1], p2[1]]
                self.viewer.draw_line(x, y)
            circle = rendering.make_circle(radius=0.03)
            self.cartrans = rendering.Transform()
            circle.add_attr(self.cartrans)
            self.viewer.add_geom(circle)

        x = self.state[:2]
        self.cartrans.set_translation(x[0], x[1])
        # sys.stdout.write('\rx {} v {}'.format(x, v))
        # sys.stdout.flush()
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def render(self, mode='human', close=False):
        self._render(mode, close)


class SingleGoalProblem(IndoorProblem):
    """Implement a problem class that has a single goal and can start from everywhere."""
    def __init__(self, floor, goal):
        IndoorProblem.__init__(self, floor)
        if goal.size == 2:
            self.goal = np.concatenate((goal.flatten(), np.zeros(2)))
        else:
            self.goal = goal
        self._disable_collision = False
        self._disable_bound = False
        self.out_vio_step = 5

    def __str__(self):
        if self.out_vio_step == 1:
            return 'IndoorSingleGoalNoCollision-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-%.2f-%.2f' % (
                        self.x0lb[0], self.x0ub[1], self.x0lb[1], self.x0ub[1], self.goal[0], self.goal[1])
        else:
            return 'IndoorSingleGoal-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-%.2f-%.2f' % (
                        self.x0lb[0], self.x0ub[1], self.x0lb[1], self.x0ub[1], self.goal[0], self.goal[1])

    def disable_collision(self):
        """Disable collision check, maybe for debugging purpose."""
        self._disable_collision = True

    def disable_bound_check(self):
        self._disable_bound = True

    def getCost(self, state, action):
        """Calculate a cost that penalizes action and state deviation from goal."""
        diff = state - self.goal
        # cost = action.dot(action) + diff.dot(diff) + vel.dot(vel)  # the default cost
        cost = diff.dot(diff) + action.dot(action)
        diff_norm = np.linalg.norm(diff[:2])
        vel_mag = np.linalg.norm(diff[2:])
        finish = 0
        if diff_norm < 0.1:
            cost -= (0.1 - diff_norm) * 10
            if vel_mag < 0.1:
                cost -= 10 * (0.1 - vel_mag)
            if diff_norm < 0.05 and vel_mag < 0.05:
                print('!!!reach goal')
                cost -= 50
                finish = 1
                if self.succeed_flag == 0:
                    self.succeed_flag = 1
                return cost, finish
        if (not self._disable_bound) and (np.any(state > self.xub) or np.any(state < self.xlb)):
            print('out of bound at %d' % self._cur_step)
            cost += 30
            finish = 0
            self._cur_out_step += 1
            if self._cur_out_step == self.out_vio_step:
                finish = 1
            # self.state[:] = self.prev_state[:]  # reset to previous state to give another chance
            # self.state[:2] -= self.state[2:] * self.dt
            if self.succeed_flag == 0:
                self.succeed_flag = -1
            return cost, finish
        if (not self._disable_collision) and self.is_path_collision(state):
            print('collision at %d' % self._cur_step)
            cost += 30
            finish = 0
            self._cur_out_step += 1
            if self._cur_out_step == self.out_vio_step:
                finish = 1
            # self.state[:] = self.prev_state[:]  # reset to previous state to give another chance
            # self.state[:2] -= self.state[2:] * self.dt  # reason is that even if I reset, I cannot invert this
            if self.succeed_flag == 0:
                self.succeed_flag = -1
            return cost, finish
        return cost, finish


class RangeGoalProblem(SingleGoalProblem):
    """No sensor reading, but the goal is in a range."""
    def __init__(self, floor, glb, gub):
        self.glb = glb
        self.gub = gub
        SingleGoalProblem.__init__(self, floor, glb)
        obs_lb = np.concatenate((self.xlb, self.xlb[:2]))
        obs_ub = np.concatenate((self.xub, self.xub[:2]))
        self.obs = np.zeros_like(obs_lb)
        self.observation_space = spaces.Box(obs_lb, obs_ub)

    def reset(self):
        """Reset the system, I have to randomly generate goal, too."""
        self.goal = np.random.uniform(self.glb, self.gub)
        return SingleGoalProblem.reset(self)

    def __str__(self):
        if self.out_vio_step == 1:
            return 'IndoorRangeGoalNoCollision-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-(%.2f-%.2f)-(%.2f-%.2f)' % (
                    self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.glb[0], self.gub[0], self.glb[1], self.gub[1])
        else:
            return 'IndoorRangeGoal-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-(%.2f-%.2f)-(%.2f-%.2f)' % (
                    self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.glb[0], self.gub[0], self.glb[1], self.gub[1])

    def get_obs(self):
        """Observation, just return current state."""
        self.obs[:4] = self.state
        self.obs[4:] = self.state[:2] - self.goal[:2]
        return self.obs


class SensorSingleGoalProblem(SingleGoalProblem):
    """Implement this indoor navigation problem using sensor readings."""
    def __init__(self, floor, goal):
        SingleGoalProblem.__init__(self, floor, goal)
        self.obs = np.zeros(4 + self.camera_resolution)
        obs_lb = np.concatenate((np.zeros(self.camera_resolution), -np.ones(4)))
        obs_ub = np.concatenate((np.ones(self.camera_resolution) * self.camera_depth, np.ones(4)))
        self.observation_space = spaces.Box(obs_lb, obs_ub)

    def __str__(self):
        """Return a name for this problem."""
        if self.out_vio_step == 1:
            return 'IndoorSensorSingleGoalNoCollision-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-%.2f-%.2f' % (
                        self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.goal[0], self.goal[1])
        else:
            return 'IndoorSensorSingleGoal-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-%.2f-%.2f' % (
                        self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.goal[0], self.goal[1])

    def get_obs(self):
        """Observation, just return current state."""
        self.obs[:-4] = self.get_dist(self.state)
        self.obs[-4:] = self.goal - self.state
        return self.obs


class SensorRangeGoalProblem(SensorSingleGoalProblem):
    """In this class, the goal is changed whenever a new target is reset."""
    def __init__(self, floor, glb, gub):
        self.glb = glb
        self.gub = gub
        SensorSingleGoalProblem.__init__(self, floor, glb)

    def reset(self):
        """Reset the system, I have to randomly generate goal, too."""
        self.goal = np.random.uniform(self.glb, self.gub)
        obs = SensorSingleGoalProblem.reset(self)
        return obs

    def __str__(self):
        if self.out_vio_step == 1:
            return 'IndoorSensorRangeGoalNoCollision-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-(%.2f-%.2f)-(%.2f-%.2f)' % (
                    self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.glb[0], self.gub[0], self.glb[1], self.gub[1])
        else:
            return 'IndoorSensorRangeGoal-from-x=(%.2f-%.2f)-y=(%.2f-%.2f)-to-(%.2f-%.2f)-(%.2f-%.2f)' % (
                    self.x0lb[0], self.x0ub[0], self.x0lb[1], self.x0ub[1], self.glb[0], self.gub[0], self.glb[1], self.gub[1])


if __name__ == '__main__':
    main()
