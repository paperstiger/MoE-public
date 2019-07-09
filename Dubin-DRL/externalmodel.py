#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
externalmodel.py

Define several user-defined models.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.integrate import odeint
import numpy as np
import numba
import sys


class oneDBug(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.dt = 0.1
        self.viewer = None
        self.dimx, self.dimu = 2, 1
        self.xlb = np.array([-3, -5])
        self.xub = np.array([3, 5])
        self.action_space = spaces.Box(-0.5*np.ones(self.dimu), 0.5*np.ones(self.dimu))
        self.observation_space = spaces.Box(self.xlb, self.xub)

    def _get_obs(self):
        # return self.state.copy()
        return np.clip(self.state, self.xlb, self.xub)

    def getCost(self):
        obs = self._get_obs()
        cost = np.sum(obs**2)
        return cost

    def _step(self, a):
        self.state[0] += self.dt * self.state[1]
        self.state[1] += self.dt * a[0]
        return self._get_obs(), -self.getCost(), False, None

    def _reset(self):
        self.state = np.random.uniform([-1, -1], [1, 1])
        return self._get_obs()

    def _render(self, mode='human', close=False):
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
            self.viewer.set_bounds(5*self.xlb[0], 5*self.xub[0], 5*self.xlb[0], 5*self.xub[0])

            l, r, t, b = -carlen/2, carlen/2, carwidth/2, -carwidth/2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

        x, v = self.state
        self.viewer.draw_line((0, -1), (0, 1))
        self.cartrans.set_translation(x, 0)
        sys.stdout.write('\rx {} v {}'.format(x, v))
        sys.stdout.flush()
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class NdimBug(gym.Env):
    """Implementation of a N-Dim bug that has simple dynamics but complicated environment"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, dim, dt=0.1, xbd=None, ubd=None, tfweight=1, Q=None, R=None, max_dt=10, x0bd=None, clip_obs=False, **kw):
        """Constructor for the class.

        Parameters
        ----------
        dim: int, dimension of space, state has 2 dim, control has 1 dim
        dt: float, integration time
        xbd: tuple of ndarray, bounds on states
        ubd: tuple of ndarray, bounds on control
        tfweight: float, the weight on time in cost function
        Q: ndarray, the lqr cost on state
        R: ndarray, the lqr cost on control
        max_dt: float, maximum time for one episode. Combine with dt for steps
        x0bd: tuple of ndarray / None, bounds on initial states
        clip_obs: if True, the observation is clipped to be inside xbd
        """
        self.dt = dt
        self.viewer = None
        self.dimx, self.dimu = 2 * dim, dim
        if xbd is not None:
            self.xlb, self.xub = xbd
        else:
            self.xlb = -np.ones(self.dimx, dtype=np.float32)
            self.xub = -self.xlb
        if ubd is not None:
            self.ulb, self.uub = ubd
        else:
            self.ulb = -np.ones(dim, dtype=np.float32)
            self.uub = np.ones(dim, dtype=np.float32)
        self.action_space = spaces.Box(self.ulb, self.uub)
        self.observation_space = spaces.Box(self.xlb, self.xub)
        self.state = np.zeros(self.dimx, dtype=np.float32)
        self.tfweight = tfweight
        if Q is None:
            self.Q = np.ones(self.dimx)
        else:
            self.Q = Q
        if R is None:
            self.R = np.ones(self.dimu)
        else:
            self.R = R
        self._cur_step = 0
        self.max_step = int(np.ceil(max_dt / dt))
        self.x0lb, self.x0ub = x0bd
        self.clip_obs = clip_obs
        self.done_state = None
        self._reset()

    def _get_obs(self):
        """Return an observation we have full state estimation"""
        if self.clip_obs:
            return np.clip(self.state, self.xlb, self.xub)
        else:
            return self.state

    def get_obs(self):
        return self._get_obs()

    def getCost(self, state, action):
        objQ = np.sum(state**2 * self.Q)
        objR = np.sum(action**2 * self.R)
        fixcost = self.tfweight  # hope this encourage quick response
        return (objQ + objR + fixcost) * self.dt

    def _step(self, a):
        self.state[:self.dimu] += self.dt * self.state[self.dimu:] + 0.5 * a * self.dt ** 2
        self.state[self.dimu:] += self.dt * a
        cost = self.getCost(self.state, a)
        state_norm = np.linalg.norm(self.state)
        finish = False
        if state_norm < 1:
            cost -= (1 - state_norm)
            if state_norm < 0.1:
                finish = True
                cost -= 5
        if self._cur_step >= self.max_step:
            cost += 5
            finish = True
        if finish:
            self.done_state = self.state.copy()
        self._cur_step += 1
        return self._get_obs(), -cost, finish, {}

    def step(self, a):
        return self._step(a)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def seed(self, seed=None):
        return self._seed(seed)

    def _reset(self):
        self.state = np.random.uniform(self.x0lb, self.x0ub)
        self._cur_step = 0
        return self._get_obs()

    def reset(self):
        return self._reset()

    def _render(self, mode='human', close=False):
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
            self.viewer.set_bounds(5*self.xlb[0], 5*self.xub[0], 5*self.xlb[0], 5*self.xub[0])

            l, r, t, b = -carlen/2, carlen/2, carwidth/2, -carwidth/2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

        x, v = self.state[:self.dimu], self.state[self.dimu:]
        self.viewer.draw_line((0, -1), (0, 1))
        self.cartrans.set_translation(x[0], x[1])
        sys.stdout.write('\rx {} v {}'.format(x, v))
        sys.stdout.flush()
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def render(self, mode='human', close=False):
        self._render(mode, close)


class DubinCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, x0=None, wrap=True, x0bd=None):
        self.dt = 0.1
        self.viewer = None
        # define problem dimension
        self.dimx, self.dimu = 4, 2
        self.drawxlb = np.array([-15.0, -15.0])
        self.drawxub = np.array([15.0, 15.0])
        self.xlb = np.array([-20, -20, -np.pi, -5.1])
        self.xub = np.array([20, 20, np.pi, 5.1])
        if x0 is None:
            if x0bd is None:
                self.x0lb = np.array([-10, -10, -np.pi, -3.1])
                self.x0ub = np.array([10, 10, np.pi, 3.1])
            else:
                self.x0lb, self.x0ub = x0bd
        else:
            self.x0lb = self.x0ub = x0
        # define costs
        self.Q = 0.1 * np.ones(self.dimx)
        self.Q[3] = 0.1
        self.Q[2] = 0.05
        self.R = 0.1 * np.ones(self.dimu)
        self.ulb = -5 * np.ones(self.dimu)
        self.uub = 5 * np.ones(self.dimu)
        self.tfweight = 1
        self.action_space = spaces.Box(self.ulb, self.uub)
        self.observation_space = spaces.Box(np.array([-100, -100, -1, -1, -10], dtype=np.float32),
                                            np.array([100, 100, 1, 1, 10], dtype=np.float32))
        self.default_state = np.array([10.0, 10.0, 0.0, 0.0])
        self.wrap = wrap
        # self._seed()
        self.viewer = None
        self.state = None
        self.done_state = None
        self.max_step = 100
        self.cur_step = 0
        self._reset()

    def getCost(self, state, action):
        cpstate = state.copy()
        cpstate[2] = np.mod(state[2] + np.pi, 2*np.pi) - np.pi
        objQ = np.sum(cpstate**2 * self.Q)
        objR = np.sum(action**2 * self.R)
        fixcost = self.tfweight  # hope this encourage quick response
        return (objQ + objR + fixcost) * self.dt

    def _step(self, action):
        # u = (self.uub - self.ulb)/2.0*action + (self.uub + self.ulb)/2.0
        u = action
        y = odeint(self.dyn, self.state, np.array([0.0, self.dt]), args=(u,))
        costs = self.getCost(self.state, u)
        self.state = y[-1]
        x, y, theta, v = self.state
        finish = 0
        # if np.abs(x) > 15 or np.abs(y) > 15 or np.abs(v) > 10 or np.abs(theta) > 4*np.pi:
        # if np.abs(x) > 15 or np.abs(y) > 15:  # or np.abs(yv) > 10:
        #    finish = 1
        #    costs = 10
        test_state = self.state.copy()
        test_state[2] = np.mod(test_state[2] + np.pi, 2*np.pi) - np.pi
        # print(test_state[2])
        state_norm = np.linalg.norm(test_state)
        if state_norm < 1:
            costs = -(1 - state_norm)  # give reward when close
            if state_norm < 0.1:
                finish = 1
                costs = -1
        self.cur_step += 1
        if self.cur_step == self.max_step:
            finish = 1
        if finish == 1:
            self.done_state = self.state.copy()
        return self._get_obs(), -costs, finish, {}

    def step(self, action):
        return self._step(action)

    def _get_obs(self):
        # agl = self.state[2]
        # outstate = np.clip(self.state, self.xlb, self.xub)
        # outstate[2] = agl
        # x, y, theta, v = outstate
        x, y, theta, v = self.state
        if self.wrap:
            return np.array([x, y, np.sin(theta), np.cos(theta), v])
        else:
            outtheta = np.mod(theta + np.pi, 2*np.pi) - np.pi
            return np.array([x, y, outtheta, v])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self, x0=None):
        if x0 is None:
            self.state = np.random.uniform(low=self.x0lb, high=self.x0ub)
        else:
            self.state = x0
        # self.state = self.np_random.uniform([-1, -5, -0.3, -0.5], [1, 5, 0.3, 0.5])
        # self.state = self.default_state + np.random.normal(size=self.dimx)*0.01
        self.cur_step = 0
        return self._get_obs()

    def reset(self, x0=None):
        return self._reset(x0)

    def render(self, mode='human', close=False):
        return self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width, screen_height = 500, 500
        scale = screen_width / (self.drawxub[0] - self.drawxlb[0])
        carlen, carwidth = 40/scale, 20/scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            print('Create rendering env now')
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(self.drawxlb[0], self.drawxub[0], self.drawxlb[1], self.drawxub[1])

            l, r, t, b = -carlen/2, carlen/2, carwidth/2, -carwidth/2
            # car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car = rendering.FilledPolygon([(l, b), (l, t), (r, 0)])
            car.set_color(.8, .3, .3)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            # targetcar = rendering.FilledPolygon([(l, b), (l, t), (r, 0)])
            # targetcar.set_color(0, 1, 0)
            # defaultcartrans = rendering.Transform()
            # targetcar.add_attr(defaultcartrans)
            # self.viewer.add_geom(targetcar)

        x, y, theta, v = self.state
        self.cartrans.set_rotation(-theta + np.pi/2)
        self.viewer.draw_line((-1, 0), (1, 0))
        self.viewer.draw_line((0, -1), (0, 1))
        # self.cartrans.set_translation(x*scale + screen_width/2, y*scale + screen_height/2)
        # self.cartrans.set_translation(screen_width/scale/2. + x, screen_height/scale/2. + y)
        self.cartrans.set_translation(x, y)
        # self.cartrans.set_translation(30, 30)
        sys.stdout.write('step {} x {} y {} theta {} v {}\n'.format(self.cur_step, x, y, theta, v))
        sys.stdout.flush()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def dyn(self, x, t0, u):
        sta = np.sin(x[2])
        cta = np.cos(x[2])
        v = x[3]
        return np.array([v*sta, v*cta, u[0]*v, u[1]])


def main():
    import time
    env = DubinCarEnv()
    env.seed(13)
    env.reset()
    print(env.state)
    for _ in range(1):
        action = np.random.normal(size=2)
        env.step(action)
        env.render()
        print(env.state)
        time.sleep(0.05)
    # raw_input("Press Enter to continue")
    #env = gym.make('Acrobot-v1')
    #env.reset()
    #for _ in xrange(100):
    #    # action = np.random.normal(size=1)
    #    action = np.random.randint(3)
    #    env.step(action)
    #    env.render()
    #    time.sleep(0.1)


if __name__ == '__main__':
    main()
