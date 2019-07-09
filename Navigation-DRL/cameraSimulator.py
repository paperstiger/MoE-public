#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
cameraSimulator.py

Simulate a camera in an environment.
This should be straight forward since we have the environment in shapely
"""
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point


class LidarSimulator:
    """A simulator that mimics a lidar in the indoor environment.

    It use shapely to find intersection point, calculate distance
    """
    def __init__(self, space, N, fov, depth):
        """Constructor for class.

        :param space: a MancraftFloorPlan object
        :param N: int, resolution of observation
        :param fov: float, radian, the field of view
        :param depth: float, observation depth
        """
        self.lines = space.convert_to_multi_line()
        self.N = N
        self.fov = fov
        self.depth = depth
        self._angle = np.linspace(-fov/2, fov/2, N)
        self.cos_angle = np.cos(self._angle)
        self.sin_angle = np.sin(self._angle)

    def check_line_collision(self, pnt0, pnt1):
        """Check if a line connecting pnt0 and pnt1 collides with the environment."""
        line = LineString([[pnt0[0], pnt0[1]], [pnt1[0], pnt1[1]]])
        result = line.intersection(self.lines)
        if isinstance(result, Point):
            return True
        else:
            if len(list(result.geoms)) == 0:
                return False
            else:
                print(list(result.geoms))
                return True

    def observe_span(self, center, angle, noise, return_point=False):
        """Get observation by a lidar spanning an angle range.

        :param center: ndarray, (2,), the starting point
        :param angle: float, the angle that the vehicle is facing towards
        :param noise: float, noise level added
        :param return_point: bool, if we also return the actual observation point
        :return: ndarray, (N,) the observation
        :return: ndarray, (N, 2) the observation points in real world
        """
        dist = np.array([self.observe_one(center, angle + _agl, noise) for _agl in self._angle])
        if not return_point:
            return dist
        else:
            points = np.zeros((self.N, 2))
            points[:, 0] = center[0] + dist * np.cos(angle + self._angle)
            points[:, 1] = center[1] + dist * np.sin(angle + self._angle)
            return dist, points

    def observe_one(self, center, angle, noise=0):
        """Get an observation. Return distance to the nearest point with noises.

        :param center: ndarray, (2,), the starting point
        :param angle: float, the angle of array
        :param noise: float, noise level added
        :return: float, depth
        """
        pnt0 = Point(center)
        pnt2 = np.array([center[0] + self.depth * np.cos(angle),
                        center[1] + self.depth * np.sin(angle)])
        line = LineString([center, pnt2])
        result = line.intersection(self.lines)
        if isinstance(result, Point):
            dist = pnt0.distance(result)
        else:
            dis = [pnt0.distance(thing) for thing in result.geoms]
            if len(dis) == 0:
                dist = self.depth
            else:
                dist = np.amin(dis)
        if noise > 0:
            return dist * (1 + noise * np.random.randn())
        else:
            return dist

    def get_local_boundary(self, dist):
        """Given observation of distance, recover the points in the local coordinate system.

        :param dist: ndarray, (N,) the depth observation
        :return: ndarray, (N, 2) the points
        """
        return np.c_[dist * self.cos_angle, dist * self.sin_angle]

    def get_world_boundary(self, center, angle, dist):
        """Given observation, recover the boundaries in the world coordinate.

        :param center: ndarray, (2,) the robot position
        :param angle: float, the direction of robot
        :param dist: ndarray, (N,) the depth observation
        :return: ndarray, (N, 2) the points in world coordinate
        """
        pntx = np.cos(angle + self._angle) * dist + center[0]
        pnty = np.sin(angle + self._angle) * dist + center[1]
        return np.c_[pntx, pnty]


def main():
    from floorPlan import construct_default_floor_plan
    space = construct_default_floor_plan()
    # create simulator
    sim = LidarSimulator(space, 72, 2*np.pi, 0.5)
    center = np.array([0.2, 0.3])
    angle = np.pi / 2
    # dis = sim.observe_one(center, angle)
    noise = 0
    obs, pnts = sim.observe_span(center, angle, noise, return_point=True)
    fig, ax = plt.subplots()
    space.draw(ax)
    ax.scatter(pnts[:, 0], pnts[:, 1], marker='*', color='r')
    ax.scatter(center[0], center[1], marker='o', color='g')
    fig.savefig('gallery/sensor.pdf')
    plt.show()


if __name__ == '__main__':
    main()
