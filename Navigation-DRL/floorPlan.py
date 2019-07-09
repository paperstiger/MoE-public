#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
planarSolver.py

A solve that solves indoor navigation problem hopefully reliably.
"""
from __future__ import print_function, division
import sys, os, time
import itertools
import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, LinearRing, MultiLineString


def construct_default_floor_plan():
    """Construct the 3-room floor plan which is not symmetric, 3 room 2 door"""
    poly1 = Polygon([(1, 0), (1, 1), (0.5, 1), (0.5, 0)])
    poly2 = Polygon([(0.5, 1), (0, 1), (0, 0.5), (0.5, 0.5)])
    poly3 = Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])
    door1 = (np.array([0.5, 0.65]), np.array([0.5, 0.85]))
    door2 = (np.array([0.15, 0.5]), np.array([0.35, 0.5]))
    return MancraftFloorPlan([poly1, poly2, poly3], [door1, door2])


class MancraftFloorPlan(object):
    """An extended FloorPlan class which decompose all floors when being created.
    However, we are unable to control the decomposition so as a result, we might have to sample close to boundaries.
    """
    def __init__(self, lst_poly, lst_door):
        """Construct the environment using a list of polygons and a list of lines."""
        self.cvx_polys = lst_poly
        self.polygons = lst_poly
        self.doors = lst_door
        self.convert_to_multi_line()

    @property
    def num_poly(self):
        """Return the number of polygons."""
        return len(self.cvx_polys)

    @property
    def num_door(self):
        """Return the number of doors."""
        return len(self.doors)

    def convert_to_multi_line(self):
        """Since we may have doors, I want to convert polygon with doors into MultiLineString"""
        lines = []
        door_line = MultiLineString([LineString(pnt_pair) for pnt_pair in self.doors])
        exterior_line = MultiLineString([poly.exterior for poly in self.cvx_polys])
        self.multi_line = exterior_line.symmetric_difference(door_line)
        return self.multi_line

    def check_straight_line_collision(self, p0, p1, with_dist=False):
        """Check if a path from p0 to p1 has collision, also return shortest distance."""
        path_line = LineString([p0, p1])
        is_collision = path_line.intersects(self.multi_line)
        if not with_dist:
            return is_collision
        else:
            dist = path_line.distance(self.multi_line)
            return is_collision, dist

    def locate_room(self, pnt):
        """Locate which room a point lies in."""
        flags = np.array([np.all(poly.A.dot(pnt) <= poly.b) for poly in self.polygons])
        if not np.any(flags):  # it is not in any room
            return -1
        else:
            return np.where(flags)[0][0]  # for repeat one, simply use the first one

    def get_point_distance(self, pnt):
        """Calculate the distance of a point to any wall"""
        pnt = Point(pnt)
        if not hasattr(self, 'multi_line'):
            self.convert_to_multi_line()
        return pnt.distance(self.multi_line)

    def draw(self, ax=None, show=False):
        """Draw a figure showing it, also for debugging"""
        from geomutil import plot_line, plot_poly
        if ax is None:
            fig, ax = plt.subplots()
        for poly in self.cvx_polys:
            plot_poly(ax, poly)
        for door in self.doors:
            plot_line(ax, door, color='w')
        if show:
            plt.show()
        return ax


def main():
    floor = construct_default_floor_plan()
    floor.draw(show=True)


if __name__ == '__main__':
    main()
