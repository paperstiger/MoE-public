#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""

"""
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from shapely.geometry import Point, Polygon, LineString, LinearRing, MultiLineString
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import shortest_path


def plot_line(ax, ob, color='r', **kw):
    if isinstance(ob, tuple):
        p1, p2 = ob
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
    else:
        x, y = ob.xy
    ax.plot(x, y, color=color, alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, **kw)


def plot_poly(ax, poly, color='b'):
    polyline = LinearRing(list(poly.exterior.coords))
    plot_line(ax, polyline, color)


def plot_sol(ax, sol, **kwargs):
    return ax.plot(sol[1::2], sol[2::2], **kwargs)[-1]


def plot_all(*args):
    """A useful function to plot all we care about."""
    fig, ax = plt.subplots()
    for arg in args:
        if isinstance(arg, np.ndarray):
            ax.plot(arg[:, 0], arg[:, 1])
        elif isinstance(arg, Polygon):
            plot_poly(ax, arg)
        elif isinstance(arg, LineString):
            plot_line(ax, arg)
        else:
            print('Cannot do anything for type {}'.format(type(arg)))
    plt.show()


def findPolyTopology(v_poly, draw=False):
    """For a list of polygons, find the topology.

    :param v_poly: a list of polygons
    :param draw: bool, if draw the intersection for debugging purpose
    """
    n_poly = len(v_poly)
    mat = dok_matrix((n_poly, n_poly), dtype=np.float)
    for i in range(n_poly):
        for j in range(i + 1, n_poly):
            intersec = v_poly[i].intersection(v_poly[j])
            if isinstance(intersec, LineString):
                if draw:
                    fig, ax = plt.subplots()
                    plot_poly(ax, v_poly[i], 'r')
                    plot_poly(ax, v_poly[j], 'g')
                    plot_line(ax, intersec, 'b')
                    plt.show()
                mat[i, j] = 1
                mat[j, i] = 1
    return mat


def isPathPolyIntersect(path, poly):
    """Given a path in the form of np.ndarray, return if it intersects with a given polygon.

    :param path: ndarray, (\*, 2) a path represented by N by 2 matrice
    """
    line = LineString(path)
    intersect = line.intersects(poly)
    print(intersect)
    if isinstance(intersect, LineString):
        return True
    else:
        return False


def locatePointAmongPolygon(point, vpoly):
    """Given a point and a list of polygons, locate the point"""
    if isinstance(point, np.ndarray):
        point = Point(point[0], point[1])
    for i, poly in enumerate(vpoly):
        if point.within(poly):
            return i
    # else we use distance and check distance is sufficient small
    vdist = [point.distance(poly) for poly in vpoly]
    i = np.argmin(vdist)
    if vdist[i] < 1e-4:
        return i
    assert False


def find_shortest_path(mat, i, j):
    """Given a csgraph, and initial and final node, return the shortest path.

    :param mat: a sparse matrix representing a graph
    :param i: int, index of starting node
    :param j: int, index of final node
    """
    if i == j:
        return [i]
    dist, m = shortest_path(mat.tocsr(), directed=False, return_predecessors=True, unweighted=True)
    path = [j]
    while True:
        if m[i, path[-1]] == -9999:
            print('graph is not connected from %d to %d' % (i, j))
            return None
        path.append(m[i, path[-1]])
        if path[-1] == i:
            break
    return path[::-1]  # so now it is path from i to j


def extractPathPolygonIntersect(path, poly, show=False):
    """Given a path and a polygon, find all the intersection.

    :param path: ndarray, (\*, 2), the path
    :param poly: polygon
    """
    path_line = LineString(path)
    inter_sec_line = path_line.intersection(poly)
    if show:
        fig, ax = plt.subplots()
        plot_poly(ax, poly)
        plot_line(ax, path_line)
        plt.show()
    print(type(inter_sec_line))
    if not isinstance(inter_sec_line, LineString):
        print('multiline has %d lines' % len(inter_sec_line.geoms))
        geoms = list(inter_sec_line.geoms)
        p0 = list(geoms[0].coords)[0]
        pf = list(geoms[-1].coords)[-1]
        ext_path = np.array([p0, pf])
        print(ext_path)
    else:
        # ext_path = np.array(list(inter_sec_line.coords))
        ext_path = np.array(list(inter_sec_line.coords))[[0, -1], :]
    return ext_path
