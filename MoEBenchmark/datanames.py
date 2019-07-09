#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
datanames.py

Get datanames
"""
from __future__ import print_function, division
import sys, os, time
from os.path import expanduser
import inspect


home = inspect.getfile(inspect.currentframe())


PENDULUM = {
        'file_path': os.path.join(home, 'data/pendulum/gridData.npz'),
        'valid_path': os.path.join(home, 'data/pendulum/validateSet1994.npz'),
        'script_path': os.path.join(home, 'data/pendulum/Script'),
        'cfg_path': os.path.join(home, 'data/pendulum/PenConfig.json'),
        'umap_path': os.path.join(home, 'data/pendulum/umapembed.npy'),
        'snn_path': os.path.join(home, 'data/pendulum/sol_2_300_75.pkl'),
        'x_name': 'x0',
        'y_name': 'Sol',
        'uniqueid': 'pen'
        }


VEHICLE = {
        'file_path': os.path.join(home, 'data/vehicle/loosedbfile_resolve.npz'),
        'valid_path': os.path.join(home, 'data/vehicle/validate.npz'),
        'script_path': os.path.join(home, 'GAO/OCPLearn'),
        'cfg_path': os.path.join(home, 'data/vehicle/CarConfig.json'),
        'umap_path': os.path.join(home, 'data/vehicle/umapembed.npy'),
        'snn_path': os.path.join(home, 'data/vehicle/sol_4_200_200_149.pkl'),
        'x_name': 'x0',
        'y_name': 'Sol',
        'uniqueid': 'car'
        }


DRONE = {
        'file_path': os.path.join(home, 'data/droneone/removedFloatPyData.npz'),
        'valid_path': os.path.join(home, 'data/droneone/validate.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/Script'),
        'cfg_path': os.path.join(home, 'data/droneone/droneConfig.json'),
        'umap_path': os.path.join(home, 'data/droneone/subset_embed.npy'),
        'x_name': 'x',
        'valid_x_name': 'obs',
        'y_name': 'tjf',
        'uniqueid': 'drone'
        }

DRONETWO = {
        'file_path': os.path.join(home, 'data/dronetwo/trajdata.npz'),
        'valid_path': os.path.join(home, 'data/dronetwo/validation_trajdata.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script'),
        'cfg_path': os.path.join(home, 'data/dronetwo/droneConfig.json'),
        'umap_path': os.path.join(home, 'data/dronetwo/subset_embed.npy'),
        'snn_path': os.path.join(home, 'data/dronetwo/snn_model_of_11_2000_2000_317.pt'),
        'x_name': 'obs',
        'y_name': 'tjf',
        'uniqueid': 'dronetwo'
        }
