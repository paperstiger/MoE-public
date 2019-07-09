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


home = expanduser('~')


PENDULUM = {
        'file_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/data/gridData.npz'),
        'valid_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/data/validateSet1994.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/Script'),
        'cfg_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/resource/PenConfig.json'),
        'umap_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/Script/data/umapembed.npy'),
        'snn_path': os.path.join(home, 'GAO/ExplicitMPC/pendulum/Script/models/sol_2_300_75.pkl'),
        'x_name': 'x0',
        'y_name': 'Sol',
        'uniqueid': 'pen'
        }


VEHICLE = {
        'file_path': os.path.join(home, 'GAO/OCPLearn/data/loosedbfile_resolve.npz'),
        'valid_path': os.path.join(home, 'GAO/OCPLearn/data/validate.npz'),
        'script_path': os.path.join(home, 'GAO/OCPLearn'),
        'cfg_path': os.path.join(home, 'GAO/ExplicitMPC/CarToy/resource/CarConfig.json'),
        'umap_path': os.path.join(home, 'GAO/OCPLearn/data/umapembed.npy'),
        'snn_path': os.path.join(home, 'GAO/OCPLearn/models/sol_4_200_200_149.pkl'),
        'x_name': 'x0',
        'y_name': 'Sol',
        'uniqueid': 'car'
        }


DRONE = {
        'file_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/Script/data/removedFloatPyData.npz'),
        'valid_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/data/validate.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/Script'),
        'cfg_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/resource/droneConfig.json'),
        'umap_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObstacle/Script/data/umap/subset_embed.npy'),
        'x_name': 'x',
        'valid_x_name': 'obs',
        'y_name': 'tjf',
        'uniqueid': 'drone'
        }

DRONETWO = {
        'file_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script/data/trajdata.npz'),
        'valid_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script/data/validation_trajdata.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script'),
        'cfg_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/resource/droneConfig.json'),
        'umap_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script/data/umap/subset_embed.npy'),
        'snn_path': os.path.join(home, 'GAO/ExplicitMPC/DroneObsObs/Script/models/snntune/snn_model_of_11_2000_2000_317.pt'),
        'x_name': 'obs',
        'y_name': 'tjf',
        'uniqueid': 'dronetwo'
        }

DRONEONE = {
        'file_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/Script/data/trajdata.npz'),
        'valid_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/Script/data/validation_trajdata.npz'),
        'script_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/Script'),
        'cfg_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/resource/droneConfig.json'),
        'umap_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/Script/data/umap/subset_embed.npy'),
        'snn_path': os.path.join(home, 'GAO/ExplicitMPC/DroneOne/Script/models/snntune/snn_model_of_7_500_500_317.pt'),
        'x_name': 'obs',
        'y_name': 'tjf',
        'uniqueid': 'droneone'
        }
