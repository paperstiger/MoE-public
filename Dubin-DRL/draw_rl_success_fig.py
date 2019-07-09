#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
draw_rl_success_fig.py

Compare success rate from DRL and compare with MoE.
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
import pyLib.plot as pld


pld.setGlobalFontSize(14)


def main():
    fig, ax = plt.subplots()
    ax.axhline(0.999, ls='--', label='MoE')
    success = [1.0, 0.805, 0.095, 0]
    ax.plot([1, 2, 3, 4], success, marker='o', label='PPO')
    ax.annotate(r"$[5, 5, 0, 0]$-" + "\n" + r"$[10, 10, 0, 0]$", xy=(1, 1.0), xytext=(1, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate(r"$[0, 5, 0, 0]$-" + "\n" + r"$[10, 10, 0, 0]$", xy=(2, success[1]), xytext=(2.2, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate(r"$[-10, 5, 0, 0]$-" + "\n" + r"$[10, 10, 0, 0]$", xy=(3, success[2]), xytext=(2.8, 0.35), arrowprops=dict(arrowstyle="->"))
    ax.annotate(r"$[-10, -10, -\pi, -3.1]$-" + "\n" + r"$[10, 10, \pi, 3.1]$", xy=(4, success[3]), xytext=(3.08, 0.15), arrowprops=dict(arrowstyle="->"))
    ax.legend()
    ax.set_ylabel('Success Rate')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('Problem Difficulty')
    fig.tight_layout()
    fig.savefig('gallery/moe_drl_success.pdf')
    plt.show()


if __name__ == '__main__':
    main()
