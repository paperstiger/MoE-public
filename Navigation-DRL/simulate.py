#!/usr/bin/env python

"""
In this file, I will randomly generate initial states, perform rollout on the learned model and statistically record
the final state that I could get.
I hope it is bad.
"""

import argparse
import os
import types

import numpy as np
import matplotlib.pyplot as plt
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs import make_env
from utils import update_current_obs


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--sim-n', type=int, default=1000,
                    help='number of simulation')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('-stoc', action='store_true', default=False,
                    help='enable stochasticity')
parser.add_argument('--stoc-level', type=float, default=0.1,
                    help='level of stochasticity, a normal is drawn using this variance')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='Dubin-v5',
                    help='environment to train on (default: Dubin-v2)')
parser.add_argument('--load-dir', default='./trained_models/ppo',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
args = parser.parse_args()


env = make_env(args.env_name, args.seed, 0, None, args.add_timestep)
env = DummyVecEnv([env])

actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))


if len(env.observation_space.shape) == 1:
    env = VecNormalize(env, ret=False)
    env.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    env._obfilt = types.MethodType(_obfilt, env)
    render_func = env.venv.envs[0].render
else:
    render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

v_state = []
v_state0 = []
for i in range(args.sim_n):
    print('\ni = %d' % i)

    current_obs = torch.zeros(1, *obs_shape)
    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)

    # render_func('human')
    obs = env.reset()
    v_state0.append(env.venv.envs[0].state.copy())
    update_current_obs(obs, current_obs, obs_shape, args.num_stack)

    while True:
        with torch.no_grad():
            value, action, _, states = actor_critic.act(current_obs,
                                                        states,
                                                        masks,
                                                        deterministic=True)
        cpu_actions = action.squeeze(1).cpu().numpy()
        if args.stoc:
            cpu_actions += args.stoc_level * np.random.normal(0, 1, cpu_actions.shape)
        # Obser reward and next obs
        obs, reward, done, _ = env.step(cpu_actions)

        if done[0]:
            v_state.append(env.venv.envs[0].done_state.copy())

        masks.fill_(0.0 if done else 1.0)

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs, current_obs, obs_shape, args.num_stack)

        if done[0]:
            break
        else:
            # render_func('human')
            pass

v_state = np.array(v_state)
v_state0 = np.array(v_state0)
if args.stoc:
    np.savez('%s_stoc_final_state.npz' % args.env_name, x0=v_state0, xf=v_state)
else:
    np.savez('%s_final_state.npz' % args.env_name, x0=v_state0, xf=v_state)

v_state[:, 2] = np.mod(v_state[:, 2] + np.pi, 2*np.pi) - np.pi

state_norm = np.linalg.norm(v_state, axis=1)
fig, ax = plt.subplots()
ax.hist(state_norm)
if args.stoc:
    fig.savefig('%s_stoc_sim_xf_norm.pdf' % args.env_name)
else:
    fig.savefig('%s_sim_xf_norm.pdf' % args.env_name)
plt.show()