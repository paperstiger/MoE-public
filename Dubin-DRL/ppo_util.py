#! /usr/bin/env python

"""
Here we test a problem of avoiding three d obstacle with simple dynamics.

Maybe I could use simpler dynamics
"""
import copy
import glob
import os
import time
import types
import json

import numpy as np
import torch

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import Policy
from storage import RolloutStorage
from utils import update_current_obs
from visualize import visdom_plot

import algo


def get_train_gym_config(fnm=None, **kw):
    """Generate configuration for a general training."""
    default_dict = {
                'seed': 1,
                'env_name': 'ND_Bug',
                'log_dir': '/tmp/gym',
                'log_interval': 10,
                'save_interval': 100,
                'save_dir': 'trained_models/ppo',
                'add_timestep': False,
                'num_processes': 4,
                'gamma': 0.99,
                'num_stack': 1,
                'recurrent_policy': False,
                'cuda': True,
                'vis': True,
                'vis_interval': 100,
                'algo': 'ppo',
                'clip_param': 0.2,
                'ppo_epoch': 4,
                'num_mini_batch': 32,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'lr': 1e-3,
                'eps': 1e-5,
                'max_grad_norm': 0.5,
                'use_gae': False,
                'tau': 0.95,
                'num_steps': 100,
                'num_frames': 1e6
            }
    if fnm is not None:
        with open(fnm) as f:
            fnm_kw = json.load(f)
        default_dict.update(fnm_kw)
    default_dict.update(kw)
    default_dict['cuda'] = default_dict['cuda'] and torch.cuda.is_available()
    if not default_dict['cuda']:
        print('WARN: Train model without GPU')
    return default_dict


def train_a_gym_model(env, config):
    """We train gym-type RL problem using ppo given environment and configuration"""
    torch.set_num_threads(1)

    seed = config.get('seed', 1)
    log_dir = config.get('log_dir', '/tmp/gym')
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 100)
    save_dir = config.get('save_dir', 'trained_models/ppo')
    add_timestep = config.get('add_timestep', False)
    num_processes = config.get('num_processes', 4)
    gamma = config.get('gamma', 0.99)
    num_stack = config.get('num_stack', 1)
    recurrent_policy = config.get('recurrent_policy', False)
    cuda = config.get('cuda', True)
    vis = config.get('vis', True)
    vis_interval = config.get('vis_interval', 100)
    env_name = config['env_name']
    save_step = config.get('save_step', None)
    if save_step is not None:
        next_save_step = save_step

    # clean the log folder, if necessary
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if vis:
        from visdom import Visdom
        port = config.get('port', 8097)
        viz = Visdom(port=port)
        win = None

    envs = [make_env(env, seed, i, log_dir, add_timestep)
            for i in range(num_processes)]

    if num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, recurrent_policy)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if cuda:
        actor_critic.cuda()

    clip_param = config.get('clip_param', 0.2)
    ppo_epoch = config.get('ppo_epoch', 4)
    num_mini_batch = config.get('num_mini_batch', 32)
    value_loss_coef = config.get('value_loss_coef', 0.5)
    entropy_coef = config.get('entropy_coef', 0.01)
    lr = config.get('lr', 1e-3)
    eps = config.get('eps', 1e-5)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    use_gae = config.get('use_gae', False)
    tau = config.get('tau', 0.95)
    num_steps = config.get('num_steps', 100)
    num_frames = config.get('num_frames', 1e6)

    num_updates = int(num_frames) // num_steps // num_processes

    agent = algo.PPO(actor_critic, clip_param, ppo_epoch, num_mini_batch,
                     value_loss_coef, entropy_coef, lr=lr,
                     eps=eps,
                     max_grad_norm=max_grad_norm)

    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(num_processes, *obs_shape)

    obs = envs.reset()
    update_current_obs(obs, current_obs, obs_shape, num_stack)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1])
    final_rewards = torch.zeros([num_processes, 1])

    if cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    def save_the_model(num=None):
        """num is additional information"""
        # save it after training
        save_path = save_dir
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        # A really ugly way to save a model to CPU
        save_model = actor_critic
        if cuda:
            save_model = copy.deepcopy(actor_critic).cpu()
        save_model = [save_model,
                      hasattr(envs, 'ob_rms') and envs.ob_rms or None]
        if num is None:
            save_name = '%s.pt' % env_name
        else:
            save_name = '%s_at_%d.pt' % (env_name, int(num))
        torch.save(save_model, os.path.join(save_path, save_name))

    start = time.time()
    for j in range(1, 1 + num_updates):
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape, num_stack)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % save_interval == 0 and save_dir != "":
            save_the_model()
            if save_step is not None:
                total_num_steps = j * num_processes * num_steps
                if total_num_steps > next_save_step:
                    save_the_model(total_num_steps)
                    next_save_step += save_step

        if j % log_interval == 0:
            end = time.time()
            total_num_steps = j * num_processes * num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         final_rewards.mean(),
                         final_rewards.median(),
                         final_rewards.min(),
                         final_rewards.max(), dist_entropy,
                         value_loss, action_loss))
        if vis and j % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, log_dir, env_name,
                                  'ppo', num_frames)
            except IOError:
                pass
    # finally save model again
    save_the_model()



def policy_rollout(env_in, config, sim_n, show=False, return_traj=False, stoc=False, exit_fun=None, valid_x0=None):
    """Perform a policy rollout for a environment, with a policy function.

    Parameters
    ----------
    env_in: a gym-compatible environment
    config: dict, a configuration dictionary
    sim_n: int, number of simulation
    show: bool, if we show the animation, if possible
    return_traj: bool, if we return the detailed trajectories
    stoc: bool, if we use stochastic policy
    exit_fun: callable, take state as arguments, if return True, we exit
    valid_x0: ndarray/None, the validation set to reset initial states
    Returns:
    v_state0: ndarray, matrix of initial states
    v_statef: ndarray, matrix of final states
    v_traj: list of dicts, trajectories from rollout, return only return_traj is True
    """
    seed = config['seed']
    add_timestep = config['add_timestep']
    num_stack = config['num_stack']
    env = make_env(env_in, seed, 0, None, add_timestep)
    env = DummyVecEnv([env])
    if valid_x0 is not None:
        sim_n = valid_x0.shape[0]

    actor_critic, ob_rms = \
                torch.load(os.path.join(config['save_dir'], config['env_name'] + ".pt"))

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

        def _ob_un_filt(self, obs):
            if self.ob_rms:
                return np.sqrt(self.ob_rms.var + self.epsilon) * obs + self.ob_rms.mean
            else:
                return obs

        env._obfilt = types.MethodType(_obfilt, env)
        env._ob_un_filt = types.MethodType(_ob_un_filt, env)
        render_func = env.venv.envs[0].render
    else:
        render_func = env.envs[0].render

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])

    v_statef = []
    v_state0 = []
    v_traj = []

    the_env = env_in  # otherwise use env.venv.envs[0]

    for i in range(sim_n):
        print('i = %d' % i)
        current_obs = torch.zeros(1, *obs_shape)
        states = torch.zeros(1, actor_critic.state_size)
        masks = torch.zeros(1, 1)

        if show:
            render_func('human')

        obs = env.reset()
        if valid_x0 is not None:
            un_flt_obs = the_env.reset(valid_x0[i])  # ugly hack to manually modify environment state on the fly
            obs = env._obfilt(un_flt_obs)
        v_state0.append(the_env.state.copy())
        update_current_obs(obs, current_obs, obs_shape, num_stack)

        vec_state = []
        vec_action = []
        vec_obs = []
        vec_state.append(the_env.state.copy())
        vec_obs.append(env._ob_un_filt(obs[0]))

        while True:
            with torch.no_grad():
                value, action, _, states = actor_critic.act(current_obs,
                                                            states,
                                                            masks,
                                                            deterministic=not stoc)
            cpu_actions = action.squeeze(1).cpu().numpy()
            # Obser reward and next obs
            obs, reward, done, _ = env.step(cpu_actions)

            vec_action.append(cpu_actions[0].copy())
            vec_state.append(the_env.state.copy())
            vec_obs.append(env._ob_un_filt(obs[0]))

            if done[0]:
                vec_state[-1] = the_env.done_state.copy()  # it seems it has been reset
                v_statef.append(the_env.done_state.copy())

            masks.fill_(0.0 if done else 1.0)

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            update_current_obs(obs, current_obs, obs_shape, num_stack)

            if done[0]:
                break
            else:
                if show:
                    render_func('human')
        if return_traj:
            v_traj.append({'state': np.array(vec_state, dtype=np.float32), 'action': np.array(vec_action, dtype=np.float32), 'obs': np.array(vec_obs, dtype=np.float32), 'dt': env_in.dt})

    v_statef = np.array(v_statef, dtype=np.float32)
    v_state0 = np.array(v_state0, dtype=np.float32)
    if not return_traj:
        return v_state0, v_statef
    else:
        return v_state0, v_statef, v_traj
