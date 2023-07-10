"""Slide object to target. Object may have different mass"""

import os
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from gym import utils

class MultiBlockPushingEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip = 10):
        self.frame_skip = frame_skip
        self.obj_mass = 5.
        self.lower_bound_mass = 5.
        self.upper_bound_mass = 200.
        self.max_path_length = 200
        self.goal = -1.0
        MujocoEnv.__init__(self, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'slider.xml'), self.frame_skip)
        utils.EzPickle.__init__(self)

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat, #9
            self.sim.data.qvel.flat, #8
            np.array([self.obj_mass]) #1
        ]).reshape(-1)

    def reset_model(self, reset_args=None):
        curr_qpos = np.zeros_like(self.sim.data.qpos.flat)
        curr_qvel = np.zeros_like(self.sim.data.qvel.flat)
        self.set_state(curr_qpos, curr_qvel)
        self.sim.forward()
        self.obj_mass = np.random.uniform(self.lower_bound_mass, self.upper_bound_mass)
        self.set_obj_mass(self.obj_mass)
        return self.get_current_obs()

    def compute_reward(self):
        curr_pos = self.sim.data.qpos.copy()
        obj_pos = curr_pos[2].copy()        
        dist = np.linalg.norm(self.goal - obj_pos)
        bonus = 0.
        if dist < 0.05:
            bonus = 1.
        return -dist + bonus

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        reward = self.compute_reward()
        done = False
        info = {}
        return next_obs, reward, done, info

    def set_obj_mass(self, mass_val):
        obj_index = self.sim.model.body_names.index('object0')
        original_mass = np.array(self.sim.model.body_mass).copy()
        original_mass[obj_index] = mass_val
        self.sim.model.body_mass[:] = original_mass.copy()
        self.sim.forward()
        
    def set_lower_bound(self, lb):
        self.lower_bound_mass = lb
    
    def set_upper_bound(self, ub):
        self.upper_bound_mass = ub

    def sample_goals(self, train_range_goal_low, train_range_goal_high, size_sample):
        # sample masses
        return [np.random.uniform(train_range_goal_low, train_range_goal_high) for _ in range(size_sample)]

    def set_goal(self, mass):
        self.set_lower_bound(mass)
        self.set_upper_bound(mass)
        return self.reset()

    def process_o(self, obs):
        return obs


# def collect_data(self, env, num_trajs, range_start, range_goal_low, range_goal_high, model_path):
#     from stable_baselines3 import SAC    
#     model = SAC.load(model_path)        
#     trajs = []
#     obs = env.reset()
#     env.set_lower_bound(range_goal_low)
#     env.set_upper_bound(range_goal_high)
#     for traj_num in range(num_trajs):
#         traj = {'obs': [], 'action': [], 'reward': [], 'done': [], 'next_obs': [], 'images': []}
#         obs = env.reset()
#         for h in range(horizon):
#             action, _states = model.predict(obs, deterministic=True)
#             next_obs, reward, done, info = env.step(action)
#             traj['obs'].append(obs.copy())
#             traj['action'].append(action.copy())
#             traj['next_obs'].append(next_obs.copy())        
#             traj['reward'].append(reward)
#             traj['done'].append(done)
#             obs = next_obs.copy()
#         traj['obs'] = np.array(traj['obs'])
#         traj['action'] = np.array(traj['action'])
#         traj['next_obs'] = np.array(traj['next_obs'])
#         traj['reward'] = np.array(traj['reward'])       
#         traj['done'] = np.array(traj['done'])        
#         trajs.append(traj) 
    
#     return trajs