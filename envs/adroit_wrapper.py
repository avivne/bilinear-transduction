import numpy as np
import pdb
import pickle
import torch

from mjrl.samplers.core import do_rollout
import mj_envs
# from mjrl.utils.gym_env import GymEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl(logpath):
    with open(logpath, 'rb') as input_file:
        pkl_data = pickle.load(input_file)
    return pkl_data

class AdroitWrapper():
    def __init__(self, env, env_name, post_process=False, post_process_idxs=None, demo_path=None):
        self.env = env
        self.env.env.env.env.relative_positions = False
        self.env.env.env.env.sample_start = False
        self.env_name = env_name
        self.post_process = post_process
        self.post_process_idxs = np.asarray(post_process_idxs)
        self.demo_path = demo_path

    # Wrap attributes
    def __getattr__(self, name):
        return getattr(self.env, name)

    # # Currently hardcoded for reach
    # def collect_data(self, pol=None, num_trajs=100, horizon=200, render=False, range_goal_low=None, range_goal_high=None, range_start=None):
    #     expert_policy = load_pkl(self.demo_path)
    #     self.env.env.env.env.relative_positions = True
    #     self.set_properties(sample_start=False, sample_goal=True, sample_goal_range_low=range_goal_low, sample_goal_range_high=range_goal_high)
    #     demos = do_rollout(num_trajs, self.env, expert_policy, eval_mode=True, horizon=horizon, render=render)  
    #     demos = self.reprocess_trajs(demos)
    #     self.env.env.env.env.relative_positions = False  
    #     return demos

    def set_goal(self, goal_pos):        
        self.env.env.env.env.goal_fixed = goal_pos
        self.env.env.env.env.sample_goal = False   
        o = self.env.reset()
        assert np.allclose(o[-3:], goal_pos), 'Goals are not set correctly'
        return o  
    
    def process_o(self, obs):
        return obs
    
    def sample_goals(self, range_goal_low, range_goal_high, size_sample):
        self.set_properties(False, True, range_goal_low, range_goal_high)
        goals = []
        for goal in range(size_sample):
            goal = np.array([self.env.env.env.env.np_random.uniform(low=self.env.env.env.env.sample_goal_range[0][0], high=self.env.env.env.env.sample_goal_range[0][1]), \
                    self.env.env.env.env.np_random.uniform(low=self.env.env.env.env.sample_goal_range[1][0], high=self.env.env.env.env.sample_goal_range[1][1]), \
                    self.env.env.env.env.np_random.uniform(low=self.env.env.env.env.sample_goal_range[2][0], high=self.env.env.env.env.sample_goal_range[2][1])])
            goals.append(goal)
        return np.array(goals)

    def set_properties(self, sample_start, sample_goal, sample_goal_range_low, sample_goal_range_high):
        self.env.env.env.env.sample_start = sample_start
        self.env.env.env.env.sample_goal = sample_goal
        sample_goal_range = list(zip(sample_goal_range_low, sample_goal_range_high))
        self.env.env.env.env.sample_goal_range = sample_goal_range        
    
    # Reprocess trajs to not have relative positions (policy does)
    def reprocess_trajs(self, trajs, reslice_dim=9):
        trajs_reprocessed = []
        for traj in trajs:
            traj_new = {'observations': [], 'actions': [], 'terminated': [], 'rewards':[]}
            traj_new['actions'] = traj['actions'].copy()
            traj_new['observations'] = np.concatenate([traj['observations'][:, :-reslice_dim], 
                                                    traj['env_infos']['obj_pos'], 
                                                    traj['env_infos']['palm_pos'],
                                                    traj['env_infos']['target_pos']], axis=-1)
            traj_new['terminated'] = traj['terminated']
            traj_new['rewards'] = traj['rewards']
            trajs_reprocessed.append(traj_new)

        # Array to get indexing to work
        trajs_reprocessed = np.asarray(trajs_reprocessed)

        # need: {'obs': [], 'action': [], 'next_obs': [], 'done': [], 'reward': []}
        # core/do_rollourt returns {'observations', 'actions', 'rewards', 'agent_infos', 'env_infos', 'terminated'}  
        new_trajs = []
        for traj in trajs_reprocessed:
            new_traj = {}
            new_traj['obs'] = traj['observations']
            new_traj['action'] = traj['actions']
            assert len(new_traj['obs'])== len(new_traj['action'])
            new_traj['next_obs'] = traj['observations'][1:] #do_rollout doesn't give the final one...
            new_traj['done'] = traj['terminated']
            new_traj['reward'] = traj['rewards']
            new_trajs.append(new_traj)
        return new_trajs

    @property
    def goal(self):
        return self.env.env.env.env.data.site_xpos[self.env.env.env.env.target_obj_sid].ravel()
    
    @property
    def max_path_length(self):
        return 200


