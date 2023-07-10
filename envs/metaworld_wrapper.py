import numpy as np
import functools
import torch
# from metaworld.tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaworldWrapper():
    def __init__(self, env, env_name, post_process=False, post_process_idxs=None):
        self.env = env
        self.env_name = env_name
        self.post_process = post_process
        self.post_process_idxs = np.asarray(post_process_idxs)

    # Wrap attributes
    def __getattr__(self, name):
        return getattr(self.env, name)

    # def collect_data(self, pol=None, num_trajs=1000, horizon=100, render=False, range_goal_low=np.array([0, 0]), range_goal_high=np.array([5, 5]), range_start=None):
    #     # If policy is None, use the expert
    #     expert_mode = False
    #     if pol is None:
    #         pol = functools.reduce(lambda a,b : a if a[0] == self.env_name else b, test_cases_latest_nonoise)[1]
    #         expert_mode = True

    #     trajs = []
    #     obj_poses = []
    #     goal_poses = []
    #     for tn in range(num_trajs):
    #         self.env.reset()

    #         goal_pos = np.random.uniform(range_goal_low, range_goal_high, size=(len(range_goal_low),))
    #         o, obj_pos, goal_pos = self.env.reset_model_ood(goal_pos=goal_pos)
            
    #         if not expert_mode:
    #             o = self.process_o(o)

    #         traj = {'obs': [], 'action': [], 'next_obs': [], 'done': [], 'reward': [], 'tcp': None}
    #         # for _ in range(self.max_path_length):
    #         done = False
    #         # i = 0
    #         # while not done:
    #         for _ in range(100): #TODO
    #             if hasattr(pol,'get_action'):
    #                 ac = pol.get_action(o)
    #             else: 
    #                 t1s = torch.Tensor(o[None]).to(device)
    #                 ac = pol(t1s).cpu().detach().numpy()[0]
    #             no, r, done, info = self.env.step(ac)
    #             # i += 1
    #             if not expert_mode:
    #                 no = self.process_o(no)
    #             traj['obs'].append(o.copy())
    #             traj['action'].append(ac.copy())
    #             traj['next_obs'].append(no.copy())
    #             traj['done'].append(info['success'])
    #             traj['reward'].append(info['in_place_reward'])
    #             o = no
                
    #             if render == True:
    #                 self.env.render()

    #         traj['obs'] = np.array(traj['obs'])
    #         traj['action'] = np.array(traj['action'])
    #         traj['next_obs'] = np.array(traj['next_obs'])
    #         traj['done'] = np.array(traj['done'])
    #         traj['tcp'] = self.env.tcp_center #gripper at end of trajectory TODO not good indicator for all tasks
    #         trajs.append(traj)
    #         obj_poses.append(obj_pos)
    #         goal_poses.append(goal_pos)
        
    #     if expert_mode:
    #         trajs = self.post_process_func(trajs)
    #     return trajs

    def process_o(self, o):
        if self.post_process:
            o_proc = o[self.post_process_idxs].copy()
        else:
            o_proc = o.copy()
        return o_proc

    def post_process_func(self, trajs_unprocessed):
        trajs = []
        for traj in trajs_unprocessed:
            traj_new = {'obs': [], 'action': [], 'next_obs': []}
            if self.post_process:
                traj_new['obs'] = traj['obs'][:, self.post_process_idxs].copy()
                traj_new['action'] = traj['action'].copy()
                traj_new['next_obs'] = traj['next_obs'][:, self.post_process_idxs].copy()
            else:
                traj_new['obs'] = traj['obs'].copy()
                traj_new['action'] = traj['action'].copy()
                traj_new['next_obs'] = traj_new['next_obs'].copy()
            trajs.append(traj_new)
        return trajs

    def sample_goals(self, range_goal_low=np.array([0, 0]), range_goal_high=np.array([5, 5]), size_sample=10):
        return np.random.uniform(range_goal_low, range_goal_high, size=(size_sample, len(range_goal_low)))

    def set_goal(self, goal):
        o, obj_pos, goal_pos = self.env.reset_model_ood(goal_pos=goal)
        return o

    def get_goal(self, obs):
        return obs[-3:].copy()
