import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import random
import pdb


class Transducer():
    def __init__(self, trajs, env, **kwargs):
        self.trajs = trajs
        self.env = env
        self.k = 1

    def choose_goal(self, query_goal):
        pass


class DeltaDistributionTransducer(Transducer):
    def __init__(self, trajs, env, train_deltas, goal_idxs, **kwargs):
        super().__init__(trajs=trajs, env=env)
        
        #compute deltas once from trajs during training
        # self.train_deltas = train_deltas[:,goal_idxs]

        #approx train deltas
        train_deltas = []
        for k in range(len(trajs)):
            for _ in range(k,len(trajs)):
                i, j = np.random.choice(range(len(trajs)), 2)
                delta = trajs[i]['obs'][0][goal_idxs] - trajs[j]['obs'][0][goal_idxs]
                train_deltas.append(delta)
        train_deltas = np.asarray(train_deltas)
        self.train_deltas = train_deltas

        self.train_goals = np.stack([demo['obs'][-1, goal_idxs] for demo in trajs])
        self.train_init = np.stack([demo['obs'][0] for demo in trajs])
        self.goal_idxs = goal_idxs
        self.demos = trajs 

    def choose_goal(self, curr_obs):
        """return train traj idx that gives a delta closest to train deltas
            curr_obs - s,g. s_0 cancels out"""
        
        # delta from curr g to every g in train
        curr_deltas = curr_obs[self.goal_idxs] - self.train_goals
        
        # compare current deltas to train deltas, choose closest
        delta_dists = [np.min(np.linalg.norm(train_d - self.train_deltas, axis=1)) for train_d in curr_deltas]

        # sample from deltas in dist
        closest_train_idx = random.choice(np.argsort(delta_dists)[:(len(self.demos)//10)])
        
        #best
        closest_train_idx = np.argsort(delta_dists)[0]
        
        return self.demos[closest_train_idx]['obs']


class DeltaDistributionTransducerSupervised(Transducer):
    def __init__(self, samples, env, train_deltas, sample_deltas, obs_idxs, type_idxs, **kwargs):
        super().__init__(trajs=samples, env=env)
       
        #compute deltas once from trajs during training
        self.train_X = samples['train_X']
        self.obs_idxs = obs_idxs
        self.samples = samples
        self.type_idxs = type_idxs

        # deltas that were actually used in training
        if sample_deltas:
            print('sampling train deltas transducer')
            self.train_deltas = train_deltas[random.sample(range(train_deltas.shape[0]), train_deltas.shape[0]//10)] #sample from train deltas
        else:
            self.train_deltas = train_deltas


        # #approx train deltas. NOT for priviledged training
        # train_deltas = []
        # for i in range(len(self.train_X)):
        #     for j in range(i,len(self.train_X)):
        #         delta = self.train_X[i][obs_idxs] - self.train_X[j][obs_idxs]
        #         train_deltas.append(delta)
        # train_deltas = np.asarray(train_deltas)
        # self.train_deltas = train_deltas


    def choose_goal(self, curr_obs, use_gt_weights=False, return_anchor=False):
        """return train traj idx that gives a delta closest to train deltas
            curr_obs - s,g. s_0 cancels out"""

        if use_gt_weights: # priviledged eval
            sample_idxs_of_type_obs = np.where(np.argwhere(self.train_X[:,self.type_idxs]) == np.argwhere(curr_obs[self.type_idxs])[0])[0]
        else:
            sample_idxs_of_type_obs = list(range(len(self.train_X)))
        
        #closest delta in dist
        curr_deltas = curr_obs[self.obs_idxs] - self.train_X[sample_idxs_of_type_obs][:,self.obs_idxs]
        delta_dists = [np.min(np.linalg.norm(train_d - self.train_deltas, axis=1)) for train_d in curr_deltas]
        # #sample from deltas in dist
        # anchor_idx = sample_idxs_of_type_obs[random.choice(np.argsort(delta_dists)[:(len(self.train_X)//10)])]
        # best delta
        anchor_idx = sample_idxs_of_type_obs[np.argsort(delta_dists)[0]]
        closest_obs = self.train_X[anchor_idx][self.obs_idxs]
        print('curr_obs ', curr_obs, ' anchor ', self.train_X[anchor_idx])
        
        if return_anchor:
            return closest_obs, anchor_idx
        return closest_obs
