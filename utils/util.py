import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from io import open
import pdb
import os.path as osp
import pickle

from envs.metaworld_wrapper import MetaworldWrapper
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from envs.slider import MultiBlockPushingEnv
from mjrl.utils.gym_env import GymEnv
from envs.adroit_wrapper import AdroitWrapper
from envs.grasping import Grasp
from utils.transducers import *
from utils.networks import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def models_save(policy, logpath):
    torch.save(policy.state_dict(), logpath)


def models_load(model, loaddir):
    return model.load_state_dict(torch.load(loaddir))


def save_pkl(data, logpath):
    with open(logpath, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(logpath):
    with open(logpath, 'rb') as input_file:
        pkl_data = pickle.load(input_file)
    return pkl_data


def data_save(demos, in_dist_goals, ood_goals, logpath):
    with open(osp.join(logpath, 'demos.pkl'), 'wb') as f:
        pickle.dump(demos, f)
    with open(osp.join(logpath, 'in_dist_goals.pkl'), 'wb') as f:
        pickle.dump(in_dist_goals, f)
    with open(osp.join(logpath, 'ood_goals.pkl'), 'wb') as f:
        pickle.dump(ood_goals, f)


def data_load(loaddir):
    with open(osp.join(loaddir, 'demos.pkl'), 'rb') as input_file:
        demos = pickle.load(input_file)
    with open(osp.join(loaddir, 'in_dist_goals.pkl'), 'rb') as input_file:
        in_dist_eval_goals = pickle.load(input_file)
    with open(osp.join(loaddir, 'ood_goals.pkl'), 'rb') as input_file:
        ood_eval_goals = pickle.load(input_file)
    return demos, in_dist_eval_goals, ood_eval_goals


def make_env(env_name, env_params=None):
    if env_name == 'reach-v2' or env_name == 'push-v2':
        reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']
        env = reach_goal_observable_cls(seed=env_params['seed'])
        env.random_init = False
        env._freeze_rand_vec = False
        env = MetaworldWrapper(env=env, env_name=env_name, post_process=env_params['post_process'], post_process_idxs=env_params['post_process_idxs'])
    elif env_name == 'grasping':
        env = Grasp(env_params)
    elif env_name == 'slider':
        env = MultiBlockPushingEnv()
    elif env_name in ['adroit']:
        env = GymEnv('relocate-v0')
        env = AdroitWrapper(env=env, env_name=env_name, post_process=env_params['post_process'], post_process_idxs=env_params['post_process_idxs'])
    else:
        raise NotImplementedError('Env not implemented')
    return env


def plot_traj(trajs, plot_idxs, goal_idxs, save_path='plot_gcp.png', env=None):
    #TODO goal idxs, agent idxs (notice if post processing, need different idxs possibly)
    # Testing standard goal conditioned policy in distribution
    plot3D = len(plot_idxs) == 3
    plot2D = len(plot_idxs) == 2
    if plot3D:
        ax = plt.axes(projection='3d')
    colors = sns.color_palette('hls', len(trajs))
    end_dists = []
    for k in range(len(trajs)):
        traj = trajs[k]['obs']
        if plot3D:
            ax.plot3D(traj[:,plot_idxs[0]], traj[:,plot_idxs[1]], traj[:,plot_idxs[2]], color=colors[k], linestyle=':') #end effector traj
            ax.scatter3D(traj[-1,goal_idxs[0]], traj[-1,goal_idxs[1]], traj[-1,goal_idxs[2]], color=colors[k], marker='x') #gt goal
        elif plot2D:
            plt.plot(np.array(traj)[:,plot_idxs[0]], np.array(traj)[:, plot_idxs[1]], linestyle=':', color=colors[k])
            plt.scatter([traj[-1,goal_idxs[0]]], [traj[-1,goal_idxs[1]]], color=colors[k])
        else:
            plt.plot(np.array(traj)[:,plot_idxs[0]], linestyle=':', color=colors[k])  
        if not isinstance(env, MultiBlockPushingEnv):         
            end_dists.append(np.linalg.norm(traj[-1][plot_idxs] - traj[-1][goal_idxs[:len(plot_idxs)]]))
        else: #slider
            end_dists.append(np.linalg.norm(traj[-1][plot_idxs] - np.copy(env.goal))) #TODO
    end_dists = np.array(end_dists)
    plt.title(str(round(np.mean(end_dists),4)) + ' $\pm$ ' + str(round(np.std(end_dists),4)))
    plt.savefig(save_path)


def define_policy(model_type, obs_size, ac_size, goal_size, hidden_layer_size, feature_dim, hidden_depth):
    if model_type == 'bc':
        policy = Policy(obs_size, ac_size, hidden_layer_size, hidden_depth)
    elif model_type == 'bilinear':
        policy = BilinearPolicy(obs_size, ac_size, hidden_layer_size, feature_dim, hidden_depth)
    else:
        print('model_type', model_type)
        raise NotImplementedError('not implemented other policies')
    return policy


def define_transducer(transducer_mode, trajs, env, train_deltas, sample_deltas=False, obs_idxs=None, goal_idxs=None, type_idxs=None):
    if transducer_mode == 'delta':
        transducer = DeltaDistributionTransducer(trajs=trajs, env=env, train_deltas=train_deltas, goal_idxs=goal_idxs)
    elif transducer_mode == 'delta_supervised':
        transducer = DeltaDistributionTransducerSupervised(samples=trajs, env=env, train_deltas=train_deltas, sample_deltas=sample_deltas, obs_idxs=obs_idxs, type_idxs=type_idxs)
    else:
        raise NotImplementedError('not implemented this type of transducer yet')
    return transducer


def eval_policy(model_type, env, policy, eval_goals, obs_idxs, plot_idxs, goal_idxs, transducer, horizon, save_path):
    print('eval_policy', save_path)
    plot3D = len(plot_idxs) == 3
    plot2D = len(plot_idxs) == 2
    if plot3D:
        ax = plt.axes(projection='3d')
    plot_obs_key = 'obs'
    colors = sns.color_palette('hls', len(eval_goals))
    end_dists = []
    trajs = []
    for k in range(len(eval_goals)):
        print('eval ', k+1, '/', len(eval_goals))
        o = env.reset()
        end_pos = copy.deepcopy(eval_goals[k])
        o_latent = env.set_goal(copy.deepcopy(end_pos)) 
        o = env.process_o(o_latent)
        traj = {'obs': [], 'obs_latent':[], 'action': [], 'reward':[], 'info': [], 'anchor': [], 'images': []}
        if model_type in ['bilinear']:
            closest_traj_obs = transducer.choose_goal(o)
            traj['anchor'].append(closest_traj_obs)
        for i in range(horizon):
            obs = torch.Tensor(o[None]).to(device)
            #get delta
            if model_type in ['nn', 'bilinear']:
                train_obs = torch.Tensor(closest_traj_obs[i][None]).to(device)
                delta = obs - train_obs
            #eval with model
            if model_type in ['bc','linear']:
                ac = policy(obs).cpu().detach().numpy()[0]
            elif model_type == 'deepsets':
                ac = policy(obs[:,obs_idxs], obs[:,goal_idxs]).cpu().detach().numpy()[0]
            elif model_type == 'nn':
                nn_input = torch.cat([train_obs, delta], dim=-1)
                ac = policy(nn_input).cpu().detach().numpy()[0]
            elif model_type == 'bilinear':
                ac = policy(train_obs, delta).cpu().detach().numpy()[0]
            no_latent, r, done, info = env.step(ac)
            no = env.process_o(no_latent)
            traj['obs'].append(np.copy(o)) #state rep (latent or image)
            traj['obs_latent'].append(np.copy(o_latent)) #gt env latent rep with goal coors
            traj['action'].append(np.copy(ac))
            traj['reward'].append(r)
            traj['info'].append(info)
            o = np.copy(no)
            o_latent = np.copy(no_latent)
            # i += 1
        traj['obs'].append(np.copy(o)) #|obs|=|a|+1 (last state)
        traj['obs_latent'].append(np.copy(o_latent)) 
        traj['obs'] = np.array(traj['obs'])
        traj['obs_latent'] = np.array(traj['obs_latent'])
        trajs.append(traj)
        if plot3D:
            ax.plot3D(traj[plot_obs_key][:,plot_idxs[0]], traj[plot_obs_key][:,plot_idxs[1]], traj[plot_obs_key][:,plot_idxs[2]], color=colors[k], linestyle=':') #end effector traj
            ax.scatter3D(end_pos[0], end_pos[1], end_pos[2], color=colors[k], marker='x') #gt goal
        elif plot2D:
            plt.plot(np.array(traj[plot_obs_key])[:,plot_idxs[0]], np.array(traj[plot_obs_key])[:,plot_idxs[1]], linestyle=':', color=colors[k])
            plt.scatter([end_pos[0]],[end_pos[1]], color=colors[k], marker='x', s=20)
        else:
            plt.plot(np.array(traj[plot_obs_key])[:,plot_idxs[0]], linestyle=':', color=colors[k])
        if not isinstance(env, MultiBlockPushingEnv):
            end_dists.append(np.linalg.norm(traj['obs'][-1][plot_idxs] - end_pos[:len(plot_idxs)].copy())) #TODO
        else: #slider
            end_dists.append(np.linalg.norm(traj[plot_obs_key][-1][plot_idxs] - np.copy(env.goal)))
    end_dists = np.array(end_dists)
    print('Average end distance is %.5f'%(np.mean(end_dists)))
    print('Average traj len is %.5f'%(np.mean([len(traj[plot_obs_key]) for traj in trajs])))
    all_returns = [np.sum(traj['reward']) for traj in trajs]
    # further goals will accumulate lower reward. look at end dist.
    plt.title(str(round(np.mean(end_dists),4)) + ' $\pm$ ' + str(round(np.std(end_dists),4)))
    plt.savefig(save_path)
    return trajs


def eval_supervised(model_type, model, eval_dataset, obs_idxs, env, transducer=None, save_path=None, use_gt_weights=False):
    test_X, X_params, test_Y, test_M = eval_dataset['test_X'], eval_dataset['X_params'], eval_dataset['test_Y'], eval_dataset['test_M']
    plot3D = test_Y.shape[1] == 3
    if plot3D:
        ax = plt.axes(projection='3d')
    colors = sns.color_palette('hls', len(test_X))
    end_dists = []
    preds = {'preds': [], 'gt': test_Y, 'anchor_idxs': []}
    for k in range(len(test_X)):
        print('eval ', k+1, '/', len(test_X))
        obs = torch.Tensor(test_X[k][obs_idxs][None]).to(device)
        gt_output = test_Y[k]
        #get delta
        if model_type in ['nn', 'bilinear']:
            closest_obs, anchor_idx = transducer.choose_goal(test_X[k], use_gt_weights, return_anchor=True)
            preds['anchor_idxs'].append(anchor_idx)
            train_obs = torch.Tensor(closest_obs[None]).to(device)
            delta = obs - train_obs
        #eval with model
        if model_type in ['bc']:
            y = model(obs).cpu().detach().numpy()[0]
        elif model_type == 'bilinear':
            y = model(train_obs, delta).cpu().detach().numpy()[0]

        preds['preds'].append(y)
        if plot3D:
            ax.scatter3D(y[0], y[1], y[2], color=colors[k]) #end effector traj
            ax.scatter3D(gt_output[0], gt_output[1], gt_output[2], color=colors[k], marker='x') #gt goal
        else:
            plt.plot(np.array(traj['obs'])[:,plot_idxs[0]], np.array(traj['obs'])[:,plot_idxs[1]], linestyle=':', color=colors[k])
            plt.scatter([end_pos[0]],[end_pos[1]], color=colors[k], marker='x', s=20)
        end_dists.append(np.linalg.norm(gt_output - y))

        # #for k: plot mug, predicted grasp point (red), gt (blue)
        # img_dir = osp.join(osp.dirname(save_path), Path(save_path).stem)
        # if not os.path.isdir(img_dir):
        #     os.makedirs(img_dir)
        # env.plot_mug(test_X[k], X_params[k], y, gt_output, os.path.join(img_dir, str(k)))

    end_dists = np.array(end_dists)
    print('Average end distance is', str(round(np.mean(end_dists),4)) + ' $\pm$ ' + str(round(np.std(end_dists),4)))
    plt.title(str(round(np.mean(end_dists),4)) + ' $\pm$ ' + str(round(np.std(end_dists),4)))
    plt.savefig(save_path)
    preds['preds'] = np.array(preds['preds'])
    return preds


#point cloud normalization
def pcs_normalize(pcs, ys):
    """pcs: batch x N x 3"""

    def pc_normalize(pc):
        """pc: N x 3"""
        centroid = np.mean(pc, axis=0) #center the pointcloud
        pc = pc - centroid 
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) #normalize by largest norm of centered point
        pc = pc / m
        return pc
    
    def y_normalize(pc, y):
        centroid = np.mean(pc, axis=0) #center the pointcloud
        pc = pc - centroid 
        y = y - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) #normalize by largest norm of centered point
        return y / m

    norm_pcs = np.array([pc_normalize(pc) for pc in copy.deepcopy(pcs)])
    norm_ys = np.array([y_normalize(pc, y) for pc, y in zip(copy.deepcopy(pcs),ys)])
    return norm_pcs, norm_ys


def plot_pcd(samples, loaddir, d='train', id=0, save_pre=''):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(samples[f'{d}_M'][id][:,0], samples[f'{d}_M'][id][:,1], samples[f'{d}_M'][id][:,2], color='b', marker='x') #pcd
    ax.scatter3D(samples[f'{d}_Y'][id][0], samples[f'{d}_Y'][id][1], samples[f'{d}_Y'][id][2], color='r', marker='x') #grasp point
    plt.savefig(osp.join(loaddir, f'{save_pre}_{d}{id}.png'))