import pdb
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from io import open
import random
from ruamel.yaml import YAML
import json
import argparse

from utils.util import make_env, plot_traj, \
                       data_load, \
                       models_save, \
                       define_policy, define_transducer, \
                       eval_policy, save_pkl, load_pkl
from utils.trainer import train_policy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_bilinear_training_and_eval(config_name='configs/pusher.yaml', logdir='log', model_type='bilinear'):
    yaml = YAML()
    v = yaml.load(open(config_name))

    # Environment
    seed = v['env']['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env_name = v['env']['env_name']
    env = make_env(env_name, v['env']) 
    obs_idxs = v['env']['obs_idxs'] #agent idxs
    plot_idxs = v['env']['plot_idxs']
    goal_idxs = v['env']['goal_idxs']    
    horizon = v['env']['horizon']    
    
    # Model
    hidden_layer_size = v['model']['hidden_layer_size']
    hidden_depth = v['model']['hidden_depth']    
    embedding_size = v['model']['feature_dim']
    batch_size = v['model']['batch_size']
    num_epochs = v['model']['num_epochs']
    transducer_test_mode = v['model']['transducer_mode']

    logdir = osp.join(logdir, env_name, f'{model_type}_lsize{str(hidden_layer_size)}_lnum{str(hidden_depth)}_embed{str(embedding_size)}')
    os.makedirs(logdir, exist_ok=True)    
    if not os.path.isfile(osp.join(logdir, 'config.txt')):
        with open(osp.join(logdir, 'config.txt'), 'a') as f:
            json.dump(v, f, indent=2)

    obs_size = len(v['env']['post_process_idxs']) # env.observation_space.shape[0]
    goal_size = len(v['env']['goal_idxs'])
    ac_size = env.action_space.shape[0]    

    # DATA
    demos, in_dist_eval_goals, ood_eval_goals = data_load(osp.join('data', env_name))
    # plot_traj(demos, plot_idxs, goal_idxs, save_path=osp.join(logdir, 'demos.png'), env=env)       
    print('loaded demos')

    # MODEL     
    """
    model_type f(x)=y
    x=[s,g], dx=x-x'
    - (baseline) behavior cloning on x
    - bilinear transduction on x,dx
    """
    #train and save policy and deltas
    policy = define_policy(model_type, obs_size, ac_size, goal_size, hidden_layer_size, embedding_size, hidden_depth)  
    print(policy) 
    policy_path = osp.join(logdir, model_type+'.pt')
    train_deltas_path = osp.join(logdir, model_type+'_train_deltas'+'.pkl')
    policy, train_deltas = train_policy(model_type, demos, policy, logdir, num_epochs, batch_size, horizon) 
    models_save(policy, logpath=policy_path)
    save_pkl(train_deltas, logpath=train_deltas_path)
    
    # define transducer this is only used at test time (train is all to all)
    test_transducer = define_transducer(transducer_test_mode, demos, env, train_deltas, goal_idxs=goal_idxs)
    #save approx deltas used in eval                                        
    save_pkl(test_transducer.train_deltas, logpath=osp.join(logdir, model_type+'_transducer_train_deltas'+'.pkl'))
    
    # EVAL
    # eval in dist   
    plt.figure()  
    eval_trajs = eval_policy(model_type, env, policy, in_dist_eval_goals, obs_idxs, plot_idxs, goal_idxs, \
                test_transducer, horizon, osp.join(logdir, model_type+'_in_dist.png'))
    save_pkl(eval_trajs, logpath=osp.join(logdir, model_type+'_eval_in_dist'+'.pkl'))       
    # eval ood
    plt.figure()  
    eval_trajs_ood = eval_policy(model_type, env, policy, ood_eval_goals, obs_idxs, plot_idxs, goal_idxs, \
                test_transducer, horizon, osp.join(logdir, model_type+'_ood.png'))
    save_pkl(eval_trajs_ood, logpath=osp.join(logdir, model_type+'_eval_ood'+'.pkl'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='bilinear')
    parser.add_argument('--config-name', default='configs/reach_metaworld.yaml') 
    args = parser.parse_args()
    run_bilinear_training_and_eval(config_name=args.config_name, model_type=args.model_type)

