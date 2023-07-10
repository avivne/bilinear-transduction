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

from utils.util import make_env, \
                       models_save, \
                       define_policy, define_transducer, \
                       eval_supervised, save_pkl, load_pkl
from utils.trainer import train_supervised

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_supervised_training_and_eval(config_name='configs/grasping.yaml', logdir='log', model_type='bilinear'):
    yaml = YAML()
    v = yaml.load(open(config_name))    

    # Environment 
    seed = v['env']['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)    
    env_name = v['env']['env_name']
    env = make_env(env_name, v['env'])

    # Model
    feature_dim = v['model']['feature_dim']
    batch_size = v['model']['batch_size']
    num_epochs = v['model']['num_epochs']
    use_gt_weights_train = v['model']['use_gt_weights_train']
    use_gt_weights_eval = v['model']['use_gt_weights_eval']
    transducer_test_mode = v['model']['transducer_mode']
    hidden_layer_size = v['model']['hidden_layer_size']
    hidden_depth = v['model']['hidden_depth']
    embedding_size = v['model']['feature_dim']

    logdir = osp.join(logdir, env_name, f'{model_type}_lsize{str(hidden_layer_size)}_lnum{str(hidden_depth)}_embed{str(embedding_size)}')
    os.makedirs(logdir, exist_ok=True)
    print('logdir', logdir) 
    #if file exists will add to it and not overwrite it
    if not os.path.isfile(osp.join(logdir, 'config.txt')):
        with open(osp.join(logdir, 'config.txt'), 'a') as f:
            json.dump(v, f, indent=2)


    obs_idxs = v['env']['obs_idxs']
    type_idxs = v['env']['type_idxs']
    x_size = len(obs_idxs)
    y_size = v['env']['output_size']

    # Data
    samples = load_pkl(osp.join('data', env_name, 'demos.pkl'))

    # Model
    """
    model_type f(x)=y
    x=[s,g], dx=x-x'
    - bc on x
    - bilinear on x,dx
    """
    predictor = define_policy(model_type, x_size, y_size, None, hidden_layer_size, feature_dim, hidden_depth)
    print(predictor)
    predictor_path = osp.join(logdir, model_type+'.pt')
    train_deltas_path = osp.join(logdir, model_type+'_train_deltas'+'.pkl')
    #train and save
    predictor, train_deltas = train_supervised(model_type, samples, predictor, logdir, obs_idxs, type_idxs, num_epochs, batch_size, use_gt_weights_train)
    # save train_deltas for further evaluation. if transducer uses approx on traj - save that??
    # save learned models in logdir for later evaluation
    models_save(predictor, logpath=predictor_path)
    save_pkl(train_deltas, logpath=train_deltas_path)
    
    transducer_deltas_path = osp.join(logdir, model_type+'_transducer_train_deltas'+'.pkl')
    #transducer might approximate/sample train deltas                                                        
    # define transducer this is only used at test time (train is all to all)
    test_transducer = define_transducer(transducer_test_mode, samples, env, train_deltas, sample_deltas=True, \
                                            obs_idxs=obs_idxs, type_idxs=type_idxs)
    #save approx deltas used in eval
    save_pkl(test_transducer.train_deltas, logpath=transducer_deltas_path)
    
    # Eval
    predictor.eval()
    #eval in dist       
    plt.figure()
    eval_samples = eval_supervised(model_type, predictor, \
                            {'test_X': samples['eval_X'], 'X_params': samples['eval_X_params'], 'test_Y': samples['eval_Y'], 'test_M': samples['eval_M']}, \
                            obs_idxs, env, transducer=test_transducer, save_path=osp.join(logdir, model_type+'_in_dist.png'), \
                            use_gt_weights=use_gt_weights_eval)
    save_pkl(eval_samples, logpath=osp.join(logdir, model_type+'_eval_in_dist'+'.pkl'))
    #eval ood
    plt.figure()
    eval_samples_ood = eval_supervised(model_type, predictor, \
                            {'test_X': samples['ood_X'], 'X_params': samples['ood_X_params'], 'test_Y': samples['ood_Y'], 'test_M': samples['ood_M']}, \
                            obs_idxs, env, transducer=test_transducer, save_path=osp.join(logdir, model_type+'_ood.png'), \
                            use_gt_weights=use_gt_weights_eval)
    save_pkl(eval_samples_ood, logpath=osp.join(logdir, model_type+'_eval_ood'+'.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='bilinear')
    parser.add_argument('--config-name', default='configs/grasping.yaml')
    args = parser.parse_args()
    run_supervised_training_and_eval(config_name=args.config_name, model_type=args.model_type)
    