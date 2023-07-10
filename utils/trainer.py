import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from io import open
import random
import pdb

from utils.util import models_save


def train_policy(model_type, demos, policy, logdir, num_epochs=500, batch_size=32, horizon=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    idxs = np.array(range(len(demos)))
    num_batches = len(idxs) // batch_size
    train_deltas = np.array([])
    # Train the model with regular SGD
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()

            t1_idx = np.random.randint(len(demos), size=(batch_size,)) # Indices of first trajectory
            t2_idx = np.random.randint(len(demos), size=(batch_size,)) # Indices of second trajectory
            idx_pertraj = np.random.randint(horizon, size=(batch_size,)) # Indices of time step, same in both trajectories
            t1_idx_pertraj = t2_idx_pertraj = idx_pertraj
            t1_states = torch.Tensor(np.concatenate([demos[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])).float().to(device)
            t1_actions = torch.Tensor(np.concatenate([demos[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])).float().to(device)
            t2_states = torch.Tensor(np.concatenate([demos[c_idx]['obs'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t2_idx_pertraj)])).float().to(device)
            t2_actions = torch.Tensor(np.concatenate([demos[c_idx]['action'][t_idx][None] for (c_idx, t_idx) in zip(t2_idx, t2_idx_pertraj)])).float().to(device)

            if model_type == 'bc':
                a1_pred = policy(t1_states) #first action prediction
                loss = torch.mean(torch.linalg.norm(a1_pred - t1_actions, dim=-1))
            elif model_type == 'bilinear':
                deltas = t2_states - t1_states
                train_deltas = np.concatenate([train_deltas, deltas.cpu().detach().numpy()], axis=0) if train_deltas.size else deltas.cpu().detach().numpy()
                a2_pred = policy(t1_states, deltas) #input: [s,g],[ds,dg]
                loss = torch.mean(torch.linalg.norm(a2_pred - t2_actions, dim=-1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 0:
                print('[%d, %5d] loss: %.8f' %
                    (epoch + 1, i + 1, running_loss / 10.))
                losses.append(running_loss/10.)
                running_loss = 0.0
            losses.append(loss.item())
    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(logdir, model_type+'_losses.png'))
    print('Finished Training')
    return policy, np.array(train_deltas)


def train_supervised(model_type, dataset, policy, logdir, obs_idxs, type_idxs=None, num_epochs=500, batch_size=32, use_gt_weights=False, checkpoint_path=None):
    X, Y, X_mesh = dataset['train_X'], dataset['train_Y'], dataset['train_M']
    # type_idxs = list(set(list(range(X.shape[1]))) - set(obs_idxs))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    idxs = np.array(range(len(X)))
    num_batches = len(idxs) // batch_size
    train_deltas = []
    # Train the model with regular SGD
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()

            t1_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of mugA
            if use_gt_weights: #priviledged training (transformation type/object class type)
                # print('priviledged training')
                t2_idx = [random.choice(np.where(X[:,len(obs_idxs)+np.where(X[c_idx][type_idxs])[0]])[0]) for c_idx in t1_idx]
            else:
                t2_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of mugB

            t1_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t1_idx])).float().to(device)
            t1_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t1_idx])).float().to(device)
            t2_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t2_idx])).float().to(device)
            t2_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t2_idx])).float().to(device)

            if model_type == 'bc':
                y1_pred = policy(t1_X)
                loss = torch.mean(torch.linalg.norm(y1_pred - t1_Y, dim=-1))
            elif model_type == 'bilinear':
                deltas = t2_X - t1_X
                train_deltas.append(deltas.cpu().detach().numpy())
                y2_pred = policy(t1_X, deltas) #input: s,ds
                loss = torch.mean(torch.linalg.norm(y2_pred - t2_Y, dim=-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 0:
                print('[%d, %5d] loss: %.8f' %
                    (epoch + 1, i + 1, running_loss / 10.))
                losses.append(running_loss/10.)
                running_loss = 0.0
            losses.append(loss.item())
        if epoch % 10 == 0 and checkpoint_path:
            models_save(policy, logpath=osp.join(checkpoint_path, str(epoch)+'.pt'))
    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(logdir, model_type+'_losses.png'))
    print('Finished Training')

    return policy, np.array(train_deltas).reshape(-1,len(obs_idxs))
