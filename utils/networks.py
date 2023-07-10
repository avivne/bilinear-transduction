import torch
import torch.nn as nn
from typing import TypeVar
from torch import nn
from abc import abstractmethod
import pdb


# Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        input_dim = obs_dim
        self.trunk = mlp(input_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred


class BilinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, feature_dim, hidden_depth):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        input_dim = obs_dim
        self.obs_trunk = mlp(input_dim, hidden_dim, feature_dim*act_dim, hidden_depth)
        self.delta_trunk = mlp(input_dim, hidden_dim, feature_dim*act_dim, hidden_depth)

    def forward(self, obs, deltas):
        ob_embedding = self.obs_trunk(obs)
        ob_embedding = torch.reshape(ob_embedding, (-1, self.act_dim, self.feature_dim)) #act_dim x feature_dim
        delta_embedding = self.delta_trunk(deltas)
        delta_embedding = torch.reshape(delta_embedding, (-1, self.feature_dim, self.act_dim)) #feature_dim x act_dim
        pred = torch.diagonal(torch.matmul(ob_embedding, delta_embedding), dim1=1, dim2=2)
        return pred
