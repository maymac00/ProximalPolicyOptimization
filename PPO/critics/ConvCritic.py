import copy

import numpy as np

from .Critic import Critic
from torch import nn
from ..layers import Linear
import torch as th
from torch.nn import functional as F

class ConvCritic(Critic):

    def __init__(self, o_size: int, h_size: int, h_layers: int, feature_map_extractor : nn.ModuleList, sample_obs, **kwargs):
        # o_size is irrelevant for the ConvSoftmaxActor
        o_size = 1
        super().__init__(o_size, h_size, h_layers)

        self.feature_map_extractor = feature_map_extractor

        # Add a flatten layer to the end of the feature map extractor if it is not already there
        if not isinstance(self.feature_map_extractor[-1], nn.Flatten):
            self.feature_map_extractor.add_module("flatten", nn.Flatten())

        if len(sample_obs.shape) == 2:
            sample_obs = th.unsqueeze(sample_obs, 0)
            sample_obs = th.unsqueeze(sample_obs, 0)
        elif len(sample_obs.shape) == 3:
            sample_obs = th.unsqueeze(sample_obs, 0)
        end_dims = self.feature_map_extraction(sample_obs).shape[1]

        # Change the input size of the linear layer
        self.fully_connected[0] = Linear(end_dims, h_size, act_fn='tanh')

    def regression_head(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
            x = F.leaky_relu(l(x))
        return self.output(x)

    def feature_map_extraction(self, x):
        for layer in self.feature_map_extractor:
            x = layer(x)
        return x

    def forward(self, x):
        # Feature map extraction only supports 3D, 4D tensors
        if len(x.shape) == 5:
            original_shape = x.shape
            x = x.reshape(-1, *x.shape[2:])
            x = self.feature_map_extraction(x)
            x = x.reshape(original_shape[0], original_shape[1], -1)
            probs = self.regression_head(x)
            return probs
        elif len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return self.regression_head(x)
        elif len(x.shape) == 3:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return self.regression_head(x)
        elif len(x.shape) == 4:
            x = self.feature_map_extraction(x)
            return self.regression_head(x)



    # TODO: Freeze and unfreeze CNN layers

class ConvCriticCat:
    def __init__(self, critic: ConvCritic, extra_info_shape):
        self.critic = critic
        layer_shape = (self.critic.fully_connected[0].in_features, self.critic.fully_connected[0].out_features)
        self.critic.fully_connected[0] = Linear(layer_shape[0] + extra_info_shape[0], layer_shape[1])

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.critic, item)

    def regression_head(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
            x = F.leaky_relu(l(x))
        return self.output(x)

    def forward(self, x, cat=None):
        cat = th.unsqueeze(cat, 0)
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        # Feature map extraction only supports 3D, 4D tensors
        if len(x.shape) == 5:
            original_shape = x.shape
            x = x.reshape(-1, *x.shape[2:])
            x = self.feature_map_extraction(x)
            x = x.reshape(original_shape[0], original_shape[1], -1)
            x = th.cat([x, cat.squeeze()], dim=-1)
            probs = self.regression_head(x)
            return probs
        elif len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.regression_head(x)
        elif len(x.shape) == 3:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.regression_head(x)
        elif len(x.shape) == 4:
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.regression_head(x)

    def update(self, b, optimizer, **kwargs):
        last_values = self(b['observations'], cat=b["extra_info"]).squeeze()
        for epoch in range(self.critic_epochs):
            values = self(b['observations'], cat=b["extra_info"]).squeeze()
            self._update(values, last_values, b, optimizer, log=epoch == self.critic_epochs - 1)

            last_values = copy.deepcopy(values.detach())
        return self.update_metrics

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
