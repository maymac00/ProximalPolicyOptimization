import numpy as np

from .Critic import Critic
from torch import nn
from ..layers import Linear
import torch as th
from torch.nn import functional as F

class ConvCritic(Critic):

    def __init__(self, o_size: int, h_size: int, h_layers: int, feature_map_extractor : nn.Sequential, sample_obs, **kwargs):
        # o_size is irrelevant for the ConvSoftmaxActor
        o_size = 1
        super().__init__(o_size, h_size, h_layers)

        if isinstance(feature_map_extractor, nn.Sequential):
            self.feature_map_extractor = feature_map_extractor
        else:
            raise ValueError("feature_map_extractor must be a list of nn.Modules or a nn.Sequential")

        # put channels first
        sample_obs = self.transform_obs(sample_obs)
        end_dims = self.feature_map_extractor(sample_obs).shape[1]

        # Add a flatten layer to the end of the feature map extractor if it is not already there
        if not isinstance(self.feature_map_extractor[-1], nn.Flatten):
            self.feature_map_extractor.add_module("flatten", nn.Flatten())

        # Change the input size of the linear layer
        self.fully_connected[0] = Linear(end_dims, h_size, act_fn='tanh')

    def forward(self, x):
        # We assume obs have 3D HWC on any order. Transform obs returns a 4D tensor with the shape (1, C, H, W)
        x = self.transform_obs(x)

        # Feature map extraction only supports 3D, 4D tensors
        if len(x.shape) == 5:
            original_shape = x.shape
            x = x.reshape(-1, *x.shape[2:])
            x = self.feature_map_extractor(x)
            x = x.reshape(original_shape[0], original_shape[1], -1)
            probs = super().forward(x).squeeze()
            return probs
        else:
            x = self.feature_map_extractor(x)
            return super().forward(x).squeeze()

    def transform_obs(self, obs):
        """
        Overwrittable method to transform the observation before feeding it to the network. Must return a 4D tensor with the shape (1, C, H, W)
        Should take into account 3D inputs (HWC) and 4D inputs (BCHW) and even 5D inputs (ESHWC) if necessary where E = episodes, S = steps.
        :param obs:
        :return:
        """
        if len(obs.shape) == 3:
            obs = obs.permute(2, 0, 1)
            obs = th.unsqueeze(obs, 0)
        elif len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif len(obs.shape) == 5:
            obs = obs.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError("Observation shape not supported")
        return obs


    # TODO: Freeze and unfreeze CNN layers