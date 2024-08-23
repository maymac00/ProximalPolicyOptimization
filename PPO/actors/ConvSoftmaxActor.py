from .SoftmaxActor import SoftmaxActor
from .filters import SoftmaxFilterI
from ..layers import Linear
import torch as th
import torch.nn as nn
import numpy as np
from ..utils import normalize, ObsTransformer


class ConvSoftmaxActor(SoftmaxActor):
    def __init__(self, o_size, a_size, h_size, h_layers, feature_map_extractor : nn.Sequential, sample_obs, action_map=None, action_filter: SoftmaxFilterI =None , **kwargs):
        # o_size is irrelevant for the ConvSoftmaxActor
        o_size = 1
        super().__init__(o_size, a_size, h_size, h_layers, action_map, action_filter, **kwargs)

        if isinstance(feature_map_extractor, nn.Sequential):
            self.feature_map_extractor = feature_map_extractor
        else:
            raise ValueError("feature_map_extractor must be a list of nn.Modules or a nn.Sequential")

        # put channels firstç
        sample_obs = ObsTransformer.transform_obs(sample_obs)
        end_dims = self.feature_map_extractor(sample_obs).shape[1]
        print("Length of the feature vector: ", end_dims)

        # Add a flatten layer to the end of the feature map extractor if it is not already there
        if not isinstance(self.feature_map_extractor[-1], nn.Flatten):
            self.feature_map_extractor.add_module("flatten", nn.Flatten())

        # Change the input size of the linear layer
        self.fully_connected[0] = Linear(end_dims, h_size, act_fn='tanh')


    def forward(self, x):
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

    # TODO: Freeze and unfreeze CNN layers