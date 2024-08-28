from .SoftmaxActor import SoftmaxActor
from .filters import SoftmaxFilterI
from ..layers import Linear
import torch as th
import torch.nn as nn
import numpy as np
from ..utils import normalize


class ConvSoftmaxActor(SoftmaxActor):
    def __init__(self, o_size, a_size, h_size, h_layers, feature_map_extractor : nn.ModuleList, sample_obs, action_map=None, action_filter: SoftmaxFilterI =None , **kwargs):
        # o_size is irrelevant for the ConvSoftmaxActor
        o_size = 1
        super().__init__(o_size, a_size, h_size, h_layers, action_map, action_filter, **kwargs)

        if len(sample_obs.shape) == 2:
            sample_obs = th.unsqueeze(sample_obs, 0)
            sample_obs = th.unsqueeze(sample_obs, 0)
        elif len(sample_obs.shape) == 3:
            sample_obs = th.unsqueeze(sample_obs, 0)

        self.feature_map_extractor = feature_map_extractor

        # Add a flatten layer to the end of the feature map extractor if it is not already there
        if not isinstance(self.feature_map_extractor[-1], nn.Flatten):
            # Add a flatten layer to the end of the feature map extractor that outputs a 1D tensor
            self.feature_map_extractor.append(nn.Flatten())

        end_dims = self.feature_map_extraction(sample_obs).shape[1]
        print("Length of the feature vector: ", end_dims)

        # Change the input size of the linear layer
        self.fully_connected[0] = Linear(end_dims, h_size, act_fn='tanh')

    def get_action(self, x, action=None):
        if len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
        prob = self.forward(x)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy
    
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
            probs = super().forward(x).squeeze()
            return probs
        elif len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return super().forward(x).squeeze()
        elif len(x.shape) == 3:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return super().forward(x).squeeze()
        elif len(x.shape) == 4:
            x = self.feature_map_extraction(x)
            return super().forward(x).squeeze()

    # TODO: Freeze and unfreeze CNN layers
