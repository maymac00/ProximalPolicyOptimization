import numpy as np

from .SoftmaxActor import SoftmaxActor
from .filters import SoftmaxFilterI
from ..layers import Linear
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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


    def get_action(self, x, action=None, *args, **kwargs):
        if len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
        prob = self.forward(x)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy
    
    def feature_map_extraction(self, x, cat=None):
        for layer in self.feature_map_extractor:
            x = layer(x)
        return x

    def classification_head(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
            x = th.relu(l(x))
        x = self.output(x)
        return F.softmax(x.to(th.float64), dim=-1)

    def forward(self, x):
        # Feature map extraction only supports 3D, 4D tensors
        if len(x.shape) == 5:
            original_shape = x.shape
            x = x.reshape(-1, *x.shape[2:])
            x = self.feature_map_extraction(x)
            x = x.reshape(original_shape[0], original_shape[1], -1)
            probs = self.classification_head(x)
            return probs
        elif len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return self.classification_head(x)
        elif len(x.shape) == 3:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            return self.classification_head(x)
        elif len(x.shape) == 4:
            x = self.feature_map_extraction(x)
            return self.classification_head(x)

    # TODO: Freeze and unfreeze CNN layers

# Wrapper class with th.cat after the feature map extraction

class ConvSoftmaxActorCat:

    def __init__(self, actor: ConvSoftmaxActor, extra_info_shape):
        self.actor = actor
        # Set fully connected layer to proper size
        layer_shape = (self.actor.fully_connected[0].in_features, self.actor.fully_connected[0].out_features)
        self.actor.fully_connected[0] = Linear(layer_shape[0] + extra_info_shape[0], layer_shape[1])

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.actor, item)

    def get_action(self, x, action=None, cat=None):
        if len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
        prob = self.forward(x, cat)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy

    def classification_head(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
            x = th.relu(l(x))
        x = self.output(x)
        return F.softmax(x.to(th.float64), dim=-1)

    def forward(self, x, cat=None):
        # Feature map extraction only supports 3D, 4D tensors
        cat = th.unsqueeze(cat, 0)
        if len(x.shape) == 5:
            original_shape = x.shape
            x = x.reshape(-1, *x.shape[2:])
            x = self.feature_map_extraction(x)
            x = x.reshape(original_shape[0], original_shape[1], -1)
            x = th.cat([x, cat.squeeze()], dim=-1)
            probs = self.classification_head(x)
            return probs
        elif len(x.shape) == 2:
            x = th.unsqueeze(x, 0)
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.classification_head(x)
        elif len(x.shape) == 3:
            x = th.unsqueeze(x, 0)
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.classification_head(x)
        elif len(x.shape) == 4:
            x = self.feature_map_extraction(x)
            x = th.cat([x, cat], dim=-1)
            return self.classification_head(x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, b, optimizer):
        for epoch in range(self.actor_epochs):
            _, _, logprob, entropy = self.get_action(b['observations'], b['actions'], cat=b['extra_info'])
            self._update(logprob, entropy, b, optimizer, log=epoch == self.actor_epochs - 1)

        return self.update_metrics

    def predict(self, x, cat=None):
        # Check if it's a tensor
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        with th.no_grad():
            prob = self.forward(x, cat)
            action = self.select_action(np.array(prob, dtype='float64').squeeze())
        return self.action_map[action]
