from .SoftmaxActorI import SoftmaxActorI
import torch as th
from PPO.layers import Linear
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftmaxActor(SoftmaxActorI, nn.Module):

    def __init__(self, o_size: int, a_size: int, h_size: int, h_layers: int):
        SoftmaxActorI.__init__(self, o_size, a_size, h_size, h_layers)
        nn.Module.__init__(self)

        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')

        # self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(h_size, a_size)

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = th.tanh(l(x))
        x = self.output(x)
        return F.softmax(x, dim=-1)

    def get_action(self, x, action=None):
        prob = self.forward(x)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy

    def get_action_data(self, prob, action=None):
        env_action = None
        if action is None:
            act = self.select_action(prob.detach())
            action, env_action = act, self.action_map[act]
            action = th.tensor(action)

        logprob = th.log(prob)
        entropy = -(prob * logprob).sum(-1)
        return env_action, action, logprob, entropy

    def predict(self, x):
        # Check if its a tensor
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        with th.no_grad():
            prob = self.forward(x)
            action, env_action = self.select_action(np.array(prob, dtype='float64').squeeze())
        return env_action

    def select_action(self, probs):
        """
        Gets a probability distribution and returns an action sampled from it. On this case we sample directly from the
        distribution. This behaviour can be modified through wrappers
        :param probs:
        :return:
        """
        return np.random.multinomial(1, probs).argmax()

    def load(self, path):
        raise NotImplementedError
