import argparse

import torch as th
from ..layers import Linear
from ..utils import normalize
from .filters import SoftmaxFilterI
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc


class SoftmaxActorI(abc.ABC, nn.Module):

    def __init__(self, o_size, a_size, h_size, h_layers, action_map=None, action_filter: SoftmaxFilterI =None , **kwargs):
        super().__init__()
        self.o_size = o_size
        self.a_size = a_size
        self.h_size = h_size
        self.h_layers = h_layers
        self.action_map = action_map
        self.action_filter = action_filter

        args = self.argparse().parse_known_args()[0]
        self.__dict__.update(args.__dict__)

        self.entropy_value = self.ent_coef

        if action_map is None:
            self.action_map = {i: i for i in range(a_size)}

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def get_action(self, x, action=None):
        pass

    @abc.abstractmethod
    def get_action_data(self, prob, action=None):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def select_action(self, probs):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def update(self, b, optimizer):
        pass

    def argparse(self):
        parser = argparse.ArgumentParser()
        # Common configuration for any PPO agent
        parser.add_argument("--actor-epochs", type=int, default=20, help="Number of epochs for the actor")
        parser.add_argument("--actor-norm-adv", type=bool, default=True, help="Normalize advantages")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter")
        parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
        parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
        return parser


class SoftmaxActor(SoftmaxActorI):

    def __init__(self, o_size: int, a_size: int, h_size: int, h_layers: int):
        SoftmaxActorI.__init__(self, o_size, a_size, h_size, h_layers)

        self.fully_connected = [None] * h_layers
        self.fully_connected[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.fully_connected)):
            self.fully_connected[i] = Linear(h_size, h_size, act_fn='tanh')

        self.output = nn.Linear(h_size, a_size)
        self.fully_connected = nn.ModuleList(self.fully_connected)

    def forward(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
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
            action = self.select_action(np.array(prob, dtype='float64').squeeze())
        return self.action_map[action]

    def select_action(self, probs):
        """
        Gets a probability distribution and returns an action sampled from it. On this case we sample directly from the
        distribution. This behaviour can be modified through wrappers
        :param probs:
        :return:
        """
        if self.action_filter is not None:
            probs = self.action_filter.select_action(probs)
        return np.random.multinomial(1, probs).argmax()

    def load(self, path):
        self.load_state_dict(th.load(path))


    def update(self, b, optimizer):
        update_metrics = {}

        for epoch in range(self.actor_epochs):
            _, _, logprob, entropy = self.get_action(b['observations'], b['actions'])
            entropy_loss = entropy.mean()

            update_metrics[f"Entropy"] = entropy_loss.detach()

            logratio = logprob - b['logprobs']
            ratio = logratio.exp()
            update_metrics[f"Ratio"] = ratio.mean().detach()

            mb_advantages = b['advantages']
            if self.actor_norm_adv:
                mb_advantages = normalize(mb_advantages)

            actor_loss = mb_advantages * ratio
            update_metrics[f"Actor Loss Non-Clipped"] = actor_loss.mean().detach()

            clipped_ratios = th.clamp(ratio, 1 - self.clip, 1 + self.clip)
            actor_clip_loss = mb_advantages * clipped_ratios

            # Log percent of clipped ratio
            update_metrics[f"Clipped Ratio"] = ((ratio < (1 - self.clip)).sum().item() + (
                        ratio > (1 + self.clip)).sum().item()) / np.prod(ratio.shape)

            # Calculate clip fraction
            actor_loss = th.min(actor_loss, actor_clip_loss).mean()
            update_metrics[f"Actor Loss"] = actor_loss.detach()

            actor_loss = -actor_loss - self.entropy_value * entropy_loss
            update_metrics[f"Actor Loss with Entropy"] = actor_loss.detach()

            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            optimizer.step()
        return update_metrics