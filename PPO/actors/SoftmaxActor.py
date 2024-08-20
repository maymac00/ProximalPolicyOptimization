import torch as th
import torchviz

from PPO.layers import Linear
from PPO.utils import normalize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc


class SoftmaxActorI(abc.ABC, nn.Module):

    def __init__(self, o_size, a_size, h_size, h_layers, action_map=None, **kwargs):
        super().__init__()
        self.o_size = o_size
        self.a_size = a_size
        self.h_size = h_size
        self.h_layers = h_layers
        self.action_map = action_map
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


    def update(self, b, optimizer):
        update_metrics = {}

        # TODO: Parametrize
        self.n_epochs = 10
        self.norm_adv = True
        self.clip = 0.2
        self.log_gradients = False
        self.entropy_value = 0.01
        self.max_grad_norm = 1.0

        for epoch in range(self.n_epochs):
            _, _, logprob, entropy = self.get_action(b['observations'], b['actions'])
            entropy_loss = entropy.mean()

            update_metrics[f"Entropy"] = entropy_loss.detach()

            logratio = logprob - b['logprobs']
            ratio = logratio.exp()
            update_metrics[f"Ratio"] = ratio.mean().detach()

            mb_advantages = b['advantages']
            if self.norm_adv: mb_advantages = normalize(mb_advantages)

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

            if self.log_gradients:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        # update_metrics[f"ZGradients: {name}"] = grad_norm

            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            optimizer.step()
        return update_metrics