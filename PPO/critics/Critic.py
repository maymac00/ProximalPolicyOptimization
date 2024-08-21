import numpy as np

from .CriticI import CriticI
from torch import nn
from ..layers import Linear
import torch as th
from torch.nn import functional as F


class Critic(CriticI):

    def __init__(self, o_size: int, h_size: int, h_layers: int):
        super().__init__(o_size, h_size, h_layers)
        self.fully_connected = [None] * h_layers
        self.fully_connected[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.fully_connected)):
            self.fully_connected[i] = Linear(h_size, h_size, act_fn='tanh')
        self.fully_connected = nn.ModuleList(self.fully_connected)
        self.output = Linear(h_size, 1, act_fn='linear')

    def forward(self, x):
        for i in range(len(self.fully_connected)):
            l = self.fully_connected[i]
            x = F.leaky_relu(l(x))
        return self.output(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def update(self, b, optimizer):

        update_metrics = {}

        for epoch in range(self.critic_epochs):
            values = self(b['observations']).squeeze()

            # Value clipping
            if self.clip_vloss:
                v_loss_unclipped = (values - b['returns']) ** 2

                v_clipped = (th.clamp(values, b['values'] - self.clip, b['values'] + self.clip) - b['returns']) ** 2
                v_loss_clipped = th.min(v_loss_unclipped, v_clipped)

                # Log percent of clipped ratio
                update_metrics[f"Critic Clipped Ratio"] = ((values < (
                            b['values'] - self.clip)).sum().item() + (values > (b['values'] + self.clip)).sum().item()) / np.prod(values.shape)

                critic_loss = 0.5 * v_loss_clipped.mean()
                update_metrics[f"Critic Loss"] = critic_loss.detach()
            else:
                # No value clipping
                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()
                update_metrics[f"Critic Loss"] = critic_loss.detach()

            optimizer.zero_grad(True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            optimizer.step()
        return update_metrics