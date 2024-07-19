from .CriticI import CriticI
from torch import nn
from PPO.layers import Linear
from torch.nn import functional as F


class Critic(nn.Module, CriticI):
    def __init__(self, o_size: int, h_size: int, h_layers: int):
        super().__init__()
        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')
        self.hidden = nn.ModuleList(self.hidden)
        self.output = Linear(h_size, 1, act_fn='linear')

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = F.leaky_relu(l(x))
        return self.output(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
