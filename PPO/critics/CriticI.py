import abc
import argparse

from torch import nn


class CriticI(abc.ABC, nn.Module):
    def __init__(self, o_size: int, h_size: int, h_layers: int):
        super().__init__()
        self.o_size = o_size
        self.h_size = h_size
        self.h_layers = h_layers

        args = self.argparse().parse_known_args()[0]
        self.__dict__.update(args.__dict__)

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def update(self, b, optimizer):
        pass

    @abc.abstractmethod
    def freeze(self):
        pass

    @abc.abstractmethod
    def unfreeze(self):
        pass

    def argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--critic_lr", type=float, default=0.001)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--v-coef", type=float, default=0.5)
        parser.add_argument("--clip", type=float, default=0.2)
        parser.add_argument("--critic-epochs", type=int, default=20)
        parser.add_argument("--clip-vloss", type=bool, default=True)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        return parser
