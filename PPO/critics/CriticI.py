import abc


class CriticI(abc.ABC):
    def __init__(self, o_size: int, h_size: int, h_layers: int):
        self.o_size = o_size
        self.h_size = h_size
        self.h_layers = h_layers

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def freeze(self):
        pass

    @abc.abstractmethod
    def unfreeze(self):
        pass
