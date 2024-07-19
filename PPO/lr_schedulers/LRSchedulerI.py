import abc


class LRSchedulerI(abc.ABC):
    def __init__(self, ppo):
        self.ppo = ppo

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError("Subclasses should implement this method.")
