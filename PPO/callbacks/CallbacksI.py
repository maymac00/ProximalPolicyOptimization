from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.agent = None


class UpdateCallback(Callback):
    def __init__(self, agent):
        self.agent = agent
        self.update_metrics = None

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def before_update(self):
        pass