import abc

import numpy as np

from .SoftmaxActorI import SoftmaxActorI


class ActionFilterI(abc.ABC):
    def __init__(self, actor: SoftmaxActorI, **kwargs):
        self.actor = actor
        pass

    @abc.abstractmethod
    def select_action(self, probs):
        pass

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.actor, item)


class LowProbFilter(ActionFilterI):
    def __init__(self, actor: SoftmaxActorI, low_prob: float, **kwargs):
        super().__init__(actor, **kwargs)
        self.low_thld = low_prob
        pass

    def select_action(self, probs):
        higher = probs[probs > self.low_thld]
        if len(higher) != len(probs):
            # Softmax it
            softmax_values = np.exp(higher) / np.exp(higher).sum(axis=0)
            cont = 0
            for i, p in enumerate(probs):
                if p > self.low_thld:
                    probs[i] = softmax_values[cont]
                    cont += 1
                else:
                    probs[i] = 0
        return self.actor.select_action(probs)

