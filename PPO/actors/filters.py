import abc

import numpy as np


class SoftmaxFilterI(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def select_action(self, probs):
        pass


class LowProbFilter(SoftmaxFilterI):
    def __init__(self, low_prob: float, **kwargs):
        super().__init__(**kwargs)
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
        return probs

class GreedyFilter(SoftmaxFilterI):
    def __init__(self, threshold=0.6, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        pass

    def select_action(self, probs):
        # Set all to 0 except the max if it is above the threshold
        max_index = np.argmax(probs)
        if probs[max_index] > self.threshold:
            probs = np.zeros_like(probs)
            probs[max_index] = 1
        return probs

