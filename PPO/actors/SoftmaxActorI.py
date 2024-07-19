import abc


class SoftmaxActorI(abc.ABC):

    def __init__(self, o_size, a_size, h_size, h_layers, action_map=None, **kwargs):
        self.o_size = o_size
        self.a_size = a_size
        self.h_size = h_size
        self.h_layers = h_layers
        self.action_map = action_map
        if action_map is None:
            self.action_map = {i: i for i in range(a_size)}
        pass

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
