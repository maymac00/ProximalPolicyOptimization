import torch as th
from torch import Tensor
import abc


@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    values_ = th.cat((b_values[1:], value_))
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t] = deltas[t] + gamma[t] * gae_lambda * last_gaelambda

    returns = advantages + b_values

    return returns, advantages


class BufferI(abc.ABC):
    """
    Class for the Buffer creation
    """

    def __init__(self, o_size: int, size: int, max_steps: int, gamma: float, gae_lambda: float, device: th.device):
        self.size = size
        self.o_size = o_size

        # Assuming all episodes last for max_steps steps; otherwise fix sampling
        self.max_steps = max_steps

        # Take agents' observation space; discrete actions have size 1
        a_size = 1
        self.idx = 0

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device

    @abc.abstractmethod
    def store(self, observation, action, logprob, reward, value, done):
        pass

    @abc.abstractmethod
    def check_pos(self, idx):
        pass

    # Store function prepared for parallelized simulations
    @abc.abstractmethod
    def store_parallel(self, idx, observation, action, logprob, reward, value, done):
        """
        Store function prepared for parallelized simulations
        :param idx:
        :param observation:
        :param action:
        :param logprob:
        :param reward:
        :param value:
        :param done:
        :return:
        """
        pass

    @abc.abstractmethod
    def compute_mc(self, value_):
        pass

    @abc.abstractmethod
    def sample(self):
        pass

    def clear(self):
        self.idx = 0

    @abc.abstractmethod
    def detach(self):
        """
        Detach the tensors from the computation graph
        :return:
        """
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        pass
