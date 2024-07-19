import abc
from collections import deque
from .actors.SoftmaxActorI import SoftmaxActorI
import argparse


# TODO: PPO agent has its own system for logging, saving and loading.

class PPOAgentI(abc.ABC):

    def __init__(self, env, actor: SoftmaxActorI, critic, buffer, **kwargs):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.buffer = buffer

        # Algorithm Variables. Cmd arguments must be declared on the constructor
        # From cmd arguments
        self.actor_lr = None
        self.critic_lr = None
        self.gamma = None
        self.ent_coef = None

        # Parse known arguments
        args = self.argparse().parse_known_args()[0]
        self.__dict__.update(args.__dict__)

        # Metrics saved for batch
        self.avg_reward = deque(maxlen=200)
        self.avg_losses = deque(maxlen=200)

        # Non-Parametrized Variables
        self.entropy_value = self.ent_coef
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def store_transition(self, *args, **kwargs):
        pass

    def argparse(self):
        parser = argparse.ArgumentParser()
        # Common configuration for any PPO agent
        parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
        parser.add_argument("--ent_coef", type=float, default=0.04, help="Entropy coefficient")
        parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate")
        parser.add_argument("--critic_lr", type=float, default=0.001, help="Critic learning rate")
        return parser
