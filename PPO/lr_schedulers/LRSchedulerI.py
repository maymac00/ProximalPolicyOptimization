import abc
from ..PPOAgent import PPOAgentI

class LRSchedulerI(abc.ABC):
    def __init__(self, agent: PPOAgentI, n_updates: int):
        self.agent = agent
        self.step_count = 0
        self.n_updates = n_updates
        self.actor_lr0 = agent.actor_lr
        self.critic_lr0 = agent.critic_lr

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def reset(self):
        self.step_count = 0
