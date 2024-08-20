from .LRSchedulerI import LRSchedulerI
from ..PPOAgent import PPOAgentI


class DefaultLrAnneal(LRSchedulerI):
    def __init__(self, agent: PPOAgentI, n_updates: int):
        super().__init__(agent, n_updates)

    def step(self):
        frac = 1.0 - (self.step_count - 1.0) / self.n_updates
        self.agent.actor_lr = frac * self.actor_lr0
        self.agent.critic_lr = frac * self.critic_lr0
        self.agent.a_optimizer.param_groups[0]["lr"] = frac * self.actor_lr0
        self.agent.c_optimizer.param_groups[0]["lr"] = frac * self.critic_lr0
