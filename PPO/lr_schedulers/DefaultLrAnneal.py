from .LRSchedulerI import LRSchedulerI


class DefaultLrAnneal(LRSchedulerI):
    def __init__(self, ppo):
        super().__init__(ppo)

    def step(self):
        update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
        frac = 1.0 - (update - 1.0) / self.ppo.n_updates
        self.ppo.actor_lr = frac * self.ppo.init_args.actor_lr
        self.ppo.critic_lr = frac * self.ppo.init_args.critic_lr

        for ag in self.ppo.agents.values():
            ag.a_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.actor_lr
            ag.c_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.critic_lr
