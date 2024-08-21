
from .CallbacksI import UpdateCallback

class AnnealEntropyCallback(UpdateCallback):

    def __init__(self, agent, n_updates, base_value=1.0, final_value=0.1, concavity=3.5, type="linear_concave"):
        super().__init__(agent)
        self.concavity = concavity
        self.base_value = base_value
        self.final_value = final_value
        self.type = type
        self.n_updates = n_updates

    def after_update(self):
        update = self.agent.run_metrics["update_count"]
        if self.type == "linear_concave":
            normalized_update = (update - 1.0) / self.n_updates
            complementary_update = 1 - normalized_update
            decay_step = normalized_update ** self.concavity / (
                    normalized_update ** self.concavity + complementary_update ** self.concavity)
            frac = (self.base_value - self.final_value) * (1 - decay_step) + self.final_value
            self.agent.actor.entropy_value = frac * self.agent.actor.ent_coef
        elif self.type == "linear":
            frac = 1.0 - (update - 1.0) / self.n_updates
            self.agent.actor.entropy_value = frac * self.agent.actor.ent_coef
        self.agent.global_metrics["Entropy Coefficient"] = self.agent.actor.entropy_value
        pass

    def before_update(self):
        pass

