from torch.utils.tensorboard import SummaryWriter

from .CallbacksI import UpdateCallback

# This is just for the agent. It is not intended to be for logging overall performance of the agent.
class TensorBoardCallback(UpdateCallback):

    def __init__(self, agent, log_dir, freq=1):
        super().__init__(agent)
        self.log_dir = log_dir
        self.freq = freq

        self.writer = SummaryWriter(log_dir=log_dir+"/"+agent.run_name)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.agent.cmd_args).items()])),
        )

    def after_update(self):
        if self.agent.run_metrics["update_count"] % self.freq == 0:
            for k, v in self.agent.actor_metrics.items():
                self.writer.add_scalar("Actor/"+k, v, self.agent.run_metrics["update_count"])
            for k, v in self.agent.critic_metrics.items():
                self.writer.add_scalar("Critic/"+k, v, self.agent.run_metrics["update_count"])
            for k, v in self.agent.global_metrics.items():
                self.writer.add_scalar("Global/"+k, v, self.agent.run_metrics["update_count"])
        pass

    def before_update(self):
        pass

