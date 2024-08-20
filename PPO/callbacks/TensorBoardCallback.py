from torch.utils.tensorboard import SummaryWriter

from .CallbacksI import UpdateCallback

# This is just for the agent. It is not intended to be for logging overall performance of the agent.
class TensorBoardCallback(UpdateCallback):

    def __init__(self, agent, log_dir, freq=1):
        super().__init__(agent)
        self.log_dir = log_dir
        self.freq = freq

        self.writer = SummaryWriter(log_dir=log_dir+"/"+agent.run_name)
        #TODO: log the parameters of the agent
        #self.writer.add_text

    def after_update(self):
        if self.agent.run_metrics["update_count"] % self.freq == 0:
            for k, v in self.agent.actor_metrics.items():
                self.writer.add_scalar("Actor/"+k, v, self.agent.run_metrics["update_count"])
            for k, v in self.agent.critic_metrics.items():
                self.writer.add_scalar("Critic/"+k, v, self.agent.run_metrics["update_count"])
            self.writer.add_scalar("Global Loss", self.agent.global_loss, self.agent.run_metrics["update_count"])
        pass

    def before_update(self):
        pass

