from .PPOAgent import PPOAgent
from .buffers import BufferCat
from .actors import ConvSoftmaxActorCat
from .critics import ConvCriticCat
from .callbacks import UpdateCallback
import torch as th

class PPOAgentExtraInfo:

    def __init__(self, agent : PPOAgent, extra_info_shape):
        self.agent = agent
        self.extra_info_shape = extra_info_shape

        self.actor = ConvSoftmaxActorCat(agent.actor, extra_info_shape)
        self.buffer = BufferCat(agent.buffer, self.extra_info_shape)
        self.critic = ConvCriticCat(agent.critic, extra_info_shape)

        self.last_cat = None

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.agent, item)

    def get_action(self, obs, cat):
        if len(obs.shape) == 2:
            obs = th.unsqueeze(obs, 0)
        with th.no_grad():
            self.env_action, self.action, self.logprob, _ = self.actor.get_action(obs, cat=cat)
            self.s_value = self.critic(obs, cat)
            self.obs = obs
            self.last_cat = cat
        return self.env_action

    def store_transition(self, r, done, *args, **kwargs):
        self.buffer.store(self.last_cat, self.obs, self.action, self.logprob, r, self.s_value,done)

    def update(self, s0, cat):
        # First, we compute the advantages and returns of the buffer with the critic
        with th.no_grad():
            value_ = self.critic(s0, cat)
            self.buffer.compute_mc(value_.reshape(-1))

        # We get the data from the buffer
        self.buffer.clear()
        b = self.buffer.sample()

        # We update the actor and critic
        actor_metrics = self.actor.update(b, self.a_optimizer)
        critic_metrics = self.critic.update(b, self.c_optimizer)
        self.global_metrics["loss"] = actor_metrics["Actor Loss with Entropy"] + critic_metrics[
            "Critic Loss"] * self.v_coef

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.run_metrics["update_count"] += 1

        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()


