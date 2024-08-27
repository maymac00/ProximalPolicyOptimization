from .critics.CriticI import CriticI
from .buffers.BufferI import BufferI
import abc
from collections import deque
from .actors.SoftmaxActor import SoftmaxActorI
from .callbacks.CallbacksI import Callback, UpdateCallback
import argparse
import torch as th
import time
import os

class PPOAgentI(abc.ABC):

    def __init__(self, actor: SoftmaxActorI, critic: CriticI, buffer: BufferI, **kwargs):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer

        # Algorithm Variables. Cmd arguments must be declared on the constructor
        # From cmd arguments
        self.actor_lr = None
        self.critic_lr = None
        self.gamma = None
        self.ent_coef = None
        self.v_coef = None

        # Parse known arguments
        self.cmd_args = self.argparse().parse_known_args()[0]
        self.__dict__.update(self.cmd_args.__dict__)

        # Metrics saved for batch
        self.avg_reward = deque(maxlen=200)
        self.avg_losses = deque(maxlen=200)

        # Non-Parametrized Variables
        self.entropy_value = self.ent_coef
        self.lr_scheduler = None
        self.callbacks = []
        self.run_name = f"ppo_{time.time()}"

        # Metrics
        self.run_metrics = {
            "update_count": 0,
        }
        self.actor_metrics = {
        }
        self.critic_metrics = {
        }
        self.global_metrics = {
        }

        # Optimizers
        self.a_optimizer = th.optim.Adam(list(self.actor.parameters()), lr=self.actor_lr, eps=1e-5)
        self.c_optimizer = th.optim.Adam(list(self.critic.parameters()), lr=self.critic_lr, eps=1e-5)
        pass

    @abc.abstractmethod
    def update(self, s0):
        pass

    @abc.abstractmethod
    def get_action(self, obs):
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
        parser.add_argument("--actor-lr", type=float, default=0.0006, help="Actor learning rate")
        parser.add_argument("--critic-lr", type=float, default=0.003, help="Critic learning rate")
        parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
        parser.add_argument("--v-coef", type=float, default=0.5, help="Value coefficient")
        return parser

    def addCallbacks(self, callbacks: list[Callback] | Callback):
        # add to the list of callbacks
        if isinstance(callbacks, list):
            self.callbacks.extend(callbacks)
        else:
            self.callbacks.append(callbacks)
        pass

class PPOAgent(PPOAgentI):
    def __init__(self, actor: SoftmaxActorI, critic: CriticI, buffer: BufferI, **kwargs):
        super().__init__(actor, critic, buffer, **kwargs)

        pass

    def update(self, s0):
        """
        Update the agent critic and actor with the buffer data.
        :param s0: Random initial state of the environment
        :return:
        """
        # First, we compute the advantages and returns of the buffer with the critic
        with th.no_grad():
            value_ = self.critic(s0)
            self.buffer.compute_mc(value_.reshape(-1))

        # We get the data from the buffer
        self.buffer.clear()
        b = self.buffer.sample()

        # We update the actor and critic
        self.actor_metrics = self.actor.update(b, self.a_optimizer)
        self.critic_metrics = self.critic.update(b, self.c_optimizer)
        self.global_metrics["loss"] = self.actor_metrics["Actor Loss with Entropy"]  + self.critic_metrics["Critic Loss"] * self.v_coef

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.run_metrics["update_count"] += 1

        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()

    def get_action(self, obs):
        """
        Get an action from the actor
        :param obs: Observation from the environment
        :return: Action
        """
        if len(obs.shape) == 2:
            obs = th.unsqueeze(obs, 0)
        with th.no_grad():
            self.env_action, self.action, self.logprob, _ = self.actor.get_action(obs)
            self.s_value = self.critic(obs)
            self.obs = obs
        return self.env_action

    def store_transition(self, r, done, *args, **kwargs):
        self.buffer.store(
            self.obs,
            self.action,
            self.logprob,
            r,
            self.s_value,
            done
        )
        pass

    def save(self, path : str, save_critic=False):
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        th.save(self.actor.state_dict(), f"{path}/actor.pth")
        if save_critic:
            th.save(self.critic.state_dict(), f"{path}/critic.pth")
        pass