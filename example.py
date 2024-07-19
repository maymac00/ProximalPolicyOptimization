from PPO.critics import Critic
from PPO.actors import SoftmaxActor
from PPO.buffers import Buffer
from PPO.PPOAgent import PPOAgent

import torch as th

if __name__ == "__main__":

    agents = []
    env = None
    for i in range(5):
        agents.append(PPOAgent(
            env,
            SoftmaxActor(81, 7, 256, 2),
            Critic(81, 256, 2),
            Buffer(81, 2500, 500, 0.8, 0.95, th.device('cpu'))
        ))

    dummy_obs = th.rand(81)
    dummy_reward = th.tensor([1.0]*5)
    for step in range(2500):
        actions = th.tensor([agent.actor.get_action(dummy_obs)[0] for i, agent in enumerate(agents)])
        dones = ([True]*5 if step == 2499 else [False]*5)
        for i, agent in enumerate(agents):
            agent.buffer.store(dummy_obs, actions[i], 0.5, dummy_reward[i], 0.5, dones[i])
