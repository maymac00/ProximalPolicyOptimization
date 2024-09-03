from collections import deque

import numpy as np
import torch
import torch as th
import torch.nn as nn
from PPO import Buffer, PPOAgentExtraInfo
from PPO.critics import ConvCritic
from PPO.actors import ConvSoftmaxActor, ConvSoftmaxActorCat
from PPO.PPOAgent import PPOAgent
from PPO.lr_schedulers.DefaultLrAnneal import DefaultLrAnneal
from PPO.callbacks import TensorBoardCallback, AnnealEntropyCallback

from EthicalGatheringGame.presets import large
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame import MAEGG
import matplotlib

matplotlib.use('TkAgg')

if __name__ == '__main__':
    # Create the CartPole environment
    large["n_agents"] = 1
    large["we"] = [1, 3]
    large["efficiency"] = [1., 1., 1., 1., 1.]
    large["inequality_mode"] = "tie"
    large["obs_mode"] = "cnn"
    env = MAEGG(**large)
    #env = NormalizeReward(env)

    obs = env.reset(seed=0)[0][0]
    sample_obs = th.Tensor(obs['image'])
    recent_action = deque([4.0 / 6] * 5, maxlen=5)
    sample_extra_info = th.Tensor([obs['donation_box'] , obs['survival_status'], *recent_action])


    # Train the agent
    total_steps = int(2e6)
    batch_size = 2500

    # Agent
    feature_map = [
        nn.Conv2d(1, 16, 3, 1, "same", bias=True),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
    ]

    feature_map = nn.ModuleList(feature_map)

    agent = PPOAgent(
        ConvSoftmaxActor(-1, 7, 128, 3, feature_map, sample_obs),
        ConvCritic(-1, 64, 2, feature_map, sample_obs),
        Buffer(tuple(sample_obs.shape), batch_size, 250, 0.8, 0.95, torch.device('cpu'))
    )

    agent = PPOAgentExtraInfo(agent, sample_extra_info.shape)

    agent.lr_scheduler = DefaultLrAnneal(agent, total_steps // batch_size)
    agent.addCallbacks([
        TensorBoardCallback(agent, "tb_cnn_example_data", 1),
        AnnealEntropyCallback(agent, total_steps // batch_size)
    ])

    # Test resize
    agent.buffer.resize(500)
    agent.buffer.resize(batch_size)

    # Reset the environment to start a new episode
    state = env.reset(seed=0)[0][0]
    obs = th.Tensor(state['image'])
    extra_info = th.Tensor([state['donation_box'] , state['survival_status'], *recent_action])

    for update in range(1, total_steps // batch_size + 1):
        score = 0
        scoress = []
        for step in range(batch_size):
            if update % 100 == 0 and step < 500:
                env.render(mode="partial_observability")
            action = agent.get_action(obs, cat=extra_info)
            state, reward, done, info = env.step([action])
            recent_action.append(action/6)
            state = state[0] # We are on single agent
            reward = reward[0]
            done = done[0]

            obs = th.Tensor(state['image'])
            extra_info = th.Tensor([state['donation_box'], state['survival_status'], *recent_action])
            agent.store_transition(reward, done)

            score += reward
            if done:
                #print(f"Score: {score}")
                recent_action = deque([4.0 / 6] * 5, maxlen=5)
                state = env.reset(seed=0)[0][0]
                obs = th.Tensor(state['image'])
                extra_info = th.Tensor([state['donation_box'] , state['survival_status'], *recent_action])
                scoress.append(score)
                score = 0

        state = env.reset(seed=0)[0][0]
        obs = th.Tensor(state['image'])
        recent_action = deque([4.0 / 6] * 5, maxlen=5)
        extra_info = th.Tensor([state['donation_box'], state['survival_status'], *recent_action])
        agent.update(obs, cat=extra_info)
        print(f"Update {update}, score: {sum(scoress[-10:]) / 10}")

    path = f"example_models/{agent.run_name}"
    agent.save(path)
    # Close the environment
    env.close()
