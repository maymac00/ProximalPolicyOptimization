import gymnasium as gym
import numpy as np
import torch

import torch as th
import torch.nn as nn

from PPO import Buffer
from PPO.callbacks.AnnealEntropyCallback import AnnealEntropyCallback
from PPO.critics import ConvCritic
from PPO.actors import ConvSoftmaxActor
from PPO.buffers import BufferCat
from PPO.PPOAgent import PPOAgent
from PPO.lr_schedulers.DefaultLrAnneal import DefaultLrAnneal
from PPO.callbacks.TensorBoardCallback import TensorBoardCallback

from EthicalGatheringGame.presets import large
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame import MAEGG

if __name__ == '__main__':
    # Create the CartPole environment
    large["n_agents"] = 1
    large["we"] = [5, 0]
    large["efficiency"] = [1., 1., 1., 1., 1.]
    large["inequality_mode"] = "tie"
    large["obs_mode"] = "cnn"
    env = MAEGG(**large)
    #env = NormalizeReward(env)

    sample_obs = th.Tensor(env.reset(seed=0)[0][0]['image'])

    # Train the agent
    total_steps = int(5e6)
    batch_size = 2500

    # Agent
    feature_map = [
        nn.Conv2d(1, 16, 5, 2, 2, bias=True),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
    ]

    feature_map = nn.Sequential(*feature_map)


    agent = PPOAgent(
            ConvSoftmaxActor(-1, 4,128, 3, feature_map, sample_obs),
            ConvCritic(-1, 64, 2, feature_map, sample_obs),
            Buffer(tuple(sample_obs.shape), batch_size, 250, 0.8, 0.95, torch.device('cpu'))
    )

    agent.lr_scheduler = DefaultLrAnneal(agent, total_steps // batch_size)
    agent.addCallbacks([
        TensorBoardCallback(agent, "tb_cnn_example_data",1),
        AnnealEntropyCallback(agent, total_steps // batch_size)
    ])


    # Reset the environment to start a new episode
    obs = th.Tensor(env.reset(seed=0)[0][0]['image'])

    for update in range(1, total_steps // batch_size + 1):
        score = 0
        scoress = []
        for step in range(batch_size):
            # env.render()
            action = agent.get_action(obs)
            obs, reward, done, info = env.step([action])
            obs = th.Tensor(obs[0]['image'])
            reward = reward[0]
            done = done[0]
            agent.store_transition(reward, done)

            score += reward
            if done:
                #print(f"Score: {score}")
                obs = th.Tensor(env.reset(seed=0)[0][0]['image'])
                obs = th.tensor(obs, dtype=th.float32)
                scoress.append(score)
                score = 0

        th.Tensor(env.reset(seed=0)[0][0]['image'])
        agent.update(obs)
        print(f"Update {update}, score: {sum(scoress[-10:]) / 10}")

    path = f"example_models/{agent.run_name}"
    agent.save(path)
    # Close the environment
    env.close()
