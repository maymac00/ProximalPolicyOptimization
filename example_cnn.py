import gymnasium as gym
import numpy as np
import torch

import torch as th
import torch.nn as nn
from PPO.callbacks.AnnealEntropyCallback import AnnealEntropyCallback
from PPO.critics import ConvCritic
from PPO.actors import ConvSoftmaxActor
from PPO.buffers import Buffer
from PPO.PPOAgent import PPOAgent
from PPO.lr_schedulers.DefaultLrAnneal import DefaultLrAnneal
from PPO.callbacks.TensorBoardCallback import TensorBoardCallback
from PPO.utils import ObsTransformer

if __name__ == '__main__':
    # Create the CartPole environment
    env = gym.make("MiniGrid-SimpleCrossingS11N5-v0", max_episode_steps=250)

    sample_obs = env.reset()[0]
    # this env returns a dict with the key 'image' for the observation and the channel is missplaced
    def transfrom_obs(obs):
        if isinstance(obs, dict):
            obs = obs['image']
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs, dtype=th.float32)
        if len(obs.shape) == 2:
            obs = th.unsqueeze(obs, 0)
            obs = th.unsqueeze(obs, 0)
        elif len(obs.shape) == 3:
            obs = obs.permute(2, 0, 1)
            obs = th.unsqueeze(obs, 0)
        elif len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif len(obs.shape) == 5:
            obs = obs.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError("Observation shape not supported")
        return obs

    ObsTransformer.transform_obs = transfrom_obs


    # Train the agent
    total_steps = int(5e6)
    batch_size = 2500

    # Agent
    feature_map = [
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(32, 16, 3, 1, "same"),
        nn.ReLU(),
        #nn.AvgPool2d(2, 2),
        nn.Flatten()
    ]

    feature_map = nn.Sequential(*feature_map)
    feature_map2 = nn.Sequential(*feature_map)

    agent = PPOAgent(
            ConvSoftmaxActor(-1, 3,128, 3, feature_map, sample_obs),
            ConvCritic(-1, 64, 2, feature_map2, sample_obs),
            Buffer(tuple(ObsTransformer.transform_obs(sample_obs).shape), batch_size, 250, 0.8, 0.95, torch.device('cpu'))
    )

    agent.lr_scheduler = DefaultLrAnneal(agent, total_steps // batch_size)
    agent.addCallbacks([
        TensorBoardCallback(agent, "tb_cnn_example_data",1),
        AnnealEntropyCallback(agent, total_steps // batch_size)
    ])


    # Reset the environment to start a new episode
    obs = env.reset()[0]['image']
    obs = th.tensor(obs, dtype=th.float32)

    for update in range(1, total_steps // batch_size + 1):
        score = 0
        scoress = []
        for step in range(batch_size):
            # env.render()
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            obs = th.tensor(obs['image'], dtype=th.float32)
            done = done or truncated

            agent.store_transition(reward, done)

            score += reward
            if done:
                print(f"Score: {score}")
                obs = env.reset()[0]['image']
                obs = th.tensor(obs, dtype=th.float32)
                scoress.append(score)
                score = 0

        obs = env.reset()[0]
        obs = th.tensor(obs['image'], dtype=th.float32)
        agent.update(obs)
        print(f"Update {update}, score: {sum(scoress[-10:]) / 10}")

    path = f"example_models/{agent.run_name}"
    agent.save(path)
    # Close the environment
    env.close()
