import random

from PPO.critics import Critic
from PPO.actors import SoftmaxActor
from PPO.buffers import Buffer
from PPO.PPOAgent import PPOAgent
from PPO.lr_schedulers.DefaultLrAnneal import DefaultLrAnneal
from PPO.callbacks.TensorBoardCallback import TensorBoardCallback
import gym
import torch as th

if __name__ == "__main__":
    # CartPole-v1 is a simple environment to test the agent
    env = gym.make("CartPole-v1")


    agent = PPOAgent(
            SoftmaxActor(4,2, 256, 2),
            Critic(4, 256, 2),
            Buffer(4, 2500, 500, 0.8, 0.95, th.device('cpu'))
    )

    # Train the agent
    scoress = []
    total_steps = int(2e6)
    batch_size = 2500

    agent.lr_scheduler = DefaultLrAnneal(agent, total_steps // batch_size)
    agent.addCallbacks([
        TensorBoardCallback(agent, "example_data",1)
    ])

    obs = env.reset()[0]
    obs = th.tensor(obs, dtype=th.float32)
    score = 0
    for update in range(1, total_steps // batch_size + 1):
        for step in range(batch_size):
            # env.render()
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            obs = th.tensor(obs, dtype=th.float32)
            done = done or truncated

            agent.store_transition(reward, done)

            score += reward
            if done:
                obs = env.reset()[0]
                obs = th.tensor(obs, dtype=th.float32)
                scoress.append(score)
                score = 0

        obs = env.reset()[0]
        obs = th.tensor(obs, dtype=th.float32)
        agent.update(obs)
        print(f"Update {update}, score: {sum(scoress[-10:]) / 10}")

    import matplotlib.pyplot as plt
    plt.plot(scoress)
    plt.show()
