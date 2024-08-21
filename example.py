import random

from PPO.callbacks.AnnealEntropyCallback import AnnealEntropyCallback
from PPO.critics import Critic
from PPO.actors import SoftmaxActor
from PPO.buffers import Buffer
from PPO.PPOAgent import PPOAgent
from PPO.lr_schedulers.DefaultLrAnneal import DefaultLrAnneal
from PPO.callbacks.TensorBoardCallback import TensorBoardCallback
from PPO.actors.filters import GreedyFilter
import gym
import torch as th

if __name__ == "__main__":


    train = True
    if train:
        # CartPole-v1 is a simple environment to test the agent
        env = gym.make("Acrobot-v1")
        # Normalized environment
        env = gym.wrappers.NormalizeObservation(env)

        # Train the agent
        scoress = []
        total_steps = int(1e6)
        batch_size = 2500

        agent = PPOAgent(
                SoftmaxActor(6,3, 64, 2),
                Critic(6, 64, 2),
                Buffer(6, batch_size, 500, 0.8, 0.95, th.device('cpu'))
        )

        agent.lr_scheduler = DefaultLrAnneal(agent, total_steps // batch_size)
        agent.addCallbacks([
            TensorBoardCallback(agent, "tb_example_data",1),
            AnnealEntropyCallback(agent, total_steps // batch_size)
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
        path = f"example_models/{agent.run_name}"
        agent.save(path)
        import matplotlib.pyplot as plt

        plt.plot(scoress)
        plt.show()

    else:
        env = gym.make("Acrobot-v1", render_mode="human")
        env = gym.wrappers.NormalizeObservation(env)
        path = f"example_models/ppo_1724236070.226411"

        # load the model
        policy = SoftmaxActor(6,3, 64, 2)
        policy.action_filter = GreedyFilter()
        policy.load(path+"/actor.pth")

        # test the model
        obs = env.reset()[0]
        obs = th.tensor(obs, dtype=th.float32)
        score = 0
        done = False
        for _ in range(3000):
            # Render in human mode
            env.render()
            action = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            score += reward
            obs = th.tensor(obs, dtype=th.float32)
            done = done or truncated
            if done:
                obs = env.reset()[0]
                obs = th.tensor(obs, dtype=th.float32)
                print(f"Score: {score}")
                score = 0