from collections import deque

from EthicalGatheringGame import large, MAEGG
import torch as th
from torch import nn

from PPO.actors import ConvSoftmaxActorCat, ConvSoftmaxActor
import matplotlib

matplotlib.use('TkAgg')

if __name__ == "__main__":
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

    # Agent
    feature_map = [
        nn.Conv2d(1, 16, 3, 1, "same", bias=True),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
    ]

    feature_map = nn.ModuleList(feature_map)

    actor = ConvSoftmaxActor(-1, 7, 128, 3, feature_map, sample_obs)
    actor = ConvSoftmaxActorCat(actor, sample_extra_info.shape)
    path = "example_models/ppo_1725008823.9529746/actor.pth"
    actor.load(path)

    env.toggleTrack(True)
    env.toggleStash(True)

    # Environment loop
    episodes = 100
    for episode in range(episodes):
        state = env.reset(seed=0)[0][0]
        obs = th.Tensor(state['image'])
        recent_action = deque([4.0 / 6] * 5, maxlen=5)
        extra_info = th.Tensor([state['donation_box'], state['survival_status'], *recent_action])
        for step in range(500):
            #env.render()
            action = actor.predict(obs, extra_info)
            state, reward, done, info = env.step([action])
            state = state[0]  # We are on single agent
            reward = reward[0]
            done = done[0]
            obs = th.Tensor(state['image'])
            recent_action.append(action/6)
            extra_info = th.Tensor([state['donation_box'], state['survival_status'], *recent_action])
            if done:
                break

    env.print_results()
    env.plot_results("median")