from .BufferI import BufferI, compute_gae
import torch as th
from copy import deepcopy


class Buffer(BufferI):
    """
    Class for the Buffer creation
    """

    def __init__(self, o_size: int | tuple[int], size: int, max_steps: int, gamma: float, gae_lambda: float, device: th.device):
        super().__init__(o_size, size, max_steps, gamma, gae_lambda, device)
        a_size = 1
        self.obs_dims = o_size
        if isinstance(o_size, int):
            self.b_observations = th.zeros((self.size, o_size)).squeeze().to(device)
        else:
            s = [self.size] + list(o_size)
            self.b_observations = th.zeros(tuple(s)).squeeze().to(device)
        self.b_actions = th.zeros((self.size, a_size)).to(device)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = deepcopy(self.b_logprobs)
        self.b_values = deepcopy(self.b_logprobs)
        self.b_dones = deepcopy(self.b_logprobs)

    def store(self, observation, action, logprob, reward, value, done, *args, **kwargs):
        """
        Store the data in the buffer
        :param observation:
        :param action:
        :param logprob:
        :param reward:
        :param value:
        :param done:
        :return:
        """
        self.b_observations[self.idx] = observation.squeeze()
        self.b_actions[self.idx] = action
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def check_pos(self, idx):
        return self.b_observations[idx], self.b_actions[idx], self.b_logprobs[idx], self.b_rewards[idx], self.b_values[
            idx], self.b_dones[idx]

    # Store function prepared for parallelized simulations
    def store_parallel(self, idx, observation, action, logprob, reward, value, done):
        self.b_observations[idx] = observation.squeeze()
        self.b_actions[idx] = action
        self.b_logprobs[idx] = logprob
        self.b_rewards[idx] = reward
        self.b_values[idx] = value
        self.b_dones[idx] = done

    def compute_mc(self, value_):
        self.returns, self.advantages = compute_gae(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma,
                                                    self.gae_lambda)

    def sample(self):
        n_episodes = int(self.size / self.max_steps)

        # Mind multidimensional observations. self.b_observations.reshape((n_episodes, self.max_steps, -1))
        obs_dims = self.b_observations.shape[1:]
        # Min dims is 3
        if len(obs_dims) == 2:
            obs_dims = (1, *obs_dims)
            obs = self.b_observations.reshape((n_episodes, self.max_steps, *obs_dims))
        else:
            obs = self.b_observations.reshape((n_episodes, self.max_steps, *obs_dims)).squeeze()
        return {
            'observations': obs,
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'logprobs': self.b_logprobs.reshape((n_episodes, self.max_steps)),
            'values': self.b_values.reshape((n_episodes, self.max_steps)),
            'returns': self.returns.reshape((n_episodes, self.max_steps)),
            'advantages': self.advantages.reshape((n_episodes, self.max_steps)),
        }

    def clear(self):
        self.idx = 0

    def detach(self):
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_logprobs = self.b_logprobs.detach()
        self.b_rewards = self.b_rewards.detach()
        self.b_values = self.b_values.detach()
        self.b_dones = self.b_dones.detach()

    def resize(self, new_size):
        self.__init__(self.obs_dims, new_size, self.max_steps, self.gamma, self.gae_lambda, self.device)

    def __add__(self, other):
        """
        Merge two buffers into a single one.

        Args:
            other (Buffer): Buffer to merge with.

        Returns:
            Buffer: Merged buffer.
        """
        # Get the device of the first buffer
        device = self.device

        # Instantiate the new merged buffer with the appropriate size
        o_size = self.b_observations.shape[-1]
        size = self.size + other.size
        max_steps = self.max_steps
        gamma = self.gamma
        gae_lambda = self.gae_lambda

        merged_buffer = Buffer(o_size, size, max_steps, gamma, gae_lambda, device)

        # Concatenate the tensors
        merged_buffer.b_observations = th.cat([self.b_observations, other.b_observations]).to(device)
        merged_buffer.b_actions = th.cat([self.b_actions, other.b_actions]).to(device)
        merged_buffer.b_logprobs = th.cat([self.b_logprobs, other.b_logprobs]).to(device)
        merged_buffer.b_rewards = th.cat([self.b_rewards, other.b_rewards]).to(device)
        merged_buffer.b_values = th.cat([self.b_values, other.b_values]).to(device)
        merged_buffer.b_dones = th.cat([self.b_dones, other.b_dones]).to(device)

        # Handle the indices correctly
        merged_buffer.idx = size

        return merged_buffer

    def __radd__(self, other):
        return self.__add__(other)
