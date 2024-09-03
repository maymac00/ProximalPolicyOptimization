from .Buffer import Buffer
import torch as th

class BufferCat(Buffer):
    def __init__(self, extra_info_shape, o_size: int | tuple[int], size: int, max_steps: int, gamma: float,
                 gae_lambda: float, device: th.device):
        # Store the provided Buffer instance
        super().__init__(o_size, size, max_steps, gamma, gae_lambda, device)
        self.extra_info_shape = extra_info_shape
        # Use the size and device properties of the wrapped Buffer
        self.b_extra_info = th.zeros((self.size, *self.extra_info_shape)).to(self.device)

    def store(self, extra_info, *args, **kwargs):
        self.b_extra_info[self.idx] = extra_info.squeeze()
        # Call the store method of the wrapped Buffer
        super().store(*args, **kwargs)

    def check_pos(self, idx):
        # Use the Buffer's check_pos method and add extra info
        return self.b_extra_info[idx], super().check_pos(idx)

    def store_parallel(self, idx, extra_info, *args, **kwargs):
        self.b_extra_info[idx] = extra_info.squeeze()
        # Call the store_parallel method of the wrapped Buffer
        super().store_parallel(idx, *args, **kwargs)

    def sample(self):
        n_episodes = int(self.size / self.max_steps)
        data = super().sample()
        data["extra_info"] = self.b_extra_info.reshape((n_episodes, self.max_steps, *self.extra_info_shape))
        return data

    def detach(self):
        super().detach()
        # Detach the extra info
        self.b_extra_info = self.b_extra_info.detach()

    def __add__(self, other):
        if isinstance(other, BufferCat):
            merged_buffer = super().__add__(other)
            # Create a new BufferCat with the merged Buffer
            merged_buffer.b_extra_info = th.cat((self.b_extra_info, other.b_extra_info), dim=0).to(self.device)
            return merged_buffer
        else:
            raise ValueError("Cannot add BufferCat with other Buffer type")

    def resize(self, new_size):
        super().resize(new_size)
        self.b_extra_info = th.zeros((self.size, *self.extra_info_shape)).to(self.device)
