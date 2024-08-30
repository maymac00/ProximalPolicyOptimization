from .Buffer import Buffer
import torch as th

class BufferCat:
    def __init__(self, buffer_instance, extra_info_shape):
        # Store the provided Buffer instance
        self.buffer = buffer_instance
        self.extra_info_shape = extra_info_shape
        # Use the size and device properties of the wrapped Buffer
        self.b_extra_info = th.zeros((self.buffer.size, *self.extra_info_shape)).to(self.buffer.device)

    def store(self, extra_info, *args, **kwargs):
        self.b_extra_info[self.buffer.idx] = extra_info.squeeze()
        # Call the store method of the wrapped Buffer
        self.buffer.store(*args, **kwargs)

    def check_pos(self, idx):
        # Use the Buffer's check_pos method and add extra info
        return self.b_extra_info[idx], self.buffer.check_pos(idx)

    def store_parallel(self, idx, extra_info, *args, **kwargs):
        self.b_extra_info[idx] = extra_info.squeeze()
        # Call the store_parallel method of the wrapped Buffer
        self.buffer.store_parallel(idx, *args, **kwargs)

    def sample(self):
        n_episodes = int(self.buffer.size / self.buffer.max_steps)
        data = self.buffer.sample()
        data["extra_info"] = self.b_extra_info.reshape((n_episodes, self.buffer.max_steps, *self.extra_info_shape))
        return data

    def detach(self):
        self.buffer.detach()
        # Detach the extra info
        self.b_extra_info = self.b_extra_info.detach()

    def __add__(self, other):
        if isinstance(other, BufferCat):
            merged_buffer = self.buffer + other.buffer
            # Create a new BufferCat with the merged Buffer
            merged = BufferCat(merged_buffer, self.extra_info_shape)
            merged.b_extra_info = th.cat((self.b_extra_info, other.b_extra_info), dim=0).to(self.buffer.device)
            return merged
        else:
            raise ValueError("Cannot add BufferCat with other Buffer type")

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.buffer, item)
