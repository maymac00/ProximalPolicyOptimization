from .Buffer import Buffer
import torch as th

class BufferCat(Buffer):

    def __init__(self, extra_info_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_info_shape = extra_info_shape
        self.b_extra_info = th.zeros((self.size, *self.extra_info_shape)).to(self.device)

    def store(self, extra_info, *args, **kwargs):
        self.b_extra_info[self.idx] = extra_info.squeeze()
        super().store(*args, **kwargs)

    def check_pos(self, idx):
        return self.b_extra_info[idx], super().check_pos(idx)

    def store_parallel(self, idx, extra_info, *args, **kwargs):
        self.b_extra_info[idx] = extra_info.squeeze()
        super().store_parallel(idx, *args, **kwargs)

    def sample(self):
        n_episodes = int(self.size / self.max_steps)
        data = super().sample()
        data["extra_info"] = self.b_extra_info.reshape((n_episodes, self.max_steps, *self.extra_info_shape))
        return data

    def detach(self):
        super().detach()
        self.b_extra_info = self.b_extra_info.detach()

    def __add__(self, other):
        merged = super().__add__(other)
        merged.b_extra_info = th.cat((self.b_extra_info, other.b_extra_info), dim=0).to(self.device)
        return merged