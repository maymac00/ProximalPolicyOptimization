from .PPOAgent import PPOAgent
from .buffers import BufferCat


class PPOAgentExtraInfo(PPOAgent):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def store_transition(self, extra_info, *args, **kwargs):
        if isinstance(self.buffer, BufferCat):
            self.buffer.store(
                extra_info,
                observation=self.obs,
                action=self.action,
                logprob=self.logprob,
                reward=args[0],
                value=self.s_value,
                done=args[1],
            )
        else:
            raise ValueError("This type of buffer does not support extra info. Use BufferCat instead.")
        pass