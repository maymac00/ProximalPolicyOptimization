from .PPOAgentI import PPOAgentI
from .actors.SoftmaxActorI import SoftmaxActorI
from .critics.CriticI import CriticI
from .buffers.BufferI import BufferI


class PPOAgent(PPOAgentI):
    def __init__(self, env, actor: SoftmaxActorI, critic: CriticI, buffer: BufferI, **kwargs):
        super().__init__(env, actor, critic, buffer, **kwargs)

        pass

    def update(self):
        print("update")
        pass

    def store_transition(self, *args, **kwargs):
        print("store_transition")
        pass

    def save(self, path):
        print("save")
        pass
