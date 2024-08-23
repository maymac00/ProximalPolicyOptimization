import argparse
import os
from typing import  Optional

import numpy as np
import torch as th
@th.jit.script
def normalize(x: th.Tensor) -> th.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ObsTransformer:
    """
    Class to transform the observations of the environment. As each environment has different observation shapes and formats,
    we use this class to transform the observations to a format the library accepts. Input/output cases:
    - 2D observations HxW --> CxHxW
    - 3D observations --> BxCxHxW
    - 4D observations --> BxCxHxW
    - 5D observations ExSxCxHxW --> E*SxCxHxW (B=E*S)

    Additionally, if the observations are not tensors (e.g. they have images+extra info), we can use this class to return the image part only.
    Extra info can be concatenated to the feature vector by overwriting the forward method of the actor/critic.
    """
    @classmethod
    def transform_obs(cls, obs):
        return obs