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
