import torch as th
import torch.nn as nn


def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = th.nn.init.calculate_gain(act_fn)
    fc = th.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc
