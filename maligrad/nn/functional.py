import numpy as np
from ..autograd.engine import DataNode


def conv(img: DataNode, ker: DataNode, stride: int | tuple, dilation: int | tuple) -> DataNode:
    assert img.ndim == ker.ndim
    at_end = tuple(range(-ker.ndim, 0)) # -ndim, , ..., -2, -1
    at_one = tuple(range(1, ker.ndim+1)) # 1, 2, ..., ndim

    seed_indices = np.indices(ker.shape) * np.expand_dims(dilation, at_end)
    n_steps = (img.shape - dilation * (np.array(ker.shape) - 1) - 1) // stride + 1
    offsets = np.indices(n_steps) * np.expand_dims(stride, at_end)
    indices = np.expand_dims(seed_indices, at_one) + np.expand_dims(offsets, at_end)

    return (img[tuple(indices)] * ker).sum(at_end)

def exp(x: DataNode) -> DataNode:
    return np.e ** x

def relu(x: DataNode) -> DataNode:
    return x.maximum(0)