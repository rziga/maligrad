import numpy as np
from ..autograd.engine import DataNode

# I want the functions here to be independent of batch dims
# e.g. I want conv to work with img with shape [..., b2, b1, c, h, w]
# and ker with shape [c, d, h, w] if 3d or [c, h, w] if 2d or [c, w] if 1d
# explicit python loops are forbidden :)

def conv(img: DataNode, ker: DataNode, stride: int | tuple, dilation: int | tuple) -> DataNode:
    assert img.ndim >= ker.ndim,\
        "Cannot convolve image with kernel of higher dimension than itself."
    if img.ndim > ker.ndim: # saves an op when dims match
        ker = ker.unsqueeze(tuple(range(img.ndim - ker.ndim)))

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

def sigmoid(x: DataNode) -> DataNode:
    return 1 / (1 + exp(-x))

def softmax(x: DataNode, axis: int = -1) -> DataNode:
    expx = exp(x)
    return expx / expx.sum(axis, keepdims=True)

def categorical_crossentropy(y: DataNode, target: DataNode):
    # TODO make this batch independent
    return y[range(target.shape[-1]), target]