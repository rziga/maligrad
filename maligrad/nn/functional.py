import numpy as np
from ..autograd.engine import DataNode

# I want the functions here to be independent of batch dims
# e.g. I want conv to work with img with shape [..., b2, b1, c, h, w]
# and ker with shape [c, d, h, w] if 3d or [c, h, w] if 2d or [c, w] if 1d
# explicit python loops are forbidden :)

def _conv_indices(img_shape, ker_shape, stride, dilation):
    assert len(img_shape) == (ndim := len(ker_shape))
    at_end = tuple(range(-ndim, 0)) # -ndim, , ..., -2, -1
    at_one = tuple(range(-2*ndim, -ndim)) # 1, 2, ..., ndim

    # indices of first possible kernel -> [ndim, *ker.shape]
    seed_indices = np.indices(ker_shape) * np.expand_dims(dilation, at_end)

    # number of steps in each dimension -> [ndim]
    n_steps = (img_shape - dilation * (np.array(ker_shape) - 1) - 1) // stride + 1

    # offsets for the first element in seed indices for all dimesnions 
    #  -> [ndim, *n_steps] or differently, [ndim, *out_shape]
    offsets = np.indices(n_steps) * np.expand_dims(stride, at_end)

    # all indices = seed indices offsetted by all offsets -> [ndim, *n_steps, *ker.shape]
    indices = np.expand_dims(seed_indices, at_one) + np.expand_dims(offsets, at_end)

    return tuple(indices) # [ndim, *out_shape, *ker_shape]

def conv(img: DataNode, ker: DataNode, stride: int | tuple, dilation: int | tuple, ker_batch_dims: int = 0) -> DataNode:
    # TODO: switch ker_batch shape and image batch shape position in output
    # TODO: instead of fixing batch_dims, its better to fix dim of conv directly
    # do this by unsqueezing img[ind] instead of kernel :) 
    # output: [*ker_batch_shape, *img_batch_shape, *conv_shape]
    # first ker_batch_dims in kernel are batch dimension
    dim_diff = img.ndim - (ker.ndim - ker_batch_dims)
    if dim_diff > 0:
        # expand so the non batch dimensions match
        ker = ker.unsqueeze(tuple(range(ker_batch_dims, ker_batch_dims+dim_diff)))

    ind = _conv_indices(img.shape, ker.shape[ker_batch_dims:], stride, dilation)
    batch_ker = ker.unsqueeze(tuple(range(ker_batch_dims, ker_batch_dims + img.ndim)))
    return (img[ind] * batch_ker).sum(tuple(range(-ker.ndim+ker_batch_dims, 0)))

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