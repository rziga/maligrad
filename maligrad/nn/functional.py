import numpy as np
from ..autograd.engine import Variable

# I want the functions here to be independent of batch dims
# e.g. I want conv to work with img with shape [..., b2, b1, c, h, w]
# and ker with shape [c, d, h, w] if 3d or [c, h, w] if 2d or [c, w] if 1d
# explicit python loops are forbidden :)

def _conv_indices(img_shape, ker_shape, stride, dilation):
    assert len(img_shape) == (ndim := len(ker_shape))
    at_end = tuple(range(-ndim, 0)) # -ndim, , ..., -2, -1
    at_start = tuple(range(-2*ndim, -ndim)) # 1, 2, ..., ndim

    # indices of first possible kernel -> [ndim, *ker.shape]
    seed_indices = np.indices(ker_shape) * np.expand_dims(dilation, at_end)

    # number of steps in each dimension -> [ndim]
    n_steps = (img_shape - dilation * (np.array(ker_shape) - 1) - 1) // stride + 1

    # offsets for the first element in seed indices for all dimesnions 
    #  -> [ndim, *n_steps] or differently, [ndim, *out_shape]
    offsets = np.indices(n_steps) * np.expand_dims(stride, at_end)

    # all indices = seed indices offsetted by all offsets -> [ndim, *n_steps, *ker.shape]
    indices = np.expand_dims(seed_indices, at_start) + np.expand_dims(offsets, at_end)

    return tuple(indices) # [ndim, *out_shape, *ker_shape]

def conv(img: Variable, ker: Variable, dim: int, stride: int | tuple = 1, dilation: int | tuple = 1) -> Variable:
    assert img.ndim >= dim and ker.ndim >= dim
    img, ker = Variable.ensure(img), Variable.ensure(ker)
    img_bdim, ker_bdim = img.ndim-dim, ker.ndim-dim

    ker = ker.unsqueeze(tuple(range(ker_bdim, ker_bdim + img.ndim)))
    inds = _conv_indices(img.shape, ker.shape[-img.ndim:], stride, dilation)
    windows = img[inds].unsqueeze(tuple(range(img_bdim, img_bdim + ker_bdim)))
    
    return (windows * ker).sum(tuple(range(-img.ndim, 0)))

def _pool(img: Variable, ker_shape: tuple, stride: int | tuple = 1, dilation: int | tuple = 1) -> tuple[Variable, tuple]:
    assert (dim_diff := img.ndim - len(ker_shape)) >= 0
    ker_shape_padded = dim_diff * (1, ) + ker_shape
    stride = dim_diff * (1, ) + (stride if isinstance(stride, tuple) else (stride,)*len(ker_shape))
    dilation = dim_diff * (1, ) + (dilation if isinstance(dilation, tuple) else (dilation,)*len(ker_shape))
    inds = _conv_indices(img.shape, ker_shape_padded, stride, dilation)
    return img[inds], ker_shape_padded

def maxpool(img: Variable, ker_shape: tuple, stride: int | tuple = 1, dilation: int | tuple = 1) -> Variable:
    windows, ker_shape_padded = _pool(img, ker_shape, stride, dilation)
    return windows.max(axis=tuple(range(-len(ker_shape_padded), 0)))

def avgpool(img: Variable, ker_shape: tuple, stride: int | tuple = 1, dilation: int | tuple = 1) -> Variable:
    windows, ker_shape_padded = _pool(img, ker_shape, stride, dilation)
    return windows.mean(axis=tuple(range(-len(ker_shape_padded), 0)))

def exp(x: Variable) -> Variable:
    return np.e ** x

def relu(x: Variable) -> Variable:
    return x.maximum(0)

def sigmoid(x: Variable) -> Variable:
    return 1 / (1 + exp(-x))

def softmax(x: Variable, axis: int = -1) -> Variable:
    expx = exp(x)
    return expx / expx.sum(axis, keepdims=True)

def categorical_crossentropy(y: Variable, target: Variable, eps=1e-8):
    return -(y[..., range(target.shape[-1]), target] + eps).log().mean()