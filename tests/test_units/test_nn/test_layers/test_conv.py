import numpy as np
from maligrad.nn.layers import Conv, Variable

def test_forward():
    img_shape = [1, 3, 4, 4]
    #X = np.arange(np.prod(img_shape)).reshape(img_shape)
    X = np.ones(img_shape)
    conv = Conv(3, 16, (3, 3), 1, 1, True)
    conv.ker.data = np.ones_like(conv.ker.data)
    conv.b.data = np.ones_like(conv.b.data)
    out = conv(X)
    
    assert np.allclose(out.data, (27+1)*np.ones((1, 16, 2, 2)))

def test_backward():
    # TODO
    img_shape = [1, 3, 4, 4]
    #X = np.arange(np.prod(img_shape)).reshape(img_shape)
    X = np.ones(img_shape)
    conv = Conv(3, 16, (3, 3), 1, 1, True)
    conv.ker.data = np.ones_like(conv.ker.data)
    conv.b.data = np.ones_like(conv.b.data)
    out = conv(X)
    out.sum().backward()

    #assert np.allclose(out.grad, np.ones((1, 16, 2, 2)))
    #assert np.allclose(conv.ker.grad, )
    #assert np.allclose(conv.b.grad, np.ones((4, 1)).sum(axis=0))
    #assert np.allclose(X.grad, )

if __name__ == "__main__":
    test_forward()
    test_backward()
    #strumbloid