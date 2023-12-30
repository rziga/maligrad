import numpy as np
from maligrad.nn.layers import Softmax, DataNode

def test_forward():
    X = DataNode(np.array([-1, 0, 1]))
    act = Softmax()
    out = act(X)

    assert np.allclose(out.data, np.array(np.exp(X.data) / np.exp(X.data).sum(axis=-1, keepdims=True)))

def test_backward():
    X = DataNode(np.array([-1., 0, 1]), requires_grad=True)
    act = Softmax()
    out = act(X)
    out[0].backward()

    sm = out.data
    assert np.allclose(X.grad, np.array([sm[0]*(1-sm[0]), -sm[0]*sm[1], -sm[0]*sm[2]]))

if __name__ == "__main__":
    test_forward()
    test_backward()