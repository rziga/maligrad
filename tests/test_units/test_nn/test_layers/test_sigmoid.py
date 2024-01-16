import numpy as np
from maligrad.nn.layers import Sigmoid, Variable

def test_forward():
    X = Variable(np.array([-1, 0, 1]))
    act = Sigmoid()
    out = act(X)

    assert np.allclose(out.data, np.array([1/(1+np.exp(+1)), 0.5, 1/(1+np.exp(-1))]))

def test_backward():
    X = Variable(np.array([-1., 0, 1]), requires_grad=True)
    act = Sigmoid()
    out = act(X)
    out.sum().backward()

    assert np.allclose(X.grad, out.data * (1-out.data))

if __name__ == "__main__":
    test_forward()
    test_backward()
