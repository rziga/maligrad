import numpy as np
from maligrad.nn.layers import ReLU, Variable

def test_forward():
    X = Variable(np.array([-1, 0, 1]))
    act = ReLU()
    out = act(X)

    assert np.allclose(out.data, np.array([0, 0, 1]))

def test_backward():
    X = Variable(np.array([-1, 0, 1.]), requires_grad=True)
    act = ReLU()
    out = act(X)
    out.sum().backward()

    assert np.allclose(X.grad, np.array([0, 1, 1]))

if __name__ == "__main__":
    test_forward()
    test_backward()
