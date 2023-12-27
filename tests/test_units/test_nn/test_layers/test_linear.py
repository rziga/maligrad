import numpy as np
from maligrad.nn.layers import Linear, DataNode

def test_forward():
    X = np.arange(12.).reshape(4, 3)
    lin = Linear(3, 1)
    lin.W.data = np.ones_like(lin.W.data)
    lin.b.data = np.ones_like(lin.b.data)
    out = lin(X)
    
    assert np.allclose(out.data, np.array([[3.+1], [12.+1], [21.+1], [30.+1]]))

def test_backward():
    X = DataNode(np.arange(12.).reshape(4, 3), True)
    lin = Linear(3, 1)
    lin.W.data = np.ones_like(lin.W.data)
    lin.b.data = np.ones_like(lin.b.data)
    out = lin(X)
    out.sum().backward() # partial is np.ones((4, 1))

    assert np.allclose(out.grad, np.ones((4, 1)))
    assert np.allclose(lin.W.grad, X.data.T @ np.ones((4, 1)))
    assert np.allclose(lin.b.grad, np.ones((4, 1)).sum(axis=0))
    assert np.allclose(X.grad, np.ones((4, 1)) @ lin.W.data.T)

if __name__ == "__main__":
    test_forward()
    test_backward()