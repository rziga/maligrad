import numpy as np
from maligrad.autograd.engine import Variable


# Two tests taken from micrograd:
# https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
# To avoid using pytorch, I just ran the tests and collected the expected results.

def test_micrograd_sanity_check():
    x = Variable(-4.0, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.maximum(0) + z * x
    h = (z * z).maximum(0)
    y = h + q + q * x
    y.backward()

    assert y.data == -20.
    assert x.grad == 46.

def test_micrograd_more_ops():
    a = Variable(-4.0, requires_grad=True)
    b = Variable(2.0, requires_grad=True)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).maximum(0)
    d += 3 * d + (b - a).maximum(0)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    
    assert np.allclose(a.grad, 138.83381924198252)
    assert np.allclose(b.grad, 645.5772594752186)
    assert np.allclose(g.data, 24.70408163265306)

# TODO: Some of my own computation graphs
# focusing on array specifics (indexing, broadcasting, ...)