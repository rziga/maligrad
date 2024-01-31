import numpy as np

from maligrad.autograd.engine import Variable, Log

from .base import OpTester


def test_basic():
    shape = (3, 3)
    a = Variable(2*np.ones(shape), True)
    tester = OpTester(
        fcn=Log(),
        inputs=[a,],
        expected_output= np.log(a.data),
        backward_seed=2 * np.ones(shape),
        expected_grads=[
            2 * 1 / a.data, # d/dx(x^b) = b * x^(b-1)a)
            ],
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()