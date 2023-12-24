import numpy as np

from milligrad.autograd.ops import Pow
from milligrad.autograd.engine import DataNode

from .base import OpTester


def test_basic():
    shape = (3, 3)
    a = DataNode(2*np.ones(shape), True)
    b = DataNode(3*np.ones(shape), True)
    tester = OpTester(
        fcn=Pow(),
        inputs=[a, b],
        expected_output= 8 * np.ones(shape),
        backward_seed=np.ones(shape),
        expected_grads=[
            b.data * a.data**(b.data-1), # d/dx(x^b) = b * x^(b-1)
            a.data**b.data * np.log(a.data), # d/dx(a^x) = a^x * log(a)
            ],
        )
    tester.test_backward()
    tester.test_forward()

def test_broadcast():
    tester = OpTester(
        fcn=Pow(),
        inputs=[
            DataNode(2*np.ones((2, 1, 2)), True),
            DataNode(3*np.ones((2, 1)), True),
            ],
        expected_output=8 * np.ones((2, 2, 2)),
        backward_seed=np.ones((2, 2, 2)),
        expected_grads=[
            2*12*np.ones((2, 1, 2)), # 3 * 2**2
            4*5.545177444479562*np.ones((2, 1)), # 2**3 * log(2)
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_broadcast()