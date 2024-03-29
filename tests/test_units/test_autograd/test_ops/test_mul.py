import numpy as np

from maligrad.autograd.engine import Variable, Mul

from .base import OpTester


def test_basic():
    shape = (3, 3)
    tester = OpTester(
        fcn=Mul(), 
        inputs=[
            Variable(np.ones(shape), True),
            Variable(2*np.ones(shape), True),
            ],
        expected_output=2*np.ones(shape),
        backward_seed=np.ones(shape),
        expected_grads=[
            2*np.ones(shape),
            np.ones(shape),
            ],
        )
    tester.test_backward()
    tester.test_forward()

def test_broadcast():
    tester = OpTester(
        fcn=Mul(),
        inputs=[
            Variable(2*np.ones((2, 1, 2)), True),
            Variable(3*np.ones((2, 1)), True),
            ],
        expected_output=2*3*np.ones((2, 2, 2)),
        backward_seed=np.ones((2, 2, 2)),
        expected_grads=[
            2*3*np.ones((2, 1, 2)),
            4*2*np.ones((2, 1)),
            ]
        )
    tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
    test_broadcast()