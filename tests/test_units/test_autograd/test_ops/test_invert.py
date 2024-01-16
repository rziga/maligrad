import numpy as np
import pytest

from maligrad.autograd.engine import Variable, Invert

from .base import OpTester


def test_basic():
    tester = OpTester(
        fcn=Invert(),
        inputs=[
            Variable(np.array([0, 1, 1, 0], dtype=bool), True),
            ], 
        expected_output=np.array([1, 0, 0, 1], dtype=bool), 
        backward_seed=None,
        expected_grads=[
            None,
            ]
        )
    with pytest.raises(AssertionError):
        tester.test_backward()
    tester.test_forward()

if __name__ == "__main__":
    test_basic()
