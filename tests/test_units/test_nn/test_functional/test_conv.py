import numpy as np
import pytest
from maligrad.nn.functional import conv, DataNode

def test_same_ndim():
    img = DataNode(np.ones((4, 4)))
    ker = DataNode(np.ones((3, 3)))
    response = conv(img, ker, 1, 1)

    assert all(response.shape == np.array([2, 2]))
    assert np.allclose(response.data, 9*np.ones((2, 2)))

def test_smaller_kernel():
    img = DataNode(np.ones((3, 4, 4)))
    ker = DataNode(np.ones((3, 3)))
    response = conv(img, ker, 1, 1)
    
    assert all(response.shape == np.array([3, 2, 2]))
    assert np.allclose(response.data, 9*np.ones((3, 2, 2)))

def test_larger_kernel():
    img = DataNode(np.ones((4, 4)))
    ker = DataNode(np.ones((4, 3, 3)))
    
    with pytest.raises(AssertionError):
        response = conv(img, ker, 1, 1)

if __name__ == "__main__":
    test_same_ndim()