import numpy as np
from maligrad.nn.functional import conv, DataNode

def test_same_ndim():
    img = DataNode(np.ones((4, 4)))
    ker = DataNode(np.ones((3, 3)))
    response = conv(img, ker, 2, 1, 1)

    assert np.allclose(response.data, 9*np.ones((2, 2)))

def test_smaller_kernel():
    img = DataNode(np.ones((3, 4, 4)))
    ker = DataNode(np.ones((3, 3)))
    response = conv(img, ker, 2, 1, 1)

    assert np.allclose(response.data, 9*np.ones((3, 2, 2)))

def test_batch_images():
    img = DataNode(np.ones((3, 1, 4, 4)))
    ker = DataNode(np.ones((3, 3)))
    response = conv(img, ker, 2, 1, 1)

    assert np.allclose(response.data, 9*np.ones((3, 1, 2, 2)))

def test_batch_kernels():
    img = DataNode(np.ones((4, 4)))
    ker = DataNode(np.ones((3, 3, 3)))
    response = conv(img, ker, 2, 1, 1)

    assert np.allclose(response.data, 9*np.ones((3, 2, 2)))

def test_batch_images_and_kernels():
    img = DataNode(np.ones((2, 1, 1, 4, 4)))
    ker = DataNode(np.ones((3, 2, 3, 3)))
    response = conv(img, ker, 2, 1, 1)
    # (3, 2) - ker batch, (2, 1, 1) - img batch, (2, 2) - conv result shape
    assert np.allclose(response.data, 9*np.ones((2, 1, 1, 3, 2, 2, 2)))

if __name__ == "__main__":
    test_same_ndim()
    test_smaller_kernel()
    test_batch_images()
    test_batch_kernels()
    test_batch_images_and_kernels()
