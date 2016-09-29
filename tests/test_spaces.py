
from rllab.spaces import Product, Discrete, Box
import numpy as np


def test_product_space():
    _ = Product([Discrete(3), Discrete(2)])
    product_space = Product(Discrete(3), Discrete(2))
    sample = product_space.sample()
    assert product_space.contains(sample)


def test_product_space_unflatten_n():
    space = Product([Discrete(3), Discrete(3)])
    np.testing.assert_array_equal(space.flatten((2, 2)), space.flatten_n([(2, 2)])[0])
    np.testing.assert_array_equal(
        space.unflatten(space.flatten((2, 2))),
        space.unflatten_n(space.flatten_n([(2, 2)]))[0]
    )


def test_box():
    space = Box(low=-1, high=1, shape=(2, 2))
    np.testing.assert_array_equal(space.flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
    np.testing.assert_array_equal(space.flatten_n([[[1, 2], [3, 4]]]), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(space.unflatten([1, 2, 3, 4]), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(space.unflatten_n([[1, 2, 3, 4]]), [[[1, 2], [3, 4]]])
