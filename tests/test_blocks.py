import numpy as np
from split_dataset import Blocks


def test_cartesian_blocks():
    test_size = (20, 20)
    a = np.ones(test_size)
    blocks = Blocks(test_size, shape_block=(3, 7), padding=(1, 2))
    for idx, block in blocks.slices():
        a[block] = 0
    np.testing.assert_array_equal(a, np.zeros(test_size))


def test_dropped_dimension():
    test_size = (5, 15, 20)
    blocks = Blocks(
        test_size, shape_block=(3, 7), padding=(1, 2), crop=((1, 1), (0, 0), (0, 0))
    )
    np.testing.assert_equal(blocks.drop_dim(1).shape_full, (5, 20))
