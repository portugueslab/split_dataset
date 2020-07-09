#!/usr/bin/env python

"""Tests for `split_dataset` package."""

import unittest
import shutil
import numpy as np
import tempfile
from split_dataset import save_to_split_dataset


class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_SplitDataset(self):
        dims = [(10, 3, 3, 3), (5, 5, 5), (5, 5), (1, 5, 5, 5)]
        block_sizes = [(2, None, None, None), (1, None, 3), (2, None), (None, 2, 5, 5)]
        all_slices = [
            [(slice(3, 8), slice(None))],
            [(slice(0, 1),), (slice(0, 2), slice(0, 1), slice(None))],
            [slice(0, 2)],
            [
                (slice(0, 1),),
                (slice(0, 2), slice(0, 1), slice(None)),
                (0, slice(0, 2), slice(0, 1)),
            ],
        ]

        for i, (di, bs, slices) in enumerate(zip(dims, block_sizes, all_slices)):
            test_data = np.arange(np.product(di)).reshape(di)

            sd = save_to_split_dataset(
                test_data,
                block_size=bs,
                root_name=self.test_dir,
                prefix="te{:02d}".format(i),
            )
            for sl in slices:
                a = sd[sl]
                b = test_data[sl]
                np.testing.assert_equal(
                    a,
                    b,
                    err_msg="Testing "
                    + str(di)
                    + " "
                    + str(sl)
                    + " of shape "
                    + str(a.shape)
                    + " and shape"
                    + str(b.shape),
                )

    def test_dask(self):
        di = (100, 100)
        bs = (10, 10)
        test_data = np.arange(np.product(di)).reshape(di)

        sd = save_to_split_dataset(
            test_data,
            block_size=bs,
            root_name=self.test_dir
        )

        sd_full = sd[:, :]
        sd_dask = np.array(sd.as_dask())
        np.testing.assert_equal(sd_full, sd_dask)
