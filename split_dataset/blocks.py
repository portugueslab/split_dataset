import numpy as np
from itertools import product
from typing import Union, Tuple, Optional


class BlockIterator:
    def __init__(self, blocks, slices=True):
        self.blocks = blocks
        self.current_block = 0
        self.slices = slices

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_block == self.blocks.n_blocks:
            raise StopIteration
        else:
            idx = self.blocks.linear_to_cartesian(self.current_block)
            self.current_block += 1
            if self.slices:
                return (
                    idx,
                    tuple(
                        slice(s, e)
                        for s, e in zip(
                            self.blocks.block_starts[idx], self.blocks.block_ends[idx]
                        )
                    ),
                )
            else:
                return (
                    idx,
                    tuple(
                        (s, e)
                        for s, e in zip(
                            self.blocks.block_starts[idx], self.blocks.block_ends[idx]
                        )
                    ),
                )


def _make_iterable(input_var, n_rep=1):
    try:
        iter(input_var)
        return input_var
    except TypeError:
        return (input_var,) * n_rep


class Blocks:
    """
    Blocks have two indexing systems:
     - linear:
     - cartesian: gives the position of the block in the general block tiling.
    """

    def __init__(
        self,
        shape_full: Tuple,
        shape_block: Optional[Tuple] = None,
        dim_split: Optional[int] = None,
        blocks_number: Optional[int] = None,
        padding: Union[int, Tuple] = 0,
        crop: Optional[Tuple] = None,
    ):
        """ Make a block structure. It can be defined using block size or number
        of blocks (number of blocks if specified will overwrite size).
        For example, one split over the 2nd and 3rd dimensions of a 100x20x40x10 block can
        equivalently defined as:
        BlockSplitter((100,20,40,10), block_size=(10,10,20,30))
        BlockSplitter((100,20,40,10), blocks_number=(1, 2, 2, 1))
        BlockSplitter((100,20,40,10), dim_split=(1,2), block_size=(10,20))
        BlockSplitter((100,20,40,10), dim_split=(1,2), blocks_number=(2,2))

        :param shape_full: dimensions of the whole stack
        :param dim_split: dimension along which to split (if undefined, start
           counting from the first dimension)
        :param shape_block: size of blocks along each dimension
        :param blocks_number: number of blocks along each dimension
        :param padding: amount of overlap between blocks
        :param crop: iterable of tuples giving the amount of cropping in
            each dimension
        """
        self._shape_full = shape_full

        if crop is None:
            crop = ((0, 0),) * len(shape_full) if shape_full is not None else None

        self._crop = crop
        self.shape_cropped = shape_full

        self.starts = None
        self.block_starts = None
        self.block_ends = None

        self.update_stack_dims()

        # Define shape block and padding allowing multiple input types.

        # Initialize block size as full stack size and 0 padding:
        self._shape_block = list(self.shape_cropped)
        self._padding = [0 for _ in range(len(self.shape_cropped))]

        if not dim_split:
            dim_split = [j for j, d in enumerate(shape_block) if d is not None]

        # Make tuple if single numbers
        self.dim_split = _make_iterable(dim_split)
        shape_block = _make_iterable(shape_block, max(self.dim_split) + 1)
        pad_amount = _make_iterable(padding, max(self.dim_split) + 1)

        if blocks_number:  # define from required number of blocks
            shape_block = []
            blocks_number = _make_iterable(blocks_number, len(self.dim_split))
            for dim, n in zip(self.dim_split, blocks_number):
                shape_block.append(int(np.ceil(self.shape_cropped[dim] / n)))

        for dim in self.dim_split:
            self._shape_block[dim] = min(shape_block[dim], self.shape_cropped[dim])
            self._padding[dim] = pad_amount[dim]

        # set property:
        self.shape_block = tuple(self._shape_block)

    @property
    def n_blocks(self):
        return np.product(self.block_starts.shape[:-1])

    @property
    def n_dims(self):
        return len(self.shape_cropped)

    @property
    def shape_full(self):
        return self._shape_full

    @shape_full.setter
    def shape_full(self, value):
        self._shape_full = value
        self.update_stack_dims()
        self.update_block_structure()

    @property
    def crop(self):
        return self._crop

    @crop.setter
    def crop(self, value):
        if value is None:
            value = ((0, 0),) * len(self.shape_full)
        self._crop = value
        self.update_stack_dims()
        self.update_block_structure()

    @property
    def shape_block(self):
        return self._shape_block

    @shape_block.setter
    def shape_block(self, value):
        self._shape_block = value
        self.update_block_structure()

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value
        self.update_block_structure()

    def update_stack_dims(self):
        """ Update stack dimensions and cropping, if shape_full or cropping
        is changed.
        :return:
        """

        if self.shape_full is not None:
            self.shape_cropped = tuple(
                d - cl - ch for d, (cl, ch) in zip(self.shape_full, self.crop)
            )
            self.starts = tuple(cl for cl, ch in self.crop)

    def update_block_structure(self):
        """
        Update the Blocks structure, e.g. when block
        shape or padding are changed.
        """
        # Cartesian product for generating a list of indexes on every split
        # dimension (i.e., dimensions where int(np.ceil(stack_size / block_size)
        #  is != 1).
        # For example, splitting one time in 2nd and 3rd dims,
        # idx_blocks = (0, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 1, 0).

        # block_starts and block_ends will be arrays of shape
        # (n_blocks_dim0, n_blocks_dim1, n_blocks_dim2 ..., shape_full)
        # by addressing the N-1 dimensions with the index of the block we
        # will get a vector with the starting position of the block on all
        # original dimensions of the full stack.
        if self.shape_block is not None:
            self.block_starts = np.empty(
                tuple(
                    int(np.ceil((stack_size - pad_size) / block_size))
                    for stack_size, block_size, pad_size in zip(
                        self.shape_cropped, self.shape_block, self.padding
                    )
                )
                + (len(self.shape_cropped),),
                dtype=np.int32,
            )
            self.block_ends = np.empty_like(self.block_starts)
            for idx_blocks in product(*(range(s) for s in self.block_starts.shape[:-1])):
                self.block_starts[idx_blocks + (slice(None),)] = [
                    st + i_bd * bs
                    for i_bd, bs, st in zip(idx_blocks, self.shape_block, self.starts)
                ]
                self.block_ends[idx_blocks + (slice(None),)] = [
                    min(maxdim + st, (i_bd + 1) * bs + pd + st)
                    for i_bd, bs, pd, maxdim, st in zip(
                        idx_blocks,
                        self.shape_block,
                        self.padding,
                        self.shape_cropped,
                        self.starts,
                    )
                ]

    def slices(self, as_tuples=False):
        return BlockIterator(self, slices=not as_tuples)

    def linear_to_cartesian(self, lin_idx):
        """
        Convert block linear index into cartesian index.
        Example: in a 3D stack split in 2x2x3 blocks,

        self.linear_to_cartesian(0) = (0,0,0)  # first block
        bs.linear_to_cartesian(11) = (1,1,2)  # last block
        :param lin_idx: block linear index (int)
        :return: block cartesian index (tuple of ints)
        """
        return np.unravel_index(lin_idx, self.block_starts.shape[:-1])

    def cartesian_to_linear(self, ca_idx):
        """
        Convert block cartesian index in linear index.
        Example: in a 3D stack split in 2x2x3 blocks

        self.cartesian_to_linear0,0,0) = 0  # first block
        bs.cartesian_to_linear(1,1,2) = 11  # last block

        :param ca_idx: block cartesian index (tuple of ints)
        :return: block linear index (int)
        """
        return np.ravel_multi_index(ca_idx, self.block_starts.shape[:-1])

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        # TODO make less brittle, support also indexing by tuples

        # TODO decide what should be returned: slices are tricky
        # with multiprocessing
        if isinstance(item, int):
            idx = self.linear_to_cartesian(item)
            return tuple(
                slice(s, e)
                for s, e in zip(self.block_starts[idx], self.block_ends[idx])
            )

    def neighbour_blocks(self, i_block, dims=None):
        """
        Return neighbouring blocks across given dimensions
        :param i_block:
        :param dims:
        :return:
        """
        block_idx = self.linear_to_cartesian(i_block)
        act_dims = np.ones(self.n_dims, dtype=np.bool)
        if dims is not None:
            act_dims[dims] = True

        neighbors = []
        for idx_neighbour in product(
            *[
                (
                    range(
                        max(block_idx[i_dim] - 1, 0),
                        min(block_idx[i_dim] + 1, self.block_starts.shape[i_dim]),
                    )
                    if act_dims[i_dim]
                    else [block_idx[i_dim]]
                )
                for i_dim in range(self.n_dims)
            ]
        ):
            if idx_neighbour != block_idx:
                neighbors.append(idx_neighbour)
        if neighbors:
            return np.ravel_multi_index(
                np.stack(neighbors, 1), self.block_starts.shape[:-1]
            )
        else:
            return np.array([])

    def blocks_to_take(self, start_take, end_take):
        """
        Find which blocks to take to cover the range:
        :param start_take: starting points in the N dims (tuple)
        :param end_take: ending points in the N dims (tuple)
        :return: tuple of tuples with the extremes of blocks to take in N dims;
                 starting index of data in the first block;
                 ending index of data in the last block.
        """
        # n_dims = len(start_take)
        block_slices = []
        take_block_s_idx = []
        take_block_e_idx = []
        for i_dim, (start, end) in enumerate(zip(start_take, end_take)):
            axis_index = tuple(
                0 if i != i_dim else slice(None) for i in range(self.n_dims)
            ) + (i_dim,)
            s = max(
                0,
                min(
                    np.searchsorted(self.block_starts[axis_index], start) - 1,
                    len(self.block_starts[axis_index]) - 1,
                ),
            )
            e = np.searchsorted(self.block_starts[axis_index], end)
            block_start = start - self.block_starts[axis_index][s]
            block_end = end - self.block_starts[axis_index][e - 1]

            block_slices.append((s, e))
            take_block_s_idx.append(block_start)
            take_block_e_idx.append(block_end)
        return block_slices, take_block_s_idx, take_block_e_idx

    @staticmethod
    def block_to_slices(block):
        return tuple(slice(lb, rb) for lb, rb in block)

    def centres(self):
        return (self.block_ends + self.block_starts) / 2

    def block_containing_coords(self, coords):
        """
        Find the linear index of a block containing the given coordinates

        :param coords: a tuple of the coordinates
        :return:
        """
        dims = []
        for ic, c in enumerate(coords):
            # Create a tuple with the starting points on current dimension
            # for all the blocks:
            starts = self.block_starts[
                tuple(slice(None) if i == ic else 0 for i in range(self.n_dims)) + (ic,)
            ]

            # find in which position our guy should be ordered, correcting
            # for 0 value:
            dims.append(max((np.searchsorted(starts, c)) - 1, 0))
        return dims

    def drop_dim(self, dim_to_drop):
        """
        Return a new BlockSplitter object with a dimension dropped,
        useful for getting spatial from spatio-temporal blocks.

        :param dim_to_drop: dimension to be dropped (int)
        :return: new BlockSplitter object
        """
        drop_ith = lambda xs: tuple(x for i, x in enumerate(xs) if i != dim_to_drop)
        return Blocks(
            drop_ith(self.shape_full),
            shape_block=drop_ith(self.shape_block),
            padding=drop_ith(self.padding),
            crop=drop_ith(self.crop),
        )

    def serialize(self):
        """
        Returns a dictionary with a complete description of the
        BlockSplitter, e.g. to save its structure as json file.
        :return:
        """
        # TODO it should be possible to initialize the BlockSplitter from
        # this dictionary!
        return dict(
            shape_full=self.shape_full,
            shape_block=self.shape_block,
            crop_start=tuple(c[0] for c in self.crop),
            crop_end=tuple(c[1] for c in self.crop),
            padding=self.padding,
        )
