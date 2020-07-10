import numpy as np
from pathlib import Path
import json
import flammkuchen as fl
from split_dataset.blocks import Blocks
import warnings
from itertools import product
import h5py


# TODO this should probably be done as a constructor of the SplitDataset
def save_to_split_dataset(
    data,
    root_name,
    block_size=None,
    crop=None,
    padding=0,
    prefix="",
    compression="blosc",
):
    """ Function to save block of data into a split_dataset.
    """

    new_name = prefix + ("_cropped" if crop is not None else "")
    padding = (
        data.padding if padding is not None and isinstance(data, Blocks) else padding
    )
    blocks = EmptySplitDataset(
        shape_full=data.shape,
        shape_block=data.shape_block if block_size is None else block_size,
        crop=crop,
        padding=padding,
        root=root_name,
        name=new_name,
    )
    for filename, (idxs, slices) in zip(blocks.files, blocks.slices()):
        fl.save(
            str(blocks.root / filename),
            {"stack_{}D".format(len(blocks.shape_cropped)): data[slices]},
            compression=compression,
        )

    return blocks.finalize()


class SplitDataset(Blocks):
    """
    Manages datasets split over multiple h5 file across arbitrary dimensions.
    To do so, uses the BlockSplitter class functions, and define blocks as
    files.

    """

    def __init__(self, root, prefix=None):
        """
        :param root: The directory containing the files
        :param prefix: The class assumes individual file names to be xxxx.h5. If there is a prefix to this,
        for example if the files are stack_xxxx.h5 this has to be passed to the object as a string, in this
        particular case it would be prefix="stack_"
        """

        # Load information about stack and splitting. Use the json metadata
        # file if possible:
        self.root = Path(root)
        try:
            stack_meta_f = next(self.root.glob("*stack_metadata.json"))

            with open(str(stack_meta_f), "r") as f:
                block_metadata = json.load(f)
        except StopIteration:
            last_data_f = sorted(list(self.root.glob("{}*.h5".format(prefix))))[-1]
            block_metadata = fl.load(str(last_data_f), "/stack_metadata")

            # Ugly keyword fix to handle transition to new json system:
            for new_k, old_k in zip(
                ["shape_block", "shape_full"], ["block_size", "full_size"]
            ):
                block_metadata[new_k] = block_metadata.pop(old_k)

            # By putting this here, we generate the proper stack_metadata
            # file when we open old version data (int conversion for some
            # weird format problem with flammkuchen dictionary):
            clean_metadata = dict()
            _save_metadata_json(block_metadata, self.root)
            for k in block_metadata.keys():
                if isinstance(block_metadata[k], tuple):
                    clean_metadata[k] = tuple(
                        int(n) if n is not None else None for n in block_metadata[k]
                    )
                else:
                    clean_metadata[k] = block_metadata[k]
            with open(str(), "w") as f:
                json.dump(clean_metadata, f)

        # Start the parent BlockSplitter:
        super().__init__(
            shape_full=block_metadata["shape_full"],
            shape_block=block_metadata["shape_block"],
        )

        if prefix != None:
            files = sorted(self.root.glob("*{}_[0-9]*.h5".format(prefix)))
        else:
            files = sorted(self.root.glob("*[0-9]*.h5"))
        self.files = np.array(files).reshape(self.block_starts.shape[:-1])


        # If available, read resolution
        try:
            self.resolution = block_metadata["resolution"]
        except KeyError:
            self.resolution = (1, 1, 1)
        # TODO check this
        self.shape = self.shape_cropped

    @property
    def data_key(self):
        """To migrate smoothly to removal of stack_ND key in favour of only stack
        """
        return [k for k in fl.meta(self.files.flatten()[0]).keys() if "stack" in k][0]

    def __getitem__(self, item):
        """
        Implement usage of the H5SplitDataset as normal numpy array.
        :param item:
        :return:
        """
        # Lot of input munging to emulate indexing in numpy array
        if np.any(self.padding) != 0:
            raise ValueError(
                "Indexing in datasets with overlap (padding) is"
                " not supported, merge them first with an"
                " appropriate merging function"
            )

        if isinstance(item, int):
            item = (slice(item, item + 1),)

        if isinstance(item, slice):
            item = (item,)

        if isinstance(item, tuple):
            # Take care of the case when only the first few dimensions
            # are specified:
            if len(item) < len(self.shape):
                item = item + (None,) * (len(self.shape) - len(item))

            # Loop over dimensions creating a list of starting and ending
            # points

            starts = []
            ends = []
            singletons = np.zeros(len(item), dtype=np.bool)
            for i_dim, (dim_slc, dim_full) in enumerate(zip(item, self.shape)):
                # i_dim: index of current dimension
                # dim_slc: slice/index for current dimension
                # fd: length of dataset on current dimension

                # If nothing specified, start from 0 and finish at end:
                if dim_slc is None:
                    starts.append(0)
                    ends.append(dim_full)

                # If a slice is specified:
                elif isinstance(dim_slc, slice):
                    if dim_slc.start is None:
                        starts.append(0)
                    else:
                        if dim_slc.start >= 0:
                            starts.append(dim_slc.start)
                        else:
                            starts.append(max(0, dim_full + dim_slc.start))

                    if dim_slc.stop is None:
                        ends.append(dim_full)
                    else:
                        if dim_slc.stop >= 0:
                            ends.append(min(dim_slc.stop, dim_full))
                        else:
                            ends.append(max(0, dim_full + dim_slc.stop))
                elif isinstance(dim_slc, int) or isinstance(dim_slc, np.int32):
                    singletons[i_dim] = True
                    if dim_slc >= 0:
                        if dim_slc > dim_full - 1:
                            raise IndexError(
                                "Indexes {} out of dimensions {}!".format(
                                    item, self.shape
                                )
                            )
                        starts.append(dim_slc)
                        ends.append(dim_slc + 1)
                    else:
                        if -dim_slc > dim_full:
                            raise IndexError(
                                "Indexes {} out of dimensions {}!".format(
                                    item, self.shape
                                )
                            )
                        starts.append(dim_full + dim_slc)
                        ends.append(dim_full + dim_slc + 1)
                else:
                    raise IndexError("Unsupported indexing")
        else:
            raise IndexError("Unsupported indexing")

        file_slices, take_block_s_idx, take_block_e_idx = self.blocks_to_take(
            starts, ends
        )
        output_size = tuple(e - s for s, e in zip(starts, ends))

        output = None

        # A lot of indexing tricks to achieve multidimensional generality
        for f_idx in product(*(range(s, e) for s, e in file_slices)):
            abs_idx = [ri - s for ri, (s, e) in zip(f_idx, file_slices)]
            sel_slices = tuple(
                slice(0 if ci != s else si, None if ci < e - 1 else ei)
                for ci, (s, e), si, ei in zip(
                    f_idx, file_slices, take_block_s_idx, take_block_e_idx
                )
            )
            arr = fl.load(
                str(self.files[f_idx]), "/" + self.data_key, sel=fl.aslice[sel_slices],
            )

            if output is None:
                output = np.empty(output_size, arr.dtype)

            output_sel_tuple = tuple(
                slice(
                    0 if st_idx == 0 else bs - first_idx + (st_idx - 1) * (bs),
                    (0 if st_idx == 0 else bs - first_idx + (st_idx - 1) * (bs)) + sz,
                )
                for st_idx, bs, first_idx, sz in zip(
                    abs_idx, self.shape_block, take_block_s_idx, arr.shape
                )
            )
            output[output_sel_tuple] = arr

        if output is None:
            raise IndexError(
                "Trying to index the split dataset outside of bounds, between "
                + str(starts)
                + " and "
                + str(ends)
            )

        output_sel = tuple(0 if singleton else slice(None) for singleton in singletons)

        return output[output_sel]

    def as_dask(self):
        """ Function to create a Dask array from a split dataset.
           :param dataset: SplitDataset object
           :return:
           Dask array
           """
        import dask.array as da

        arrays = np.empty(self.files.shape, dtype=object)

        for s, _ in self.slices():
            arrays[s] = da.from_array(
                h5py.File(self.files[s], mode="r")[f"/{self.data_key}"])

        return da.block(arrays.tolist())

    def apply_crop(self, crop):
        """ Take out the data with a crop

        """
        # TODO there is the crop atrribute, which is a lazy crop, this should actually return a non-cropped dataset
        ds_cropped = EmptySplitDataset(
            shape_full=self.shape,
            shape_block=self.shape_block,
            padding=self.padding,
            crop=crop,
            root=self.root.parent,
            name=self.root.name + "_cropped",
        )
        # the slices iterator does not return just the slices, but also the indicesS
        for (i_slice, block_slices), file_name in zip(
            ds_cropped.slices(), ds_cropped.files
        ):
            fl.save(
                str(self.root / file_name),
                {"stack": self[block_slices]},
            )

        ds_cropped.finalize()


class EmptySplitDataset(Blocks):
    """ Class to initialize an empty dataset for which we have to save metadata
    after filling its blocks.
    """

    def __init__(self, root, name, *args, resolution=None, **kwargs):
        """
        :param root: folder where the stack will be saved;
        :param name: name of the dataset, for the folder name;
        :param resolution: resolution of the stack, in microns;
        """
        super().__init__(*args, **kwargs)
        self.root = Path(root) / name
        if not self.root.is_dir():
            self.root.mkdir(parents=True)
        else:
            warnings.warn('Existing directory')

        self.files = ["{:04d}.h5".format(i) for i in range(self.n_blocks)]
        self.resolution = resolution

    def save_block_data(self, n, data, verbose=False):
        """ Optional method to save data in a block. Often we don't use it,
        as we directly save data in the parallelized function. Might be good to
        find ways of centralizing saving here?
        :param n: n of the block we are saving in;
        :param data: data to be pured in the block;
        :param verbose:
        :return:
        """
        fname = "{:04d}.h5".format(n)
        if verbose:
            print("Saving ", str(self.root / fname))

            if data.shape != self.shape_block:
                print(" - data has different dimension from block!")

        to_save = {f"stack": data}

        fl.save(str(self.root / fname), to_save, compression="blosc")

    def finalize(self):
        n_dims = len(self.shape_block)
        block_dict = self.serialize()
        block_dict["shape_full"] = self.shape_cropped
        block_dict["crop_start"] = (0,) * n_dims
        block_dict["crop_end"] = (0,) * n_dims
        block_dict["resolution"] = self.resolution \
            if self.resolution is not None else (1,) * n_dims

        block_dict["axis_order"] = "tzyx" if n_dims == 4 else "zyx"

        _save_metadata_json(block_dict, self.root)
        return SplitDataset(self.root)


def _save_metadata_json(dictionary, root):
    """ Save json file preventing type failures for stack shapes
    :param path: path for saving
    :param dictionary: dictionary to be saved
    :return:
    """
    METADATA_FILENAME = "stack_metadata.json"
    for k in dictionary.keys():
        if type(dictionary[k]) is tuple:
            # funny fix for variable type mysterious error:
            if type(dictionary[k][0]) == np.int64 or type(dictionary[k][0]) == int:
                dictionary[k] = tuple([int(i) for i in dictionary[k]])

    json.dump(dictionary, open(root / METADATA_FILENAME, "w"))
