
[![Python Version](https://img.shields.io/pypi/pyversions/split_dataset.svg)](https://pypi.org/project/split_dataset)
[![PyPI](https://img.shields.io/pypi/v/split_dataset.svg)](
    https://pypi.python.org/pypi/split_dataset)
[![Tests](https://img.shields.io/github/workflow/status/brainglobe/bg-atlasapi/tests)](
    https://github.com/brainglobe/bg-atlasapi/actions)
[![Coverage Status](https://coveralls.io/repos/github/portugueslab/split_dataset/badge.svg?branch=master)](https://coveralls.io/github/portugueslab/split_dataset?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)



A package for saving and reading large HDF5-based chunked arrays. 

This package has been developed in the [`Portugues lab`](http://www.portugueslab.com) for volumetric calcium imaging data. `split_dataset` is extensivly used in the calcium imaging analysis package [`fimpy`](https://github.com/portugueslab/fimpy); SplitDatasets are saved by the microscope control libraries [`sashimi`](https://github.com/portugueslab/sashimi) and [`brunoise`](https://github.com/portugueslab/brunoise).

[`napari-split-dataset`](https://github.com/portugueslab/napari-split-dataset) support the visualization of SplitDatasets in `napari`

# Features
The package contains the definition of  `SplitDataset` objects, that save large arrays in memory as separate  `.h5` files. Any n of dimensions and block sizes are supported in principle, but the package has been used only with 3D and 4D arrays.
Numpy-style indexing can then be used to retrieve data from a `SplitDataset` object.

# Minimal example
```python
# Load a  SplitDataset via a SplitDataset object:
from split_dataset import SplitDataset 
ds = SplitDataset(path_to_dataset)

# Retrieve data in an interval:
ds[n_start:n_end, :, :, :]
```


# TODO
* support for cropping a `SplitDataset`
* support for resolution and frequency metadata


# History

### 0.4.0 (2021-03-23)
* Added support to use a `SplitDataset` as data in a `napari` layer.

...

### 0.1.0 (2020-05-06)
* First release on PyPI.


Credits
-------

Part of this package was inspired by  [Cookiecutter](https://github.com/audreyr/cookiecutter) and [this](https://github.com/audreyr/cookiecutter-pypackage) template.

.. _`Portugues lab`: 
.. _Cookiecutter: 
.. _this: 
