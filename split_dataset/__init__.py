"""Top-level package for Split Dataset."""

__author__ = """Vilim Stih & Luigi Petrucco @portugueslab"""
__version__ = "0.4.2"

from split_dataset.blocks import Blocks
from split_dataset.split_dataset import (
    EmptySplitDataset,
    SplitDataset,
    save_to_split_dataset,
)
