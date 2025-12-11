"""Core dataset class for indexed multimodal token datasets."""

import os
import sys
from itertools import accumulate
from typing import Optional, Tuple, Union

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from io_utils.readers import IndexReader, MMapBinReader, FileBinReader
from utils.types import DType


class IndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): Whether to mmap the .bin files. Defaults to True.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool = False,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.path_prefix = None
        self.multimodal = None
        self.mmap = None

        self.index = None
        self.bin_reader = None

        self.initialize(path_prefix, multimodal, mmap)

    def initialize(
        self, path_prefix: str, multimodal: bool, mmap: bool
    ) -> None:
        """Initialize the dataset

        This method is called by IndexedDataset.__init__ during object creation and by
        IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal

            mmap (bool): Whether to mmap the .bin file
        """
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"

        assert os.path.exists(idx_path) and os.path.exists(
            bin_path
        ), f"One or both of the .idx and .bin files cannot be found at the path prefix {path_prefix}"

        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.mmap = mmap

        if mmap:
            self.bin_reader = MMapBinReader(bin_path)
        else:
            self.bin_reader = FileBinReader(bin_path)

        self.index = IndexReader(idx_path, self.multimodal)

    def __getstate__(self) -> Tuple[str, bool, bool]:
        """Get the state during pickling

        Returns:
            Tuple[str, bool, bool]: The state tuple
        """
        return self.path_prefix, self.multimodal, self.mmap

    def __setstate__(self, state: Tuple[str, bool, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (Tuple[str, bool, bool]): The state tuple
        """
        path_prefix, multimodal, mmap = state
        self.initialize(path_prefix, multimodal, mmap)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (Union[int, numpy.integer, slice]): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous

            TypeError: When the index is of an unexpected type

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index or index slice
        """
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.

        Args:
            idx (Union[int, numpy.integer]): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (int): The number of tokens to grab from the sequence

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index
        """
        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=length, offset=sequence_pointer
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def get_document_indices(self) -> numpy.ndarray:
        """Get the document indices

        This method is slated for deprecation.

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def set_document_indices(self, document_indices: numpy.ndarray) -> None:
        """Set the document indices

        This method is slated for deprecation.

        Args:
            document_indices (numpy.ndarray): The document indices
        """
        self.index.document_indices = document_indices

    @property
    def sequence_modes(self) -> numpy.ndarray:
        """Get the sequence modes

        Returns:
            numpy.ndarray: The sequence modes
        """
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the IndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the IndexedDataset exists on disk at the prefix
        """
        return os.path.exists(f"{path_prefix}.idx") and os.path.exists(
            f"{path_prefix}.bin"
        )
