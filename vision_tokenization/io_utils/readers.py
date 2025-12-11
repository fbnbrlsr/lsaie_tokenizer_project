"""Reader classes for index and binary files."""

import logging
import struct
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Tuple, Type

import numpy

from utils.types import DType

logger = logging.getLogger(__name__)


INDEX_HEADER = b"MMIDIDX\x00\x00"

class IndexReader:
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:

        print(f"Load the {type(self).__name__} from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        # print(f"\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        # print(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        # print(f"\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        # print(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        # print(f"\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        # print(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        self.sequence_modes = None
        if multimodal:
            # print(f"\tExtract the sequence modes")
            t_beg = time.time()
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()
            # print(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        print(f"> total number of sequences: {len(self)}")
        print(f"> total number of documents: {self.document_indices.shape[0] - 1}")

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        pass


class MMapBinReader(BinReader):
    """A BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap


class FileBinReader(BinReader):
    """A BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode='rb', buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence
