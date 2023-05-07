import sys
from typing import Generator, Iterable

import numpy as np


class IteratorException(Exception):
    """Custom exception for the SILIterator class"""

    ...


def iterator(iterable: Iterable) -> Generator:
    try:
        iterable = (item for item in iter(iterable))
        yield from iterable
    except Exception as e:
        raise IteratorException(e)


class CustomGenerator:
    def __init__(self, iterable: Iterable) -> None:
        self.original_iterable = np.array(tuple(iterable))
        self.index = 0

    def __len__(self):
        return len(self.original_iterable)

    def __getitem__(self, key):
        return self.original_iterable[key]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.original_iterable):
            raise StopIteration

        result = self.original_iterable[self.index]
        self.index += 1
        return result

    def __str__(self):
        size_of_orig = sys.getsizeof(self.original_iterable) / (1024 * 1024)
        size_of_gen = sys.getsizeof(iterator(self.original_iterable)) / (
            1024 * 1024
        )

        return f"""
        Number of elements: {format(len(self), ",")}\n
        Size on memory\n
        \tSize of original iterable - {round(size_of_orig, 4)} MBs
        \tSize of the generator - {round(size_of_gen, 4)} MBs
        """

    def __repr__(self) -> str:
        return str(self.original_iterable)
