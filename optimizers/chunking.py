from typing import Iterable

from optimizers.iteration import CustomGenerator


def iterable_chunker(iterable: Iterable, chunk_size: int):
    """Split into groups of `chunk_size`"""
    iterable = CustomGenerator(iterable)
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]  # noqa
