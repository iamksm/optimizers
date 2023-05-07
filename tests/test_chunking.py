from typing import Generator

from optimizers.chunking import CustomGenerator, iterable_chunker


def test_sil_chunker():
    number_of_elements = 1_000_000
    chunk_size = 1000
    iterable = (x for x in range(number_of_elements))
    chunks = iterable_chunker(iterable, chunk_size)

    assert isinstance(chunks, Generator)

    number_of_groups = number_of_elements / chunk_size
    assert len(CustomGenerator(chunks)) == number_of_groups
