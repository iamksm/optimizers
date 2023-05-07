import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from optimizers.iteration import CustomGenerator, IteratorException, iterator, sys


class TestSILGen(unittest.TestCase):
    def test_sil_gen_length(self):
        iterable = [1, 2, 3, 4]
        sil_gen = CustomGenerator(iterable)
        self.assertEqual(len(sil_gen), 4)

    def test_sil_gen_getitem(self):
        iterable = [1, 2, 3, 4]
        sil_gen = CustomGenerator(iterable)
        self.assertEqual(sil_gen[0], 1)
        self.assertEqual(sil_gen[1], 2)

    def test_sil_gen_iteration(self):
        iterable = [1, 2, 3, 4]
        sil_gen = CustomGenerator(iterable)
        result = [item for item in sil_gen]
        self.assertEqual(result, [1, 2, 3, 4])

    def test_sil_gen_string_representation(self):
        iterable = [1, 2, 3, 4]
        sil_gen = CustomGenerator(iterable)

        size_of_orig = sys.getsizeof(sil_gen.original_iterable) / (1024 * 1024)
        size_of_gen = sys.getsizeof(iterator(sil_gen.original_iterable)) / (
            1024 * 1024
        )

        expected_result = f"""
        Number of elements: 4\n
        Size on memory\n
        \tSize of original iterable - {round(size_of_orig, 4)} MBs
        \tSize of the generator - {round(size_of_gen, 4)} MBs
        """
        self.assertEqual(str(sil_gen).strip(), expected_result.strip())

    def test_sil_gen_representation(self):
        iterable = np.array([1, 2, 3, 4])
        sil_gen = CustomGenerator(iterable)
        self.assertEqual(repr(sil_gen), str(iterable))

    def test_sil_iterator_raises_exception(self):
        iterable = 1
        with self.assertRaises(IteratorException):
            tuple(iterator(iterable))

    def test_sil_iterator_creates_generator(self):
        iterable = [1, 2, 3, 4]
        gen = iterator(iterable)
        self.assertTrue(hasattr(gen, "__next__"))
        self.assertTrue(hasattr(gen, "__iter__"))
        self.assertListEqual(list(gen), iterable)
