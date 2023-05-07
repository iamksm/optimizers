import unittest
from concurrent.futures import Future
from unittest.mock import Mock, patch

from optimizers.iteration import CustomGenerator
from optimizers.threads import psutil, thread_and_iterate


def my_function_with_exception(item, *args):
    raise ValueError


class TestSilThreadAndIterate(unittest.TestCase):
    def test_iterable_is_SILGen(self):
        iterable = CustomGenerator([1, 2, 3])
        results = set(thread_and_iterate(lambda x: x + 1, iterable))
        self.assertEqual(results, {2, 3, 4})

    def test_wait_time_is_excess(self):
        iterable = CustomGenerator([1, 2, 3])
        with self.assertRaises(ValueError):
            set(thread_and_iterate(lambda x: x + 1, iterable, wait_time=100))

    def test_iterable_is_not_SILGen(self):
        iterable = [1, 2, 3]
        results = set(thread_and_iterate(lambda x: x + 1, iterable))
        self.assertEqual(results, {2, 3, 4})

    def test_static_args(self):
        iterable = CustomGenerator([1, 2, 3])
        static_args = ["foo", "bar"]
        mocked_function = Mock()
        tuple(thread_and_iterate(mocked_function, iterable, static_args))
        mocked_function.assert_any_call(1, "foo", "bar")
        mocked_function.assert_any_call(2, "foo", "bar")
        mocked_function.assert_any_call(3, "foo", "bar")

    def test_thread_count(self):
        iterable = CustomGenerator([1, 2, 3, 4, 5])

        mocked_executor = Mock()
        mocked_executor.__enter__ = Mock(return_value=mocked_executor)
        mocked_executor.__exit__ = Mock(return_value=False)

        futures = [Future() for _ in range(5)]
        for i, future in enumerate(futures, start=1):
            future.set_result(i + 1)

        mocked_executor.submit.side_effect = futures
        with patch(
            "optimizers.threads.ThreadPoolExecutor", return_value=mocked_executor
        ) as mocked_cm:
            results = tuple(
                thread_and_iterate(lambda x: x + 1, iterable, thread_count=2)
            )
            mocked_cm.assert_called_once_with(max_workers=2)
            self.assertListEqual(sorted(results), sorted((2, 3, 4, 5, 6)))

    def test_set_high_thread_count(self):
        iterable = CustomGenerator([1, 2, 3, 4, 5])

        mocked_executor = Mock()
        mocked_executor.__enter__ = Mock(return_value=mocked_executor)
        mocked_executor.__exit__ = Mock(return_value=False)

        futures = [Future() for _ in range(5)]
        for i, future in enumerate(futures, start=1):
            future.set_result(i + 1)

        mocked_executor.submit.side_effect = futures
        with patch(
            "optimizers.threads.ThreadPoolExecutor", return_value=mocked_executor
        ) as mocked_cm:
            results = tuple(
                thread_and_iterate(lambda x: x + 1, iterable, thread_count=1000)
            )
            mocked_cm.assert_called_once_with(max_workers=psutil.cpu_count() * 2)
            self.assertListEqual(sorted(results), sorted((2, 3, 4, 5, 6)))

    @patch("optimizers.threads.psutil.getloadavg")
    def test_set_high_system_load(self, mock_load):
        iterable = CustomGenerator([1, 2, 3, 4, 5])

        mock_load.return_value = (10, 10, 9)

        mocked_executor = Mock()
        mocked_executor.__enter__ = Mock(return_value=mocked_executor)
        mocked_executor.__exit__ = Mock(return_value=False)

        futures = [Future() for _ in range(5)]
        for i, future in enumerate(futures, start=1):
            future.set_result(i + 1)

        mocked_executor.submit.side_effect = futures
        with patch(
            "optimizers.threads.ThreadPoolExecutor", return_value=mocked_executor
        ) as mocked_cm:
            results = tuple(
                thread_and_iterate(lambda x: x + 1, iterable, thread_count=1000)
            )
            optimal_threads = min(4, psutil.cpu_count())
            mocked_cm.assert_called_once_with(max_workers=optimal_threads)
            self.assertListEqual(sorted(results), sorted((2, 3, 4, 5, 6)))

    def test_wait_time_and_comp_time_sum_to_one(self):
        iterable = CustomGenerator([1, 2, 3])
        with self.assertRaises(ValueError):
            tuple(
                thread_and_iterate(
                    my_function_with_exception, iterable, wait_time=0.4, comp_time=0.6
                )
            )
