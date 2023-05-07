import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import psutil

from optimizers.iteration import CustomGenerator


def thread_and_iterate(
    function,
    iterable: Iterable,
    static_args: Iterable = tuple(),
    thread_count: int = 0,
    wait_time: float = 0.5,
    comp_time: float = 0.5,
):
    """
    :param function: An unexecuted python function that will be executed
        for each item in your iterable. The function should have the item
        as its first parameter (a dynamic arg).

    :param iterable: An iterable containing the elements you want to process.

    :param static_args: Extra arguments to pass to the function that don't change
        for each item (excluding the arg that comes from the iterable).

    :param thread_count: The preferred number of threads to use.

    Leave the below unchanged if unsure.
    :param wait_time: The average wait time for I/O operations.
    :param comp_time: The average CPU computation time.

    Usage example:
    >>> def my_function(item, *args):
    >>>         # Function logic goes here
    >>>     pass

    >>> my_iterable = [1, 2, 3, 4, 5]
    >>> static_args = [arg1, arg2, arg3, ...]
    >>> thread_count = 2
    >>> wait_time = 0.5
    >>> comp_time = 0.5

    thread_and_iterate(
        my_function, my_iterable, static_args, thread_count, wait_time, comp_time
    )
    """

    if float(wait_time + comp_time) != 1.0:
        msg = "wait_time + comp_time should add up to 1.0"
        raise ValueError(msg)

    iterable = CustomGenerator(iterable)

    # Get system CPU count and average load
    cpu_count = psutil.cpu_count()
    avg_load = psutil.getloadavg()[0]
    load_avg_percent = int(avg_load * 100 / cpu_count)

    # Use tools like cProfile to get a exact config for the below times
    wait_time = wait_time  # Average wait time for I/O operations
    comp_time = comp_time  # Average CPU computation time

    # Calculate optimal number of threads
    optimal_threads = int(cpu_count * (1 + wait_time / comp_time))

    with open("/proc/sys/kernel/threads-max", "r") as f:
        total_system_threads = int(f.read().strip())

    running_threads = threading.active_count()
    free_threads = total_system_threads - running_threads
    suggested_number_of_threads = min(free_threads, optimal_threads)
    optimal_threads = suggested_number_of_threads

    # Adjust number of threads based on current system load
    if load_avg_percent > 80:
        optimal_threads = min(4, psutil.cpu_count())

    # Ensure the specified threads are not above the default/suggested number of threads
    if thread_count and thread_count > optimal_threads:
        thread_count = 0

    # Create the threads
    thread_count = thread_count or optimal_threads
    with ThreadPoolExecutor(max_workers=thread_count) as thread_exec:
        if not static_args:
            threads = (thread_exec.submit(function, item) for item in iterable)
        else:
            threads = (
                thread_exec.submit(function, dynamic_arg, *static_args)
                for dynamic_arg in iterable
            )

        results = (item.result() for item in as_completed(threads))
        yield from results
