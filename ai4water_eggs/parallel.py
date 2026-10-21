"""Shared helpers for configurable training periods and bounded multiprocessing."""

import multiprocessing


def parallel_map(function, tasks, workers=None):
    workers = min(workers or 12, multiprocessing.cpu_count(), len(tasks))
    if workers <= 1:
        return [function(*task) for task in tasks]
    with multiprocessing.Pool(processes=workers) as pool:
        return pool.starmap(function, tasks)
