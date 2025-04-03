from functools import wraps
import time
from typing import Any, Callable, Tuple
import tracemalloc


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that measures the execution time of a function.

    Args:
    func (Callable[..., Any]): The function to be timed.

    Returns:
    Callable[..., Any]: The wrapped function.
    """
    @wraps(func)
    def timeit_wrapper(*args: Tuple[Any], **kwargs: Any) -> Any:
        start_time = time.process_time_ns()
        result = func(*args, **kwargs)
        end_time = time.process_time_ns()
        total_time_ns = end_time - start_time
        total_time_ms = total_time_ns / float(1000000)
        return result, total_time_ms
    return timeit_wrapper



def memory(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that measures the memory usage of a function
    Args:
    func - the function who's memory is to be measured
    
    Returns:
    the wrapped function and the memory usage in KB
    """
    def memory_wrapper(*args, **kwargs) -> Tuple[Any, float]:
        tracemalloc.start()  
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  

        peak_memory_kb = peak / 1024  
        return result, peak_memory_kb  

    return memory_wrapper
    