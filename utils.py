import resource
from math import sqrt

def multiply_left(v, A):
    return A.multiply_left(v)

def print_memory_usage():
    """
    Print the memory usage of the current process.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory_mb = usage.ru_maxrss / 1024  # 1 KB = 1024 B
    print("Memory usage (MB):", memory_mb)

def vector_length(vector):
    return sqrt(sum(v_**2 for v_ in vector))