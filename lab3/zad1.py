from numba import cuda
import numpy as np
from time import time
from math import ceil, sqrt
import warnings
warnings.filterwarnings('ignore')

@cuda.jit
def prime_kernel(arr, res, num_thread_task):
    index = cuda.grid(1) * num_thread_task
    
    for j in range(num_thread_task):
        index_elem = index + j

        if index_elem >= len(arr): 
            return

        n = arr[index_elem]

        prime = True
        for i in range(2, ceil(sqrt(n)) + 1):
            if n % i == 0 and i != n:
                prime = False
                break

        res[index_elem] = prime

@cuda.jit
def sum_kernel(res, num_primes, num_thread_task):
    index = cuda.grid(1) * num_thread_task
    for i in range(num_thread_task):
        index_elem = index + i
        if index_elem >= len(res): return
        cuda.atomic.add(num_primes, 0, res[index_elem])

def main():
    length = 2**25

    arr = np.arange(1, length+1)
    res = np.zeros(length)
    
    threads_per_block = 1024
    blocks_per_grid = 1024

    num_thread_task = length // (threads_per_block * blocks_per_grid) + 1 
    
    start = time() 
    prime_kernel[blocks_per_grid, threads_per_block](arr, res, num_thread_task)
    end = time()
    time_total = end-start
    print(f'Time: {time_total}s')

    # num_primes = int(np.sum(res))

    num_primes = np.array([0])
    sum_kernel[blocks_per_grid, threads_per_block](res, num_primes, num_thread_task)
    num_primes = num_primes[0]

    print(f'Number of primes in a sequence of {length} numbers is: {num_primes}')

main()
