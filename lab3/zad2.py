from numba import cuda
import numpy as np
from time import time
import warnings
warnings.filterwarnings('ignore')

@cuda.jit
def prime_kernel(res, num_procs):
    n = 10**7
    id_ = cuda.grid(1)

    h = 1.0 / n
    sum_ = 0.0
    for i in range(id_ + 1, n + 1, num_procs):
        x = h * (i-0.5)
        sum_ += 4.0 / (1.0 + x**2)
    
    mypi = h * sum_
    res[id_] = mypi

def parallel():
    threads_per_block = 512
    blocks_per_grid = 2048
    num_procs = threads_per_block * blocks_per_grid

    res = np.zeros(num_procs)
    start = time()
    prime_kernel[blocks_per_grid, threads_per_block](res, num_procs)
    pi_approx = np.sum(res)
    end = time()
    total_time = end - start

    print(f'Time taken for parallel version: {total_time}')
    print(f'Approximation for pi is: {pi_approx}')


def sequential():
    start = time()
    n = 10**7
    h = 1.0 / n
    sum_ = 0.0
    for i in range(1, n+1):
        x = h * (i-0.5)
        sum_ += 4.0 / (1.0 + x**2)
    pi_approx = h * sum_
    end = time()
    total_time = end - start
    print(f'Time taken for sequential version: {total_time}')
    print(f'Approximation for pi is: {pi_approx}')

def main():
    # parallel()
    sequential()

main()